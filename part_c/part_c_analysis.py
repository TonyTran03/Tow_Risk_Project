# %%
from pathlib import Path
import os
os.environ['SM_FORMULA_ENGINE'] = 'patsy'
import hashlib, json, platform, sys, unicodedata, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels
import scipy
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import stats
from IPython.display import display, Image

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.set_option('display.max_colwidth', 55)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p/'data/processed/tow_risk_clean.csv').exists()), None)
if ROOT is None:
    ROOT = next((p/'Tow_Risk_Project' for p in candidates if (p/'Tow_Risk_Project/data/processed/tow_risk_clean.csv').exists()), None)
assert ROOT is not None, 'Open the notebook from the project folder or its part_c subfolder.'
OUT = ROOT/'part_c'
for name in ['tables', 'figures', 'handoff']:
    (OUT/name).mkdir(parents=True, exist_ok=True)
SOURCE = ROOT/'data/processed/tow_risk_clean.csv'
data = pd.read_csv(SOURCE, encoding='utf-8-sig', parse_dates=['date']).sort_values(['date','borough']).reset_index(drop=True)
versions = {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__, 'matplotlib': matplotlib.__version__}
print('Packages:', versions)
print('Source SHA256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
display(data.head())

# %%
expected_dates = pd.date_range('2021-01-01','2026-06-30')
boroughs = sorted(data.borough.unique())
assert data.shape == (34119,12)
assert len(boroughs) == 17
assert not data.isna().any().any()
assert not data.duplicated(['date','borough']).any()
assert (data.groupby('date').size() == 17).all()
assert pd.DatetimeIndex(data.date.unique()).equals(expected_dates)
assert (data.groupby('borough').size() == len(expected_dates)).all()
assert data.tows.ge(0).all() and np.allclose(data.tows, np.round(data.tows))
assert data.snow_7c.between(0,30).all()
assert (data.dow == data.date.dt.dayofweek).all()
assert (data.month == data.date.dt.month).all()
assert (data.moving_day == ((data.date.dt.month == 7) & (data.date.dt.day == 1)).astype(int)).all()
assert np.allclose(data.trend, (data.date-pd.Timestamp('2021-01-01')).dt.days/365.25)
weather_cols = ['snow_7c','snow_8_14','snow_1','rain_1','temp_7']
assert (data.groupby('date')[weather_cols].nunique() == 1).all().all()
yearly = data.groupby(data.date.dt.year).tows.agg(['sum','size']).rename(columns={'sum':'tows','size':'borough_days'})
yearly.to_csv(OUT/'tables/data_yearly_totals.csv')
display(yearly)
print(f'Panel checks passed: {len(data):,} rows, {len(boroughs)} boroughs, {len(expected_dates):,} dates.')

# %%
parts = []
for year in range(2020,2027):
    w = pd.read_csv(ROOT/f'montreal_weather_{year}.csv', encoding='utf-8-sig')
    w = w.rename(columns={'Date/Time':'date','Total Snow (cm)':'snow','Total Rain (mm)':'rain','Mean Temp (°C)':'temp'})
    w['date'] = pd.to_datetime(w['date'])
    parts.append(w.loc[w.date.dt.year.eq(year), ['date','snow','rain','temp']])
weather = pd.concat(parts).set_index('date').sort_index()
assert not weather.index.duplicated().any()
weather = weather.reindex(pd.date_range('2020-01-01','2026-06-30'))
for c in ['snow','rain','temp']:
    weather[c] = pd.to_numeric(weather[c],errors='coerce')
missing_weather = weather.isna().sum().to_dict()
weather[['snow','rain']] = weather[['snow','rain']].fillna(0)
rebuilt = pd.DataFrame(index=weather.index)
rebuilt['snow_7c'] = weather.snow.shift(1).rolling(7).sum().clip(upper=30)
rebuilt['snow_8_14'] = weather.snow.shift(8).rolling(7).sum()
rebuilt['snow_1'] = weather.snow.shift(1)
rebuilt['rain_1'] = weather.rain.shift(1)
rebuilt['temp_7'] = weather.temp.interpolate(method='time',limit_direction='both').shift(1).rolling(7).mean()
daily = data.drop_duplicates('date').set_index('date')
assert np.allclose(daily[weather_cols],rebuilt.loc[daily.index,weather_cols])
temp_past = weather.temp.ffill().shift(1).rolling(7).mean()
assert temp_past.loc[daily.index].notna().all()
changed = pd.Series(~np.isclose(daily.temp_7, temp_past.loc[daily.index]),index=daily.index)
temp_audit = pd.DataFrame({'temp_7_supplied':daily.temp_7, 'temp_7_past_only':temp_past.loc[daily.index]})
temp_audit['changed'] = changed
temp_audit.to_csv(OUT/'tables/temperature_feature_audit.csv',index_label='date')
data_original = data.copy()
data['temp_7'] = data.date.map(temp_past)
data.to_csv(OUT/'handoff/model_input.csv',index=False,encoding='utf-8-sig')
print('Missing raw weather values before filling:',missing_weather)
print('All supplied weather features reproduced from raw files.')
print('Temperature feature changed on dates:',int(changed.sum()), 'of',len(daily))
print('Changed train dates:',int(changed.loc[:'2025-06-30'].sum()),'| changed test dates:',int(changed.loc['2025-07-01':].sum()))

def name_key(value):
    if pd.isna(value): return ''
    value = unicodedata.normalize('NFKD',str(value).strip())
    value = ''.join(c for c in value if not unicodedata.combining(c))
    value = value.replace('’',"'").replace('–','-').replace('—','-')
    return re.sub(r'\s+',' ',re.sub(r'\s*-\s*','-',value)).lower().strip()
if (ROOT/'remorquages.csv').exists():
    raw_tow = pd.read_csv(ROOT/'remorquages.csv',usecols=['DATE_ORIGINE','ARRONDISSEMENT_ORIGINE'],dtype=str,encoding='utf-8-sig')
    raw_tow['date'] = pd.to_datetime(raw_tow.DATE_ORIGINE.str.strip().str[:10],format='%Y-%m-%d',errors='coerce')
    mapping = {name_key(b):b for b in boroughs}
    mapping[name_key('Plateau-Mont-Royal')] = 'Le Plateau-Mont-Royal'
    raw_tow['borough'] = raw_tow.ARRONDISSEMENT_ORIGINE.map(lambda x:mapping.get(name_key(x)))
    raw_tow = raw_tow.loc[raw_tow.date.between('2021-01-01','2026-06-30') & raw_tow.borough.notna()]
    raw_counts = raw_tow.groupby(['date','borough']).size().reindex(pd.MultiIndex.from_frame(data[['date','borough']]),fill_value=0)
    assert np.array_equal(raw_counts.to_numpy(),data.tows.to_numpy())
    print('All 34,119 towing counts match an independent aggregation of the raw records.')
    del raw_tow
else:
    print('Raw towing file unavailable: count reconciliation skipped.')

# %%
train = data.loc[data.date <= '2025-06-30'].copy()
test = data.loc[data.date >= '2025-07-01'].copy()
assert train.date.max() < test.date.min()
assert len(train) == 27914 and len(test) == 6205
for frame in [train,test]:
    frame['log_tows'] = np.log1p(frame.tows)
scales = train.assign(error=(train.tows-train.groupby('borough').tows.shift(7)).abs()).groupby('borough').error.mean()
assert scales.gt(0).all()
scales.rename('training_seasonal_naive_mae').to_csv(OUT/'tables/mase_scales.csv')
print('Train:',train.date.min().date(),'to',train.date.max().date(),f'({len(train):,} rows)')
print('Test: ',test.date.min().date(),'to',test.date.max().date(),f'({len(test):,} rows)')
display(scales.to_frame())

# %%
calendar = 'log_tows ~ C(borough) + C(dow) + C(month) + trend + moving_day'
weather_formula = calendar + ' + snow_7c + snow_8_14 + snow_1 + rain_1 + temp_7'
final_formula = weather_formula + ' + C(borough):snow_7c'
formulas = {'M1 Calendar':calendar,'M2 Add weather':weather_formula,'M3 Borough snow slopes':final_formula}
models = {}
score_rows, metric_rows = [], []
predictions = test[['date','borough','tows']].copy()
smear_by_model = {}
for name,formula in formulas.items():
    model = smf.ols(formula,data=train).fit()
    models[name] = model
    eta = model.predict(test)
    point = np.maximum(np.expm1(eta),0)
    smear = pd.DataFrame({'borough':train.borough,'exp_resid':np.exp(model.resid)}).groupby('borough').exp_resid.mean()
    smear_by_model[name] = smear
    mean_point = np.maximum(np.exp(eta)*test.borough.map(smear)-1,0)
    predictions[name] = point
    predictions[name+' smearing'] = mean_point
    errors = test[['borough']].copy()
    errors['abs_error'] = np.abs(test.tows-point)
    errors['smear_abs_error'] = np.abs(test.tows-mean_point)
    by = errors.groupby('borough').mean()
    by['test_mase'] = by.abs_error/scales
    by['test_mase_smearing'] = by.smear_abs_error/scales
    for b,row in by.iterrows():
        score_rows.append({'model':name,'borough':b,'test_mae':row.abs_error,'test_mase':row.test_mase,'test_mase_smearing':row.test_mase_smearing})
    metric_rows.append({'model':name,'training_r2':model.rsquared,'training_adj_r2':model.rsquared_adj,'parameters':len(model.params),'test_mase':by.test_mase.mean(),'test_mae':errors.abs_error.mean(),'test_mase_smearing':by.test_mase_smearing.mean()})
    (OUT/'tables'/f'{name[:2].lower()}_ols_summary.txt').write_text(model.summary().as_text(),encoding='utf-8')
ladder = pd.DataFrame(metric_rows)
borough_scores = pd.DataFrame(score_rows)
ladder.to_csv(OUT/'tables/model_ladder.csv',index=False)
borough_scores.to_csv(OUT/'tables/model_mase_by_borough.csv',index=False,encoding='utf-8-sig')
predictions.to_csv(OUT/'handoff/test_predictions.csv',index=False,encoding='utf-8-sig')
display(ladder.round(4))
print('Primary prediction: clipped expm1. Smearing results are a separate sensitivity, not a replacement chosen after seeing test data.')

# %%
final = models['M3 Borough snow slopes']
time_id = (train.date-train.date.min()).dt.days.to_numpy()
robust14 = final.get_robustcov_results(cov_type='hac-groupsum',time=time_id,maxlags=14,use_correction='cluster',df_correction=False,use_t=False)
robust28 = final.get_robustcov_results(cov_type='hac-groupsum',time=time_id,maxlags=28,use_correction='cluster',df_correction=False,use_t=False)
# Daily score aggregation accepts date-major rows. Use asymptotic normal inference,
# avoiding a panel-count degrees-of-freedom adjustment that assumes borough-major rows.
assert np.isfinite(robust14.conf_int()).all() and np.isfinite(robust28.conf_int()).all()
ci = final.conf_int().to_numpy()
rci = np.asarray(robust14.conf_int())
full_coefs = pd.DataFrame({'term':final.params.index,'estimate':final.params.to_numpy(),'se_ols':final.bse.to_numpy(),'p_ols':final.pvalues.to_numpy(),'ci_low_ols':ci[:,0],'ci_high_ols':ci[:,1],'se_panel14':np.asarray(robust14.bse),'p_panel14':np.asarray(robust14.pvalues),'ci_low_panel14':rci[:,0],'ci_high_panel14':rci[:,1]})
full_coefs.to_csv(OUT/'tables/full_coefficients.csv',index=False,encoding='utf-8-sig')
weather_table = full_coefs.loc[full_coefs.term.isin(weather_cols)].copy()
weather_table['meaning'] = weather_table.term.map({'snow_7c':f'Past 7 days, per cm; reference borough {boroughs[0]}','snow_8_14':'Snow 8–14 days ago, per cm','snow_1':'Yesterday snow, per cm','rain_1':'Yesterday rain, per mm','temp_7':'Past 7 days mean temperature, per degree C'})
weather_table.to_csv(OUT/'tables/weather_coefficients.csv',index=False,encoding='utf-8-sig')
display(weather_table[['term','estimate','p_ols','p_panel14','ci_low_panel14','ci_high_panel14','meaning']].round(5))
fstat, fp, df_diff = final.compare_f_test(models['M2 Add weather'])
interaction_terms = [i for i,p in enumerate(final.params.index) if ':snow_7c' in p]
R = np.eye(len(final.params))[interaction_terms]
joint = robust14.wald_test(R,scalar=True)
joint_test = {'ols_f':float(fstat),'ols_p':float(fp),'added_parameters':int(df_diff),'panel14_wald_p':float(joint.pvalue)}
print('Joint test of equal borough snow slopes:',joint_test)

# %%
effect_rows = []
for b in boroughs:
    x0 = train.iloc[[0]].copy()
    x0['borough'] = b
    x0['snow_7c'] = 0.0
    x1 = x0.copy()
    x1['snow_7c'] = 1.0
    design_info = getattr(final.model.data,'design_info',None)
    if design_info is None: design_info = final.model.data.model_spec
    X0 = np.asarray(build_design_matrices([design_info],x0)[0])
    X1 = np.asarray(build_design_matrices([design_info],x1)[0])
    contrast = (X1-X0).ravel()
    row = {'borough':b}
    for label,fit in [('ols',final),('panel14',robust14),('panel28',robust28)]:
        tt = fit.t_test(contrast)
        slope = float(np.asarray(tt.effect).item())
        lo,hi = np.asarray(tt.conf_int()).ravel()
        row.update({f'slope_{label}':slope,f'se_{label}':float(np.asarray(tt.sd).item()),f'p_{label}':float(np.asarray(tt.pvalue).item()),f'pct_{label}':100*np.expm1(10*slope),f'pct_low_{label}':100*np.expm1(10*lo),f'pct_high_{label}':100*np.expm1(10*hi)})
    effect_rows.append(row)
effects = pd.DataFrame(effect_rows).sort_values('pct_ols',ascending=False).reset_index(drop=True)
effects.insert(0,'rank',np.arange(1,len(effects)+1))
effects.to_csv(OUT/'tables/borough_snow_effects.csv',index=False,encoding='utf-8-sig')
display(effects[['rank','borough','pct_ols','pct_low_ols','pct_high_ols','pct_low_panel14','pct_high_panel14']].round(1))

def effect_chart(robust=False):
    label = 'panel14' if robust else 'ols'
    fig,ax = plt.subplots(figsize=(10.8,7.8))
    y = np.arange(len(effects))
    ax.barh(y,effects.pct_ols,color='#256878',height=.65)
    ax.errorbar(effects.pct_ols,y,xerr=np.vstack([effects.pct_ols-effects[f'pct_low_{label}'],effects[f'pct_high_{label}']-effects.pct_ols]),fmt='none',ecolor='#172f3e',capsize=3,linewidth=1.1)
    ax.set_yticks(y,effects.borough)
    ax.invert_yaxis()
    ax.set_xlabel('Model-implied increase in tows + 1 for 10 cm more past-week snow (%)')
    fig.suptitle('Past-week snow is associated with more towing in every borough',x=.02,ha='left',fontsize=13,y=.97)
    ax.grid(axis='x',alpha=.18)
    ax.set_axisbelow(True)
    ax.set_xlim(0,float(effects[f'pct_high_{label}'].max())*1.08)
    fig.text(.02,.018,('Whiskers: 95% panel-robust intervals, 14-day bandwidth.' if robust else 'Whiskers: conventional OLS 95% intervals. Correlated errors can make these too narrow.')+'\nAdjusted association; comparisons stay within the 30 cm cap. Other model terms held fixed.',fontsize=8.5,color='#444444')
    fig.subplots_adjust(left=.405,right=.98,top=.90,bottom=.15)
    stem = 'borough_snow_effects_panel14' if robust else 'borough_snow_effects'
    fig.savefig(OUT/'figures'/f'{stem}.png',dpi=210)
    fig.savefig(OUT/'figures'/f'{stem}.svg')
    plt.close(fig)
effect_chart()
effect_chart(robust=True)
display(Image(filename=str(OUT/'figures/borough_snow_effects.png')))

# %%
winter_test = test.loc[test.month.isin([12,1,2,3])].copy()
assert winter_test.date.nunique() == 121
baseline = winter_test.copy()
for c in weather_cols: baseline[c] = 0.0
plus10 = baseline.copy()
plus10['snow_7c'] = 10.0
eta0,eta10 = final.predict(baseline),final.predict(plus10)
factor = baseline.borough.map(smear_by_model['M3 Borough snow slopes'])
scenario = baseline[['date','borough']].copy()
scenario['pred_tows_snow0'] = np.maximum(factor*np.exp(eta0)-1,0)
scenario['pred_tows_snow10'] = np.maximum(factor*np.exp(eta10)-1,0)
scenario['extra_tows'] = scenario.pred_tows_snow10-scenario.pred_tows_snow0
scenario['uncorrected_extra_tows'] = np.maximum(np.expm1(eta10),0)-np.maximum(np.expm1(eta0),0)
cars = scenario.groupby('borough')[['pred_tows_snow0','pred_tows_snow10','extra_tows','uncorrected_extra_tows']].mean().reset_index()
cars['training_smearing_factor'] = cars.borough.map(smear_by_model['M3 Borough snow slopes'])
cars = cars.merge(effects[['borough','pct_ols']],on='borough',validate='one_to_one').sort_values('extra_tows',ascending=False)
cars.to_csv(OUT/'tables/extra_cars_scenario.csv',index=False,encoding='utf-8-sig')
display(cars.round(2))
fig,ax = plt.subplots(figsize=(10.8,7.8))
y = np.arange(len(cars))
ax.barh(y,cars.extra_tows,color='#256878',height=.65)
ax.set_yticks(y,cars.borough)
ax.invert_yaxis()
ax.set_xlabel('Additional model-estimated cars per borough-day, 0 vs 10 cm past-week snow')
fig.suptitle('Absolute towing increases depend on the borough baseline',x=.02,ha='left',fontsize=13,y=.97)
ax.grid(axis='x',alpha=.18)
ax.set_axisbelow(True)
fig.text(.02,.018,'Standardized Dec–Mar test calendar; 0 °C, other snow and rain terms = 0.\nTraining-only borough smearing adjustment. Model scenario, not an observed or causal effect.',fontsize=8.5,color='#444444')
fig.subplots_adjust(left=.405,right=.98,top=.90,bottom=.15)
fig.savefig(OUT/'figures/extra_cars_scenario.png',dpi=210)
fig.savefig(OUT/'figures/extra_cars_scenario.svg')
plt.close(fig)
display(Image(filename=str(OUT/'figures/extra_cars_scenario.png')))

# %%
original_train = data_original.loc[data_original.date <= '2025-06-30'].copy()
original_test = data_original.loc[data_original.date >= '2025-07-01'].copy()
original_train['log_tows'] = np.log1p(original_train.tows)
original_fit = smf.ols(final_formula,data=original_train).fit()
original_pred = np.maximum(np.expm1(original_fit.predict(original_test)),0)
original_mae = pd.DataFrame({'borough':original_test.borough,'abs_error':np.abs(original_test.tows-original_pred)}).groupby('borough').abs_error.mean()
original_mase = (original_mae/scales).mean()
sensitivity = pd.DataFrame([
    {'temperature':'past-only fill (main)','training_r2':final.rsquared,'test_mase':float(ladder.loc[ladder.model.eq('M3 Borough snow slopes'),'test_mase'].iloc[0])},
    {'temperature':'Part A interpolation (sensitivity)','training_r2':original_fit.rsquared,'test_mase':original_mase}
])
sensitivity.to_csv(OUT/'tables/temperature_model_sensitivity.csv',index=False)
display(sensitivity.round(5))
# Preserve Patsy formula metadata. Load pickles only from trusted sources.
for name,model in models.items():
    model.save(str(OUT/'handoff'/f'{name[:2].lower()}_model.pkl'))
manifest = {
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'source':'data/processed/tow_risk_clean.csv',
    'model_input':'part_c/handoff/model_input.csv',
    'train_start':'2021-01-01','train_end':'2025-06-30','test_start':'2025-07-01','test_end':'2026-06-30',
    'train_rows':len(train),'test_rows':len(test),'boroughs':len(boroughs),
    'formulas':formulas,'versions':versions,'temperature_dates_changed':int(changed.sum()),
    'temperature_train_dates_changed':int(changed.loc[:'2025-06-30'].sum()),
    'temperature_test_dates_changed':int(changed.loc['2025-07-01':].sum()),
    'primary_inverse':'max(expm1(predicted log(tows+1)), 0)',
    'secondary_inverse':'max(exp(predicted log(tows+1)) * borough mean(exp(training residual)) - 1, 0)',
    'mase':'Borough test MAE divided by its training mean absolute lag-7 difference; overall mean of 17 borough MASEs',
    'intervals':'Primary conventional OLS; supplementary hac-groupsum, Bartlett, 14 and 28 daily lags',
    'extra_cars_scenario':'Dec–Mar test dates; snow_7c 0 vs 10, other weather terms 0; train-only borough smearing; average cars per borough-day',
    'joint_interaction_test':joint_test
}
(OUT/'handoff/analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print('Final training R-squared:',round(final.rsquared,4))
print('Largest relative snow association:',effects.iloc[0].borough,round(effects.iloc[0].pct_ols,1),'%')
print('Smallest relative snow association:',effects.iloc[-1].borough,round(effects.iloc[-1].pct_ols,1),'%')
print('Largest extra-cars scenario:',cars.iloc[0].borough,round(cars.iloc[0].extra_tows,2),'cars per day')
print('All analysis tables, figures, model input and D handoff files saved under',OUT)
