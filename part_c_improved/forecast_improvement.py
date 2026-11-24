# %%
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['SM_FORMULA_ENGINE']='patsy'
import json, hashlib, platform, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn, scipy, statsmodels
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import joblib
from IPython.display import display, Image

roots=[Path.cwd(),*Path.cwd().parents]
ROOT=next((p for p in roots if (p/'part_c/handoff/model_input.csv').exists()),None)
if ROOT is None:
    ROOT=next((p/'Tow_Risk_Project' for p in roots if (p/'Tow_Risk_Project/part_c/handoff/model_input.csv').exists()),None)
assert ROOT is not None,'Keep this folder in the original project, alongside part_c.'
OUT=ROOT/'part_c_improved'
for folder in ['tables','figures','handoff']: (OUT/folder).mkdir(parents=True,exist_ok=True)
source=ROOT/'part_c/handoff/model_input.csv'
data=pd.read_csv(source,encoding='utf-8-sig',parse_dates=['date']).sort_values(['date','borough']).reset_index(drop=True)
assert len(data)==34119 and not data.duplicated(['date','borough']).any()
assert (data.groupby('date').size()==17).all() and not data.isna().any().any()
data['log_tows']=np.log1p(data.tows)
boroughs=sorted(data.borough.unique())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
versions={m.__name__:m.__version__ for m in [pd,np,scipy,sklearn,statsmodels,matplotlib]}
print('Package versions:',versions)
print('Input:',source)
display(data.head())

# %%
g=data.groupby('borough',sort=False)
for lag in [1,2,3,7,14]:
    data[f'tow_lag{lag}']=g.tows.shift(lag)
for window in [3,7,14]:
    data[f'tow_mean{window}']=g.tows.transform(lambda s:s.shift(1).rolling(window).mean())
data['tow_max7']=g.tows.transform(lambda s:s.shift(1).rolling(7).max())
data['tow_change1']=data.tow_lag1-data.tow_lag2
data['log_lag1']=np.log1p(data.tow_lag1)
data['log_lag7']=np.log1p(data.tow_lag7)
daily=data.groupby('date').agg(city_tows=('tows','sum'),snow_1=('snow_1','first')).sort_index()
for lag in [1,2,7]: daily[f'city_lag{lag}']=daily.city_tows.shift(lag)
for window in [3,7]: daily[f'city_mean{window}']=daily.city_tows.shift(1).rolling(window).mean()
daily['snow_2_3']=daily.snow_1.shift(1).rolling(2).sum()
daily['snow_4_7']=daily.snow_1.shift(3).rolling(4).sum()
daily['snow_7_uncapped']=daily.snow_1.rolling(7).sum()
age=[]
last=None
for date,snow in daily.snow_1.items():
    if snow>=5: last=date
    age.append(30 if last is None else min((date-last).days,30))
daily['days_since_snow5']=age
new_daily=[c for c in daily if c not in ['city_tows','snow_1']]
data=data.merge(daily[new_daily],left_on='date',right_index=True,how='left',validate='many_to_one')
data['other_boroughs_lag1']=data.city_lag1-data.tow_lag1
data['log_city_lag1']=np.log1p(data.city_lag1)
data['borough_id']=data.borough.map({b:i for i,b in enumerate(boroughs)}).astype(int)
data['day_of_year']=data.date.dt.dayofyear
calendar_features=['borough_id','dow','month','trend','moving_day','day_of_year']
weather_features=['snow_7c','snow_8_14','snow_1','rain_1','temp_7','snow_2_3','snow_4_7','snow_7_uncapped','days_since_snow5']
own_features=[f'tow_lag{k}' for k in [1,2,3,7,14]]+[f'tow_mean{k}' for k in [3,7,14]]+['tow_max7','tow_change1']
city_features=['city_lag1','city_lag2','city_lag7','city_mean3','city_mean7','other_boroughs_lag1']
feature_sets={
    'weather':calendar_features+weather_features,
    'own_lags':calendar_features+weather_features+own_features,
    'city_lags':calendar_features+weather_features+own_features+city_features,
}
all_features=feature_sets['city_lags']
panel=data.dropna(subset=all_features).copy()
assert panel.date.min()==pd.Timestamp('2021-01-15')
assert np.allclose(panel.tow_lag1, data.groupby('borough').tows.shift(1).loc[panel.index])
feature_notes={
 'calendar':calendar_features,'weather':weather_features,'own_towing_history':own_features,'city_towing_history':city_features,
 'horizon':'One-step ahead, updated daily using preceding observed outcomes, with fitted parameters fixed within each evaluation year.',
 'availability':'Assumes complete borough and city tow counts through yesterday are available before the forecast.'
}
(OUT/'handoff/feature_definitions.json').write_text(json.dumps(feature_notes,indent=2),encoding='utf-8')
print('Advanced-model panel starts:',panel.date.min().date(),'| rows:',len(panel))
print('Maximum feature history: 14 preceding days.')

# %%
formula='log_tows ~ C(borough) + C(dow) + C(month) + trend + moving_day + snow_7c + snow_8_14 + snow_1 + rain_1 + temp_7 + C(borough):snow_7c'
formula_ar=formula+' + log_lag1 + log_lag7 + log_city_lag1'
configs=[
 {'name':'Naive yesterday','kind':'naive'},
 {'name':'Seasonal naive 7 days','kind':'seasonal'},
 {'name':'Past 7 days mean','kind':'mean7'},
 {'name':'Required weather OLS','kind':'ols'},
 {'name':'OLS plus towing lags','kind':'ar_ols'},
]
for fs in ['weather','own_lags','city_lags']:
    for size,leaves,iters,minleaf in [('small',15,180,70),('medium',31,260,45)]:
        configs.append({'name':f'Boost {fs} {size}','kind':'boost','features':fs,'leaves':leaves,'iterations':iters,'min_leaf':minleaf,'target':'count'})
for size,leaves,iters,minleaf in [('small',15,180,70),('medium',31,260,45)]:
    configs.append({'name':f'Boost naive correction {size}','kind':'boost','features':'city_lags','leaves':leaves,'iterations':iters,'min_leaf':minleaf,'target':'delta'})
folds=[('2023-24','2023-07-01','2024-06-30'),('2024-25','2024-07-01','2025-06-30')]
protocol={'selection':'Lowest mean borough-MASE across the two earlier full-year folds; no final-year tuning.','folds':folds,'configurations':configs,'final_start':'2025-07-01','final_end':'2026-06-30','metric':'Mean of 17 borough MASEs, training seasonal lag-7 scale','random_seed':20261123,'test_status':'Previously inspected for original OLS, so the improved analysis is exploratory.'}
(OUT/'handoff/protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

def mase_scales(training):
    tr=training.sort_values(['date','borough'])
    return tr.assign(e=(tr.tows-tr.groupby('borough').tows.shift(7)).abs()).groupby('borough').e.mean()

def score(evaluation,pred,scales):
    e=evaluation[['date','borough','tows']].copy()
    e['pred']=np.maximum(np.asarray(pred),0)
    e['absolute_error']=(e.tows-e.pred).abs()
    e['scaled_error']=e.absolute_error/e.borough.map(scales)
    winter=e.date.dt.month.isin([12,1,2,3])
    city_observed=e.groupby('date').tows.sum()
    peak=e.date.map(city_observed).ge(150)
    return {'mase':e.scaled_error.mean(),'mae':e.absolute_error.mean(),'winter_mase':e.loc[winter,'scaled_error'].mean(),'other_mase':e.loc[~winter,'scaled_error'].mean(),'high_activity_mase':e.loc[peak,'scaled_error'].mean(),'r2_count':r2_score(e.tows,e.pred)}

def fit_predict(config,training,evaluation,scales,feature_override=None):
    k=config['kind']
    if k=='naive': return evaluation.tow_lag1.to_numpy(),None
    if k=='seasonal': return evaluation.tow_lag7.to_numpy(),None
    if k=='mean7': return evaluation.tow_mean7.to_numpy(),None
    if k in ['ols','ar_ols']:
        use=training if k=='ols' else training.dropna(subset=['log_lag1','log_lag7','log_city_lag1'])
        fit=smf.ols(formula if k=='ols' else formula_ar,data=use).fit()
        pred=np.maximum(np.expm1(fit.predict(evaluation)),0)
        return pred,fit
    cols=feature_override if feature_override is not None else feature_sets[config['features']]
    tr=training.dropna(subset=all_features)
    categorical=[c for c in ['borough_id','dow','month'] if c in cols]
    fit=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.07,max_iter=config['iterations'],max_leaf_nodes=config['leaves'],min_samples_leaf=config['min_leaf'],l2_regularization=1.,categorical_features=categorical,early_stopping=False,random_state=20261123)
    target=tr.tows-tr.tow_lag1 if config['target']=='delta' else tr.tows
    weights=1./tr.borough.map(scales)
    fit.fit(tr[cols],target,sample_weight=weights)
    pred=fit.predict(evaluation[cols])
    if config['target']=='delta': pred=pred+evaluation.tow_lag1.to_numpy()
    return np.maximum(pred,0),fit

validation=[]
for fold,start,end in folds:
    tr=data.loc[data.date<pd.Timestamp(start)].copy()
    va=panel.loc[panel.date.between(start,end)].copy()
    scales=mase_scales(tr)
    for cfg in configs:
        begin=time.monotonic()
        pred,fit=fit_predict(cfg,tr,va,scales)
        result={'fold':fold,'model':cfg['name'],**score(va,pred,scales),'fit_seconds':time.monotonic()-begin}
        validation.append(result)
        print(f"{fold} | {cfg['name']:<33} | MASE {result['mase']:.4f}",flush=True)
validation=pd.DataFrame(validation)
validation.to_csv(OUT/'tables/validation_folds.csv',index=False)
ranking=validation.groupby('model')[['mase','mae','winter_mase','other_mase','high_activity_mase']].mean().sort_values('mase')
ranking.to_csv(OUT/'tables/validation_ranking.csv')
# The benchmark can win. Do not force a complex model into first place.
winner_name=ranking.index[0]
winner_config=next(c for c in configs if c['name']==winner_name)
learned_ranking=ranking.loc[[c['name'] for c in configs if c['kind'] in ['ols','ar_ols','boost']]].sort_values('mase')
best_learned_name=learned_ranking.index[0]
best_learned_config=next(c for c in configs if c['name']==best_learned_name)
selection={'winner':winner_name,'winner_config':winner_config,'best_learned':best_learned_name,'best_learned_config':best_learned_config,'selected_before_test':True}
(OUT/'handoff/selected_model.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
print('VALIDATION WINNER:',winner_name)
display(ranking.round(4))

# %%
train=data.loc[data.date<='2025-06-30'].copy()
test=panel.loc[panel.date.between('2025-07-01','2026-06-30')].copy()
assert len(test)==6205
scales=mase_scales(train)
scales.rename('training_lag7_mae').to_csv(OUT/'tables/mase_scales.csv')
weather_names=[c['name'] for c in configs if c['kind']=='boost' and c['features']=='weather']
weather_winner=ranking.loc[weather_names].sort_values('mase').index[0]
eval_names=list(dict.fromkeys(['Naive yesterday','Seasonal naive 7 days','Past 7 days mean','Required weather OLS','OLS plus towing lags',weather_winner,winner_name,best_learned_name]))
predictions=test[['date','borough','tows']].copy()
test_scores=[]
fitted={}
for name in eval_names:
    cfg=next(c for c in configs if c['name']==name)
    pred,fit=fit_predict(cfg,train,test,scales)
    predictions[name]=pred
    fitted[name]=fit
    test_scores.append({'model':name,'role':('validation winner' if name==winner_name else 'comparator'),**score(test,pred,scales)})
if best_learned_config['kind']=='boost':
    no_weather=[c for c in feature_sets[best_learned_config['features']] if c not in weather_features]
    pred,fit=fit_predict(best_learned_config,train,test,scales,feature_override=no_weather)
    predictions['Selected learner without weather']=pred
    test_scores.append({'model':'Selected learner without weather','role':'ablation',**score(test,pred,scales)})
test_scores=pd.DataFrame(test_scores)
test_scores.to_csv(OUT/'tables/final_year_scores.csv',index=False)
predictions.to_csv(OUT/'handoff/final_year_predictions.csv',index=False,encoding='utf-8-sig')
by_borough=[]
for name in predictions.columns[3:]:
    e=test[['borough','tows']].copy()
    e['absolute_error']=np.abs(e.tows-predictions[name])
    per=e.groupby('borough').absolute_error.mean()
    for b in boroughs: by_borough.append({'model':name,'borough':b,'mae':per[b],'mase':per[b]/scales[b]})
by_borough=pd.DataFrame(by_borough)
by_borough.to_csv(OUT/'tables/final_year_borough_scores.csv',index=False,encoding='utf-8-sig')
print('Frozen validation winner:',winner_name)
print('Best learned model selected on validation:',best_learned_name)
display(test_scores.round(4))
if fitted[best_learned_name] is not None:
    joblib.dump({'model':fitted[best_learned_name],'config':best_learned_config,'feature_names':(feature_sets[best_learned_config['features']] if best_learned_config['kind']=='boost' else None),'borough_mapping':{b:i for i,b in enumerate(boroughs)},'versions':versions},OUT/'handoff/best_learned_model.joblib',compress=3)

# %%
def daily_losses(name):
    e=test[['date','borough','tows']].copy()
    e['loss']=np.abs(e.tows-predictions[name])/e.borough.map(scales)
    return e.groupby('date').loss.mean()

selected_loss=daily_losses(winner_name)
rng=np.random.default_rng(20261123)
N=len(selected_loss)
B=1200
block=14
starts=rng.integers(0,N,size=(B,int(np.ceil(N/block))))
ix=(starts[:,:,None]+np.arange(block)[None,None,:])%N
ix=ix.reshape(B,-1)[:,:N]
comparisons=[]
for baseline in ['Naive yesterday','Seasonal naive 7 days','Required weather OLS']:
    bl=daily_losses(baseline).to_numpy()
    sl=selected_loss.to_numpy()
    gain=100*(1-sl.mean()/bl.mean())
    draws=100*(1-sl[ix].mean(axis=1)/bl[ix].mean(axis=1))
    low,high=np.quantile(draws,[.025,.975])
    comparisons.append({'selected_model':winner_name,'baseline':baseline,'mase_reduction_pct':gain,'block14_low_pct':low,'block14_high_pct':high})
comparisons=pd.DataFrame(comparisons)
comparisons.to_csv(OUT/'tables/improvement_intervals.csv',index=False)
display(comparisons.round(2))

actual_city=data.groupby('date').tows.sum()
prior_quiet=pd.concat([actual_city.shift(k).lt(150) for k in [1,2,3]],axis=1).all(axis=1)
onsets=(actual_city.ge(150)&prior_quiet).reindex(pd.DatetimeIndex(test.date.unique()))
activity=[]
city_predictions=predictions.groupby('date')[list(predictions.columns[3:])].sum()
for name in city_predictions:
    warning=city_predictions[name].ge(150)
    truth=actual_city.reindex(warning.index).ge(150)
    activity.append({'model':name,'activity_starts':int(onsets.sum()),'starts_warned':int((warning&onsets).sum()),'high_activity_days':int(truth.sum()),'high_activity_days_warned':int((warning&truth).sum()),'false_warning_days':int((warning&~truth).sum()),'total_warning_days':int(warning.sum())})
activity=pd.DataFrame(activity)
activity.to_csv(OUT/'tables/activity_warning_check.csv',index=False)
display(activity)
city_predictions.insert(0,'actual_tows',actual_city.reindex(city_predictions.index))
city_predictions.to_csv(OUT/'tables/citywide_final_year_predictions.csv')

# %%
names=list(dict.fromkeys(['Required weather OLS','Naive yesterday','Seasonal naive 7 days','OLS plus towing lags',weather_winner,winner_name,best_learned_name]))
chart=test_scores.set_index('model').loc[names].sort_values('mase')
fig,ax=plt.subplots(figsize=(11,5.8))
colors=['#1e7568' if name==winner_name else '#9baeb7' for name in chart.index]
y=np.arange(len(chart))
ax.barh(y,chart.mase,color=colors,height=.65)
ax.set_yticks(y,chart.index)
ax.invert_yaxis()
for i,v in enumerate(chart.mase): ax.text(v+.015,i,f'{v:.3f}',va='center')
ax.set_xlabel('Test MASE, average over 17 boroughs (lower is better)')
ax.set_xlim(0,float(chart.mase.max())*1.17)
ax.grid(axis='x',alpha=.17)
ax.set_axisbelow(True)
fig.suptitle('Forecasts must beat yesterday’s towing count',x=.025,ha='left',fontsize=14,y=.97)
fig.text(.025,.025,'Model choice uses 2023–25 validation years. Evaluation: Jul 2025–Jun 2026, daily one-step forecasts.\nRecent towing counts must be available through the preceding day.',fontsize=9,color='#444444')
fig.subplots_adjust(left=.34,right=.95,bottom=.18,top=.86)
fig.savefig(OUT/'figures/forecast_comparison.png',dpi=200)
fig.savefig(OUT/'figures/forecast_comparison.svg')
plt.close(fig)
display(Image(filename=str(OUT/'figures/forecast_comparison.png')))

city=city_predictions.loc['2025-12-01':'2026-03-31']
fig,ax=plt.subplots(figsize=(12,5.4))
ax.plot(city.index,city.actual_tows,color='#172f3e',lw=1.4,label='Recorded tows')
ax.plot(city.index,city['Naive yesterday'],color='#b47e46',lw=1,alpha=.75,label='Yesterday baseline')
if winner_name!='Naive yesterday': ax.plot(city.index,city[winner_name],color='#178772',lw=1.4,label='Validation-selected forecast')
ax.set_ylabel('Citywide cars towed per day')
ax.set_xlabel('Forecast day')
ax.grid(alpha=.15)
ax.legend(frameon=False)
fig.suptitle('Winter spikes remain the difficult forecasting problem',x=.025,ha='left',fontsize=14,y=.97)
fig.text(.025,.02,'Daily one-step predictions using only prior-day information. Fitted parameters fixed through the evaluation year.',fontsize=9,color='#444444')
fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.87)
fig.savefig(OUT/'figures/winter_predictions.png',dpi=200)
fig.savefig(OUT/'figures/winter_predictions.svg')
plt.close(fig)

wide=by_borough.pivot(index='borough',columns='model',values='mase')
gain=(100*(1-wide[winner_name]/wide['Naive yesterday'])).sort_values(ascending=False)
fig,ax=plt.subplots(figsize=(11,7.5))
ax.barh(np.arange(len(gain)),gain.values,color=['#1e7568' if x>=0 else '#b76b58' for x in gain.values])
ax.set_yticks(np.arange(len(gain)),gain.index)
ax.invert_yaxis()
ax.axvline(0,color='#333333',lw=.8)
ax.set_xlabel('MASE reduction relative to yesterday baseline (%)')
ax.grid(axis='x',alpha=.15)
ax.set_axisbelow(True)
fig.suptitle('Forecast improvement varies across boroughs',x=.025,ha='left',fontsize=14,y=.97)
fig.text(.025,.02,'Positive values favour the validation-selected model. Evaluation: July 2025–June 2026.',fontsize=9,color='#444444')
fig.subplots_adjust(left=.41,right=.97,bottom=.13,top=.88)
fig.savefig(OUT/'figures/borough_forecast_improvement.png',dpi=200)
fig.savefig(OUT/'figures/borough_forecast_improvement.svg')
plt.close(fig)

selected_row=test_scores.loc[test_scores.model.eq(winner_name)].iloc[0]
naive_row=test_scores.loc[test_scores.model.eq('Naive yesterday')].iloc[0]
ols_row=test_scores.loc[test_scores.model.eq('Required weather OLS')].iloc[0]
onset_row=activity.loc[activity.model.eq(winner_name)].iloc[0]
summary={
 'selection':selection,'versions':versions,
 'selected_test_mase':float(selected_row.mase),'selected_test_mae':float(selected_row.mae),'selected_test_r2_count':float(selected_row.r2_count),
 'naive_test_mase':float(naive_row.mase),'required_ols_test_mase':float(ols_row.mase),
 'gain_vs_naive_pct':float(100*(1-selected_row.mase/naive_row.mase)),
 'gain_vs_ols_pct':float(100*(1-selected_row.mase/ols_row.mase)),
 'boroughs_beating_naive':int(gain.gt(0).sum()),'boroughs':17,
 'activity_starts':int(onset_row.activity_starts),'starts_warned':int(onset_row.starts_warned),
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'test_year_status':'Previously inspected for original OLS. Candidate selection restricted to 2023–25 validation. Exploratory improvement; needs a future winter for confirmation.'
}
(OUT/'handoff/results_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
