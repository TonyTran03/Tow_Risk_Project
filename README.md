# Montréal Tow Risk Project

This repository brings together the supplied data preparation notebook and the
Part C modelling analyses. It follows the A–E team responsibilities in the
final analysis plan. Team letters refer to responsibilities, not model versions.

## Team responsibilities and files

| Role | Responsibility | Available work |
| --- | --- | --- |
| A | Prepare towing and weather data; build the complete borough/day panel and past-weather features | `Part_A_Tow_Risk.ipynb` and `data/processed/tow_risk_clean.csv` were supplied |
| B | Explore snowfall timing, correlations, monthly/weekday patterns; merge and edit the report | No Person B notebook or report was supplied; see `B_exploration/README.md` |
| C | Fit the OLS ladder, estimate borough snow associations, and explain model results | `part_c/Part_C_Tow_Risk.ipynb` and `part_c/part_c_analysis.py` |
| C extension | Compare stronger next-day forecasts with simple baselines using earlier validation years | `part_c_improved/Part_C_Forecast_Improvement.ipynb` and `part_c_improved/forecast_improvement.py` |
| D | Final benchmark validation, rolling checks, residual diagnostics and limitations | No standalone Person D submission was supplied; some forecast comparisons are included in the C extension |
| E | Framing, recommendations, report conclusion and presentation coordination | No Person E submission is included |

The supplied Part A notebook is preserved as received. Part C code and its
forecasting extension were developed with Codex assistance. These files do not
establish who completed the team's other sections. Add teammates' actual
submissions before assigning them credit or marking their sections complete.

## Repository contents

- `Part_A_Tow_Risk.ipynb`: supplied data preparation notebook.
- `data/processed/tow_risk_clean.csv`: supplied 34,119-row borough/day panel.
- `montreal_weather_2020.csv` through `montreal_weather_2026.csv`: supplied weather.
- `B_exploration/`: location reserved for B's actual exploration submission.
- `part_c/`: original OLS analysis, coefficients and borough-effect calculations.
- `part_c/handoff/model_input.csv`: model input after the past-only temperature repair.
- `part_c_improved/`: next-day forecasting extension and reproducible validation.
- `docs/analysis_plan.docx`: supplied team scope and ownership reference.

Generated model binaries, reports, charts, slide files, ZIPs and prediction
exports are excluded from Git. They can be regenerated from the notebooks.
The original source files remain available locally.

## Run the project

Use Python 3.12. From this repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open the repository folder in VS Code or start Jupyter from this root. Run the
notebook cells in order, using that Python environment.

1. **A data preparation:** open `Part_A_Tow_Risk.ipynb`. It requires the seven
   weather files and `remorquages.csv` in this root. The raw towing file is
   available locally but excluded from Git because of its size. Obtain it from
   the team's shared data source for a fresh checkout. Its filename, original
   date and borough fields are checked by the notebook.
2. **C original models:** run `python part_c/part_c_analysis.py`, or open
   `part_c/Part_C_Tow_Risk.ipynb`. These use the included cleaned panel and
   weather files. The optional raw-count check runs if `remorquages.csv` is present.
3. **C forecasting extension:** run `python part_c_improved/forecast_improvement.py`,
   or open its notebook. It uses the included repaired model input and can run
   without rerunning A or C first.
4. **B exploration:** add B's actual notebook when available; it should read the
   shared cleaned panel. Its required analyses are listed in `B_exploration/README.md`.

The scripts write generated results to their respective analysis folders.

## Model and data interpretation

The original required models use OLS on `log(tows + 1)` and train through June
2025, with July 2025–June 2026 as the test year. Their snow slopes describe
adjusted associations. A temperature interpolation look-ahead issue in the
supplied preparation was repaired for modelling, without changing A's notebook
or original cleaned CSV.

The forecasting extension uses recent observed towing counts and learns a
correction to yesterday's count. Model selection uses two earlier validation
years. It is a daily one-step forecast and assumes yesterday's counts are
available in time. It is not a street-level towing probability or a reliable
warning before every new operation. The final-year original OLS score had
already been inspected, so confirmation should use a future winter.

## Contribution tracking

Each teammate should add their own notebook and a short description of what
they contributed. The team plan defines A–E roles but does not assign verified
individual names in this repository. Person B's submission is currently missing.
