# Blood Glucose Time-Series Prediction

OhioT1DM multi-step blood glucose forecasting with classical machine learning,
statistical time-series models, and recurrent neural networks.

## Overview

This project builds a complete modeling pipeline for the OhioT1DM dataset. It
parses patient XML files, aligns CGM and event/sensor signals onto a 5-minute
grid, engineers model-ready features, creates sliding windows, trains multiple
forecasting models, and exports evaluation results and plots.

The default task is:

- Input: the previous 30 minutes of data, or 6 samples at 5-minute frequency
- Output: the next 30 minutes of glucose, or 6 forecast steps
- Target: raw CGM glucose for evaluation
- Input variants: smoothed glucose and raw glucose
- Feature variants: engineered feature set and glucose-only baseline

## Models

The main pipeline can run:

| Family | Models |
|---|---|
| Classical ML | Linear Regression, SVR, XGBoost, Decision Tree, Random Forest, KNN |
| Statistical | ARIMA, SARIMAX |
| Deep learning | LSTM, GRU |

Classical and deep models run across four dataset variants:

- `with features + Savitzky-Golay`
- `baseline + Savitzky-Golay`
- `with features + no Savitzky-Golay`
- `baseline + no Savitzky-Golay`

ARIMA runs as a glucose-only baseline. SARIMAX uses the aligned non-glucose
features as exogenous variables.

## Project Structure

```text
blood-glucose-prediction/
|-- configs/
|   `-- config.yaml                  # Paths, dataset settings, windows, models
|-- src/
|   |-- data/
|   |   |-- parser.py                # OhioT1DM XML parser
|   |   `-- loader.py                # XML discovery, alignment, master DataFrames
|   |-- features/
|   |   `-- windowing.py             # Sliding windows and scaling
|   |-- models/
|   |   |-- classical.py             # LR, SVR, XGBoost, DT, RF, KNN
|   |   |-- arima_model.py           # Patient-wise ARIMA
|   |   |-- sarimax_model.py         # Patient-wise SARIMAX with exogenous inputs
|   |   `-- deep_learning.py         # LSTM and GRU
|   |-- evaluation/
|   |   |-- diagnostics.py           # Missingness, ADF, aligned correlations
|   |   |-- metrics.py               # Metrics, CV, ResultStore
|   |   `-- visualizer.py            # Result plots
|   `-- utils/
|       |-- config_loader.py
|       |-- logger.py
|       `-- seed.py
|-- scripts/
|   |-- export_master_dataframes.py  # Build cached train/test master CSVs
|   |-- run_classical.py
|   |-- run_deep.py
|   |-- run_arima.py
|   `-- run_single_model.py
|-- notebooks/
|   |-- EDA.ipynb
|   `-- model_results_comparison.ipynb
|-- output/
|   |-- README.md
|   |-- processed/                   # Cached master CSVs
|   |-- logs/
|   |-- results/
|   |-- plots/
|   `-- checkpoints/
|-- main.py
|-- requirements.txt
|-- pytest.ini
`-- README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Request access to the OhioT1DM dataset and extract it so it matches the default
config path:

```text
src/data/OhioT1DM/
|-- 2018/
|   |-- train/
|   `-- test/
`-- 2020/
    |-- train/
    `-- test/
```

If your dataset lives elsewhere, update `paths.data_dir` in
`configs/config.yaml`.

## Running The Pipeline

Build the cached per-patient master CSVs:

```bash
python scripts/export_master_dataframes.py
```

By default this writes:

```text
output/processed/train/*_train_master.csv
output/processed/test/*_test_master.csv
```

Then run the full pipeline:

```bash
python main.py
```

`main.py` first tries to load cached masters from `output/processed`. If the
cache is incomplete, it rebuilds the master DataFrames from XML.

To force a rebuild from XML:

```bash
python main.py --rebuild-masters
```

To use another cache directory:

```bash
python main.py --masters-dir path/to/processed
```

## Common Commands

Run only selected models:

```bash
python main.py --models lr rf xgb
python main.py --models arima sarimax
python main.py --models lstm gru
```

Skip deep learning during faster experiments:

```bash
python main.py --skip-deep
```

Skip plot generation:

```bash
python main.py --no-plots
```

Run grouped helper scripts:

```bash
python scripts/run_classical.py
python scripts/run_deep.py
python scripts/run_arima.py
```

Run one model through the helper script:

```bash
python scripts/run_single_model.py --model rf
python scripts/run_single_model.py --model xgb --no-plots
```

Note: `main.py` supports `sarimax` directly with `--models sarimax`. The
`run_single_model.py` helper currently supports `lr`, `svr`, `xgb`, `dt`, `rf`,
`knn`, `arima`, `lstm`, and `gru`.

## Configuration

All primary settings are in `configs/config.yaml`.

| Section | Purpose |
|---|---|
| `paths` | Dataset, output, logs, plots, results, checkpoint paths |
| `dataset` | Cohorts, XML suffixes, parsed features, CGM frequency |
| `window` | Lookback and forecast horizon |
| `feature_engineering` | Savitzky-Golay smoothing, rolling insulin features, excluded features |
| `evaluation` | CV folds, shuffle flag, SVR subsampling, metrics |
| `models` | Classical and statistical model hyperparameters |
| `deep_learning` | LSTM/GRU architecture, epochs, batch size, callbacks |
| `seed` | Reproducibility seed |
| `logging` | Console/file logging options |

The default window settings are:

```yaml
window:
  lookback_minutes: 30
  forecast_minutes: 30
  freq_minutes: 5
```

This becomes 6 lookback steps and 6 forecast steps.

## Data Processing

The data loader builds one master DataFrame per patient and split. The important
steps are:

1. Discover train/test XML files from the configured cohorts.
2. Parse CGM glucose and align it to a dense 5-minute grid.
3. Keep `raw_glucose_level` as the evaluation target.
4. Build `glucose_level` as a causal Savitzky-Golay smoothed input.
5. Snap point events such as bolus, meal, basal, GSR, finger-stick, and skin
   temperature to the nearest glucose timestamp.
6. Expand interval events such as exercise, sleep, and temporary basal onto the
   glucose grid.
7. Impute selected event/sensor features and engineer insulin/exercise features.
8. Export optional missingness reports before imputation.

The cached files are written by `save_master_dataframes()` in
`src/data/loader.py` and loaded by `load_master_dataframes()`.

## Feature Engineering

| Feature | Description |
|---|---|
| `glucose_level` | Causal Savitzky-Golay smoothed CGM input |
| `raw_glucose_level` | Unsmoothed CGM target used for evaluation |
| `bolus` | Bolus insulin aggregated per 5-minute glucose timestamp |
| `basal` | Basal rate aligned to the glucose grid |
| `meal` | Meal carbohydrates aggregated per timestamp |
| `exercise` | Interval exercise signal aligned to the grid |
| `basis_gsr` | Wearable GSR signal snapped to nearest glucose timestamp |
| `basis_sleep` | Sleep interval mask |
| `finger_stick` | Finger-stick glucose readings aligned to CGM timestamps |
| `basis_skin_temperature` | Skin temperature aligned to the grid |
| `total_insulin` | Bolus plus basal contribution for each 5-minute step |
| `insulin_count` | Number of bolus events in each step |
| `insulin_3h_std` | Rolling 3-hour standard deviation of total insulin |
| `daily_exercise` | Day-level exercise flag |

High-missing features configured under `feature_engineering.exclude_features`
are excluded from model inputs. The current defaults are:

```text
acceleration
basis_steps
basis_air_temperature
basis_heart_rate
```

## Savitzky-Golay Smoothing

The default smoothing config is:

```yaml
feature_engineering:
  savgol_window: 11
  savgol_polyorder: 2
```

Because the CGM grid is 5 minutes, a window of 11 samples covers 55 minutes.
The implementation is causal: each smoothed value uses only the current reading
and prior readings. This reduces sensor noise without leaking future glucose
values into the lookback window.

The pipeline also builds no-Savitzky-Golay variants by replacing
`glucose_level` with `raw_glucose_level` before windowing.

## Windowing And Scaling

Window creation happens in `src/features/windowing.py`.

`create_multistep_dataset()` creates the per-patient sliding windows:

```text
X[i] = feature rows i through i + lookback_steps - 1
y[i] = target rows i + lookback_steps through i + lookback_steps + forecast_steps - 1
```

`build_global_dataset()` applies that per patient and concatenates windows, so
windows never cross patient boundaries.

`prepare_datasets()` builds full-feature and baseline variants, then scales:

- 2-D flattened arrays for classical ML
- 3-D sequence arrays for LSTM/GRU
- target arrays for deep learning inverse transforms

Windows with timestamp gaps or missing values in inputs/targets are skipped.

## Evaluation

The project reports:

- RMSE
- MAE
- R2
- MARD

Classical models use KFold cross-validation with `shuffle: false` by default.
Deep learning models report a validation-loss proxy for CV metrics and evaluate
on the hold-out test split. ARIMA and SARIMAX run patient-wise multi-step
forecasts and report averaged patient metrics.

Results are accumulated in `ResultStore` and saved to:

```text
output/results/model_results.csv
```

## Outputs

After a successful run, expected generated outputs include:

| Path | Contents |
|---|---|
| `output/processed/` | Cached train/test master CSVs and missingness reports |
| `output/results/model_results.csv` | Sorted model evaluation table |
| `output/plots/rmse_r2_bars.png` | RMSE and R2 comparison plot |
| `output/plots/feature_impact.png` | Feature-vs-baseline comparison |
| `output/plots/best_model_predictions.png` | Prediction diagnostics for the best model |
| `output/plots/horizon_rmse.png` | Per-step horizon RMSE for LSTM/GRU |
| `output/plots/summary_heatmap.png` | Summary metric heatmap |
| `output/logs/bg_prediction_*.log` | Timestamped run logs |
| `output/checkpoints/` | Optional Keras checkpoints |

Generated outputs are intentionally ignored by git.

## Diagnostics And Notebooks

`src/evaluation/diagnostics.py` contains reusable EDA helpers for:

- missingness summaries across configured regular signals
- Augmented Dickey-Fuller stationarity checks
- feature correlations after timestamp alignment to glucose

The notebooks in `notebooks/` are for exploratory analysis and result
comparison.

## Testing

Run tests with:

```bash
pytest
```

The local `pytest.ini` sets the project root on `PYTHONPATH`.

## Dataset Citation

Marling, C. and Bunescu, R. (2020). The OhioT1DM Dataset for Blood Glucose
Level Prediction: Update 2020. KHD@IJCAI 2020.

Marling, C. and Bunescu, R. (2018). The OhioT1DM Dataset for Blood Glucose
Level Prediction. KHD@IJCAI 2018.

## Notes

This repository does not include patient data. The OhioT1DM dataset is subject
to its own data-use agreement.
