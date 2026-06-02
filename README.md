# Blood Glucose Time-Series Prediction
### OhioT1DM Dataset · Multi-Step Forecasting · Classical ML + Deep Learning

---

## Overview

This project implements a full machine-learning pipeline for **blood glucose level (BGL)
prediction** using the [OhioT1DM dataset](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html)
— a clinical dataset of 12 Type-1 diabetes patients (2018 + 2020 cohorts) with continuous
glucose monitoring (CGM), insulin, and wearable sensor data sampled at 5-minute intervals.

**Prediction task:** Given the last 30 minutes of multi-modal data (6 CGM readings +
engineered features), forecast the next 30 minutes of glucose values (6 steps ahead, i.e.
+5 min, +10 min, …, +30 min).

**Models evaluated:**
- Linear Regression, SVR, XGBoost, Decision Tree, Random Forest, KNN *(classical ML)*
- ARIMA *(statistical time-series)*
- LSTM, GRU *(deep learning)*

Each model is evaluated in two configurations:
- **with features** – full feature set (glucose + insulin + exercise signals)
- **baseline** – glucose-only

---

## Project Structure

```
bg_prediction/
│
├── configs/
│   └── config.yaml              # All hyperparameters, paths, model settings
│
├── src/
│   ├── data/
│   │   ├── parser.py            # OhioT1DM XML parser (all features, both cohorts)
│   │   └── loader.py            # File discovery + master DataFrame builder
│   │
│   ├── features/
│   │   └── windowing.py         # Sliding-window datasets, MinMax scaling
│   │
│   ├── models/
│   │   ├── classical.py         # LR, SVR, XGBoost, DT, RF, KNN
│   │   ├── arima_model.py       # Patient-wise ARIMA forecasting
│   │   ├── sarimax_model.py     # Patient-wise SARIMAX/ARIMAX with exogenous features
│   │   └── deep_learning.py     # LSTM & GRU (Keras)
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # RMSE, MAE, R², MARD; KFold CV; ResultStore
│   │   └── visualizer.py        # All comparison & diagnostic plots
│   │
│   └── utils/
│       ├── config_loader.py     # YAML config loader
│       ├── logger.py            # Timestamped logging to file + console
│       └── seed.py              # Global reproducibility seed
│
├── scripts/
│   ├── run_classical.py         # Run all classical ML models
│   ├── run_deep.py              # Run LSTM + GRU only
│   ├── run_arima.py             # Run ARIMA + SARIMAX only
│   └── run_single_model.py      # Run any one model by short name
│
├── notebooks/
│   └── EDA.py                   # EDA-only notebook (convert to .ipynb with jupytext)
│
├── tests/
│   ├── test_parser.py
│   ├── test_metrics.py
│   └── test_windowing.py
│
├── output/
│   ├── logs/                    # Timestamped pipeline logs
│   ├── results/                 # model_results.csv, eda_summary_stats.csv
│   ├── plots/                   # PNG figures
│   └── checkpoints/             # Keras model weights (optional)
│
├── main.py                      # Pipeline orchestration entry point
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd bg_prediction
pip install -r requirements.txt
```

### 2. Obtain the dataset

Request access to OhioT1DM at http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html.
Extract the archive so the structure matches:

```
data/OhioT1DM/
├── 2018/
│   ├── train/  (e.g. 559-ws-training.xml …)
│   └── test/   (e.g. 559-ws-testing.xml …)
└── 2020/
    ├── train/
    └── test/
```

Update `configs/config.yaml` → `paths.data_dir` if you use a different location.

### 3. Run the full pipeline

Build the model-ready per-patient master CSVs once:

```bash
python scripts/export_master_dataframes.py
```

Then run the model pipeline. `main.py` will load the cached files from
`output/processed` instead of rebuilding them from XML:

```bash
python main.py
```

### 4. Run individual model groups

```bash
# Classical ML only (fast)
python scripts/run_classical.py

# Deep learning only
python scripts/run_deep.py

# ARIMA baseline + SARIMAX with exogenous features
python scripts/run_arima.py

# A specific model
python scripts/run_single_model.py --model xgb
```

### 5. Command-line options

| Flag | Description |
|---|---|
| `--config PATH` | Path to a custom YAML config (default: `configs/config.yaml`) |
| `--models lr svr …` | Run only the listed models |
| `--skip-deep` | Skip LSTM / GRU (faster for prototyping) |
| `--no-plots` | Disable figure generation |
| `--masters-dir PATH` | Load exported master CSVs from this directory (default: `output/processed`) |
| `--rebuild-masters` | Force rebuilding master DataFrames from XML |

### 6. EDA notebook

```bash
# Convert the EDA script to a Jupyter notebook
pip install jupytext
jupytext --to notebook notebooks/EDA.py
jupyter notebook notebooks/EDA.ipynb
```

### 7. Run tests

```bash
pytest
```

---

## Configuration

All parameters live in `configs/config.yaml`. Key sections:

| Section | Controls |
|---|---|
| `paths` | Data and output directory locations |
| `dataset` | Patient IDs, feature list, resample frequency |
| `window` | Lookback / forecast horizon in minutes |
| `feature_engineering` | Savitzky-Golay smoothing, interpolation, rolling windows |
| `evaluation` | Number of CV folds, SVR sub-sample size, metrics |
| `models` | Per-model hyperparameters + `enabled: true/false` toggle |
| `deep_learning` | Epochs, batch size, callbacks, LSTM/GRU architecture |
| `seed` | Global reproducibility seed |

---

## Evaluation Strategy

This replicates the strategy from the original exploratory notebook exactly:

1. **5-Fold KFold (no shuffle)** — preserves temporal ordering for classical models.
2. **Hold-out test split** — official OhioT1DM train/test XML files are used as-is.
3. **Deep learning CV proxy** — best validation loss from training history (no temporal leakage).
4. **Metrics:** RMSE (primary), MAE, R², MARD (Mean Absolute Relative Difference %).
5. **Two feature conditions:** full features vs. glucose-only baseline.

ARIMA is kept as the glucose-only baseline. SARIMAX/ARIMAX is the multivariate
ARIMA-style model and uses the aligned engineered feature columns as exogenous
inputs.

---

## Feature Engineering

| Feature | Description |
|---|---|
| `glucose_level` | CGM signal, Savitzky-Golay smoothed (window=11, poly=2); linear interpolation for gaps <=30 min |
| `basis_gsr` | GSR wearable signal snapped to the nearest CGM timestamp, then forward-filled for model windows |
| `basis_sleep` | Sleep interval mask on the CGM grid; 1 while asleep, otherwise 0 |
| `total_insulin` | Bolus dose + basal rate x 5 min / 60 (units per window); insulin events are snapped to the nearest CGM timestamp before aggregation |
| `insulin_count` | Number of bolus events per 5-min window |
| `insulin_3h_std` | Rolling 3-hour std of total insulin (variability indicator) |
| `daily_exercise` | Binary flag: 1 for rows on a day with exercise, otherwise 0 |

Features with very high missingness (`acceleration`, `basis_steps`,
`basis_air_temperature`, `basis_heart_rate`) are excluded from
the engineered model input. Missingness reporting is limited to regularly checked signals
configured in `dataset.missing_report_features`.

All non-CGM timestamps are rounded to the nearest `glucose_level` timestamp before
windowing or correlation. Correlation diagnostics compare only rows that share the same
timestamp after this alignment. The ADF helper drops invalid values and returns a
non-testable result for constant or too-short series instead of raising.

---

## Output

After a successful run you will find:

| Path | Contents |
|---|---|
| `output/results/model_results.csv` | Per-model CV RMSE, CV R², Test RMSE, Test MAE, Test R², Test MARD |
| `output/plots/rmse_r2_bars.png` | Grouped bar charts for all metrics |
| `output/plots/feature_impact.png` | ΔRMSE and ΔR² (with features − baseline) |
| `output/plots/best_model_predictions.png` | Time-series + scatter for the best model |
| `output/plots/horizon_rmse.png` | Per-step RMSE across the 30-min horizon (LSTM & GRU) |
| `output/plots/summary_heatmap.png` | Heatmap of Test RMSE and R² for all model × feature combos |
| `output/logs/bg_prediction_YYYYMMDD_HHMMSS.log` | Full run log |

---

## Dataset Citation

> Marling, C. & Bunescu, R. (2020). *The OhioT1DM Dataset for Blood Glucose Level
> Prediction: Update 2020*. KHD@IJCAI 2020.

> Marling, C. & Bunescu, R. (2018). *The OhioT1DM Dataset for Blood Glucose Level
> Prediction*. KHD@IJCAI 2018.

---

## License

This repository contains no patient data. Model code is released under the MIT License.
The OhioT1DM dataset is subject to its own data-use agreement — see the dataset website.
