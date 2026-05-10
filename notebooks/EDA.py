"""
notebooks/EDA.py
~~~~~~~~~~~~~~~~
Script representation of the EDA notebook.
Convert to .ipynb with:
    jupytext --to notebook notebooks/EDA.py
Or run interactively in Jupyter after converting.
"""
# %% [markdown]
# # OhioT1DM – Exploratory Data Analysis
# **Purpose:** Understand the dataset structure, glucose dynamics, missing data patterns,
# physiological feature distributions, stationarity, and feature-target correlations
# before any modelling is performed.
#
# > This notebook is **EDA only**. All modelling lives in `main.py` and `src/models/`.

# %% Setup
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on path when running from notebooks/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed
from src.data.parser import OhioT1DMParser
from src.data.loader import find_xml_files, build_all_masters

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.dpi"] = 100

cfg = load_config("../configs/config.yaml")
set_global_seed(cfg["seed"])

DATA_DIR   = cfg["paths"]["data_dir"]
PLOTS_DIR  = cfg["paths"]["plots_dir"]
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Config loaded. Data dir:", DATA_DIR)

# %% [markdown]
# ## 1. File Discovery

# %% File discovery
ds_cfg = cfg["dataset"]
train_files, test_files = find_xml_files(
    DATA_DIR,
    ds_cfg["cohorts"],
    ds_cfg.get("train_suffix", "-ws-training.xml"),
    ds_cfg.get("test_suffix",  "-ws-testing.xml"),
)

print(f"Train files: {len(train_files)}")
print(f"Test  files: {len(test_files)}")
for k in sorted(train_files):
    print(f"  {k}: {train_files[k]}")

# %% [markdown]
# ## 2. Missing Data Analysis

# %% Missing data heatmap
FEATURES = ds_cfg["features"]
missing_records = []

for key, path in train_files.items():
    for feat in FEATURES:
        parser = OhioT1DMParser(path, feat)
        info   = parser.missing_stats()
        missing_records.append(dict(patient=key, feature=feat, **info))

missing_df = pd.DataFrame(missing_records)
pivot = missing_df.pivot(index="patient", columns="feature", values="missing_pct")

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Missing %"})
ax.set_title("Missing Data (%) per Patient & Feature", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_missing_heatmap.png"), bbox_inches="tight")
plt.show()

# Glucose missing summary
print("\nGlucose-level missing data:")
display(
    missing_df[missing_df.feature == "glucose_level"][
        ["patient", "actual", "expected", "missing", "missing_pct"]
    ]
)

# %% [markdown]
# ## 3. Raw Glucose Visualisation

# %% Raw glucose plots
n = len(train_files)
fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=False)
if n == 1:
    axes = [axes]

for ax, (key, path) in zip(axes, sorted(train_files.items())):
    df = OhioT1DMParser(path, "glucose_level").make_dataframe()
    ax.plot(df.index, df["value"], lw=0.8, color="royalblue")
    ax.axhline(70,  ls="--", color="red",    lw=0.8, alpha=0.7, label="Hypo (70)")
    ax.axhline(180, ls="--", color="orange", lw=0.8, alpha=0.7, label="Hyper (180)")
    ax.set_ylabel("mg/dL", fontsize=9)
    ax.set_title(f"Patient {key} – Training CGM", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.suptitle("Raw Glucose Levels – All Training Patients", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_raw_glucose.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Glucose Distribution per Patient

# %% Glucose distributions
fig, axes = plt.subplots(2, n // 2 + 1, figsize=(16, 6))
axes = axes.flatten()

for i, (key, path) in enumerate(sorted(train_files.items())):
    df = OhioT1DMParser(path, "glucose_level").make_dataframe()
    gl = df["value"].dropna()
    axes[i].hist(gl, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[i].axvline(gl.mean(), color="red",    ls="--", lw=1.2, label=f"Mean={gl.mean():.0f}")
    axes[i].axvline(gl.median(), color="green", ls="--", lw=1.2, label=f"Med={gl.median():.0f}")
    axes[i].set_title(f"Patient {key}", fontsize=9)
    axes[i].set_xlabel("mg/dL")
    axes[i].legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Glucose Distribution – All Training Patients", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_glucose_distributions.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Time-in-Range (TIR) Analysis

# %% TIR
tir_records = []
for key, path in sorted(train_files.items()):
    df = OhioT1DMParser(path, "glucose_level").make_dataframe()
    gl = df["value"].dropna()
    total = len(gl)
    tir_records.append(dict(
        patient=key,
        hypo_pct   = (gl < 70).sum()   / total * 100,
        normal_pct = ((gl >= 70) & (gl <= 180)).sum() / total * 100,
        hyper_pct  = (gl > 180).sum()  / total * 100,
    ))

tir_df = pd.DataFrame(tir_records)
ax = tir_df.set_index("patient").plot(
    kind="bar", stacked=True, figsize=(12, 5),
    color=["tomato", "seagreen", "orange"], alpha=0.85,
)
ax.set_title("Time-in-Range (%) per Patient", fontsize=13)
ax.set_ylabel("%")
ax.set_xlabel("")
ax.legend(["Hypo (<70)", "Normal (70-180)", "Hyper (>180)"], loc="upper right")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_time_in_range.png"), bbox_inches="tight")
plt.show()
print(tir_df.to_string(index=False))

# %% [markdown]
# ## 6. Stationarity Check (ADF Test)

# %% ADF test
adf_records = []
for key, path in sorted(train_files.items()):
    parser = OhioT1DMParser(path, "glucose_level")
    df     = parser.make_dataframe().resample("5min").mean().interpolate()
    series = df["value"].dropna()
    if len(series) < 20:
        continue
    result = adfuller(series, autolag="AIC")
    adf_records.append(dict(
        patient   = key,
        adf_stat  = round(result[0], 4),
        p_value   = round(result[1], 4),
        stationary= "Yes" if result[1] < 0.05 else "No",
    ))

adf_df = pd.DataFrame(adf_records)
print("Augmented Dickey-Fuller Stationarity Test – Glucose Level")
display(adf_df)

# %% [markdown]
# ## 7. ACF & PACF for a Representative Patient

# %% ACF / PACF
sample_key  = sorted(train_files.keys())[0]
sample_path = train_files[sample_key]
gl_series   = (
    OhioT1DMParser(sample_path, "glucose_level")
    .make_dataframe()
    .resample("5min").mean()
    .interpolate()["value"]
    .dropna()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(gl_series,  lags=60, ax=axes[0], title=f"ACF – Patient {sample_key}")
plot_pacf(gl_series, lags=60, ax=axes[1], title=f"PACF – Patient {sample_key}")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_acf_pacf.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Feature Engineering & Correlation Analysis

# %% Build masters
train_masters = build_all_masters(train_files, cfg)
all_train     = pd.concat(list(train_masters.values()), ignore_index=True)

feature_cols = ["total_insulin", "insulin_count", "insulin_3h_std", "daily_exercise"]

# %% Pearson & Spearman correlation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pearson_r, spearman_r = {}, {}

for col in feature_cols:
    r, p = stats.pearsonr(all_train[col].fillna(0), all_train["glucose_level"])
    pearson_r[col]  = (round(r, 4), round(p, 4))
    r2, p2 = stats.spearmanr(all_train[col].fillna(0), all_train["glucose_level"])
    spearman_r[col] = (round(r2, 4), round(p2, 4))

pearson_df  = pd.DataFrame(pearson_r,  index=["r", "p-value"]).T
spearman_df = pd.DataFrame(spearman_r, index=["r", "p-value"]).T

for ax, corr_df, title, xlabel in [
    (axes[0], pearson_df,  "Pearson r vs Glucose Level",  "Pearson r"),
    (axes[1], spearman_df, "Spearman ρ vs Glucose Level", "Spearman ρ"),
]:
    colors = ["steelblue" if v >= 0 else "tomato" for v in corr_df["r"]]
    ax.barh(corr_df.index, corr_df["r"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)

plt.suptitle("Feature–Target Correlation", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_correlation.png"), bbox_inches="tight")
plt.show()

print("\nPearson:"); display(pearson_df)
print("\nSpearman:"); display(spearman_df)

# %% [markdown]
# ## 9. Engineered Feature Distributions

# %% Feature distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, col in zip(axes.flatten(), feature_cols):
    data = all_train[col].dropna()
    sns.histplot(data, bins=60, ax=ax, color="steelblue", kde=True)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("")
plt.suptitle("Engineered Feature Distributions (All Patients)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_feature_distributions.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Insulin vs Glucose Relationship

# %% Insulin vs glucose scatter (sub-sampled for readability)
sample = all_train.sample(min(5000, len(all_train)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(sample["total_insulin"], sample["glucose_level"],
                c=sample["insulin_3h_std"], cmap="viridis", alpha=0.4, s=10)
plt.colorbar(sc, ax=ax, label="insulin_3h_std")
ax.set_xlabel("Total Insulin per 5-min Window")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_title("Total Insulin vs Glucose (coloured by 3-h Insulin Std)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eda_insulin_glucose_scatter.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 11. Summary Statistics

# %% Summary table
summary = all_train.describe().T
display(summary)
summary.to_csv(os.path.join(cfg["paths"]["results_dir"], "eda_summary_stats.csv"))
print("EDA complete. Plots saved to:", PLOTS_DIR)
