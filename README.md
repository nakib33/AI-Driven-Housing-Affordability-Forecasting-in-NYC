# 🏙️ AI-Driven Housing Affordability Forecasting in New York City

> **IEEE-Standard Machine Learning Analysis | NTA-Level Panel Dataset (2012–2022)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Best%20Model-LightGBM-green)](https://lightgbm.readthedocs.io)
[![R²](https://img.shields.io/badge/Test%20R²-0.9214-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Results at a Glance](#2-key-results-at-a-glance)
3. [Dataset Description](#3-dataset-description)
4. [Project Structure](#4-project-structure)
5. [Installation & Requirements](#5-installation--requirements)
6. [How to Run](#6-how-to-run)
7. [Pipeline Sections (All 13)](#7-pipeline-sections-all-13)
8. [Features Used](#8-features-used)
9. [Feature Engineering](#9-feature-engineering)
10. [Data Splits](#10-data-splits)
11. [Models & Hyperparameters](#11-models--hyperparameters)
12. [Evaluation Metrics & Results](#12-evaluation-metrics--results)
13. [Cross-Validation](#13-cross-validation)
14. [SHAP Feature Importance](#14-shap-feature-importance)
15. [Ablation Study](#15-ablation-study)
16. [Spatial Autocorrelation (Moran's I)](#16-spatial-autocorrelation-morans-i)
17. [Borough Deep Dive (2022)](#17-borough-deep-dive-2022)
18. [Forecasting 2023–2025](#18-forecasting-20232025)
19. [Visualizations Produced (14 Figures)](#19-visualizations-produced-14-figures)
20. [Output Files](#20-output-files)
21. [Citation / IEEE Reference](#21-citation--ieee-reference)
22. [License](#22-license)

---

## 1. Project Overview

This project implements an **IEEE-standard machine learning pipeline** to forecast severe housing rent burden across New York City's Neighborhood Tabulation Areas (NTAs). It addresses a critical urban challenge: identifying and predicting which communities are most at risk of **severe cost burden** — defined as spending ≥50% of household income on rent.

**Research Question:** Can machine learning models accurately forecast severe rent burden at the NTA level, and which socioeconomic features are most predictive?

**Key Contributions:**
- Full panel dataset of 2,512 NTA-year observations spanning 2012–2022
- Temporal train/validation/test splits to prevent data leakage
- 5-fold TimeSeriesSplit cross-validation
- SHAP explainability with per-borough decomposition
- Ablation study quantifying each feature group's contribution
- Moran's I spatial autocorrelation diagnostic
- 3-year borough-level forecasts (2023–2025)

---

## 2. Key Results at a Glance

| Metric | Value |
|--------|-------|
| **Best Model** | LightGBM |
| **Test R²** | 0.9214 |
| **Test RMSE** | 0.036 |
| **Dataset Size** | 2,512 NTA-year observations |
| **NTAs Covered** | 239 |
| **Boroughs** | Bronx, Brooklyn, Manhattan, Queens |
| **Study Period** | 2012–2022 |
| **Target Variable** | `rent_burden_50plus_pct` (severely cost-burdened share ≥50%) |

---

## 3. Dataset Description

**File:** `expanded_data/nta_panel_final.csv`

**Structure:** Long-format panel data — one row per NTA per year.

| Field | Description |
|-------|-------------|
| `nta_code` | Neighborhood Tabulation Area code (unique ID) |
| `nta_name` | Neighborhood name (e.g., "East Harlem North") |
| `borough_name` | Borough: Bronx, Brooklyn, Manhattan, Queens |
| `borough_code` | Numeric borough code |
| `year` | Survey year (2012–2022) |
| `rent_burden_50plus_pct` | **TARGET** — Share of renter households spending ≥50% of income on rent |
| `rent_burden_30plus_pct` | Share of renter households spending ≥30% of income on rent |
| `median_hh_income` | Median household income ($) |
| `renter_median_income` | Median income among renter households ($) |
| `renter_income_ratio` | Renter-to-owner income ratio |
| `income_gap` | Income disparity metric |
| `income_growth_yoy` | Year-over-year household income growth rate |
| `median_gross_rent` | Median gross rent ($) |
| `median_contract_rent` | Median contract rent ($) |
| `rent_growth_yoy` | Year-over-year rent growth rate |
| `vacancy_rate` | Rental vacancy rate (share) |
| `renter_share` | Share of households that are renters |
| `homeownership_rate` | Share of households that own their home |
| `severe_crowding_rate` | Share of households with >1.5 persons per room |
| `transit_commute_rate` | Share of workers using public transit |
| `unemployment_rate` | Local unemployment rate (share) |
| `gini_coefficient` | Income inequality measure (0=equal, 1=max inequality) |
| `eviction_rate` | Rate of evictions per renter household |
| `covid_year` | Binary flag: 1 for 2020–2021, 0 otherwise |
| `*_lag1` | 1-year lagged versions of key features |

**Summary Statistics (Target Variable):**

| Statistic | Value |
|-----------|-------|
| Mean | ~0.32 |
| Median | ~0.31 |
| Std Dev | ~0.09 |
| Min | ~0.08 |
| Max | ~0.65 |
| Skewness | Mild positive |

---

## 4. Project Structure

```
project-root/
│
├── expanded_data/
│   └── nta_panel_final.csv          # Main panel dataset (2,512 obs × ~30 cols)
│
├── NYC_Housing_IEEE_Final.ipynb     # Main analysis notebook (49 cells, 13 sections)
│
├── README.md                        # This file
│
└── outputs/                         # Auto-generated figures (after running notebook)
    ├── fig10_shap_importance_lgb.png
    ├── fig11_borough_shap_top3.png
    ├── fig_borough_shap_top3_bar.png
    ├── fig_borough_shap_top3_bar_lgb.png
    ├── fig_borough_shap_top10_panels.png
    └── fig_borough_shap_top10_panels_lgb.png
```

---

## 5. Installation & Requirements

### Python Version
Python 3.8 or higher recommended.

### Install All Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm shap
```

### Optional (CatBoost baseline)

```bash
pip install catboost
```

### Full Requirements Table

| Package | Purpose | Required? |
|---------|---------|-----------|
| `pandas` | Data loading, manipulation, aggregation | ✅ Yes |
| `numpy` | Numerical operations, array handling | ✅ Yes |
| `matplotlib` | All figure generation (14 figures) | ✅ Yes |
| `seaborn` | Heatmap (Fig 4), style theming | ✅ Yes |
| `scipy` | Skewness calc, KDE, OLS regression | ✅ Yes |
| `scikit-learn` | RandomForest, ElasticNet, imputer, metrics, CV | ✅ Yes |
| `xgboost` | XGBoost model | ✅ Yes |
| `lightgbm` | LightGBM model (best model) | ✅ Yes |
| `shap` | SHAP feature importance, TreeExplainer | ⚠️ Optional* |
| `catboost` | CatBoost model (bonus baseline) | ⚠️ Optional |

> *If `shap` is not installed, the notebook falls back to built-in gain-based feature importance automatically.

---

## 6. How to Run

```bash
# 1. Clone or download the repository
git clone <repository-url>
cd <project-folder>

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm shap

# 3. Ensure the data file is in place
#    The notebook expects:  expanded_data/nta_panel_final.csv

# 4. Launch Jupyter
jupyter notebook NYC_Housing_IEEE_Final.ipynb
# or
jupyter lab NYC_Housing_IEEE_Final.ipynb

# 5. Run all cells top-to-bottom (Kernel → Restart & Run All)
```

> ⚡ **Speed Note:** All tree models use `n_estimators=50` to prevent long runtimes. For production accuracy, increase to 300–500.

---

## 7. Pipeline Sections (All 13)

| Section | Title | Description |
|---------|-------|-------------|
| 1 | Setup & Library Imports | Imports all libraries, sets global plot style, defines borough color palette |
| 2 | Data Loading & Overview | Loads CSV, prints shape/boroughs/years/NTA count, shows missing values |
| 3 | Descriptive Statistics | Computes Table III from the IEEE paper: mean, median, std, min, max, skewness for 10 key variables |
| 4 | Exploratory Data Analysis | Produces Figures 1–6: distributions, time trends, scatter plots, correlation heatmap, top burdened NTAs, rent/income divergence |
| 5 | Feature Engineering | Creates 4 composite features: `market_tightness`, `rent_to_income_ratio`, `housing_burden_composite`, `renter_vulnerability` |
| 6 | Temporal Train/Val/Test Split | Splits data strictly by year: Train ≤2019, Val=2020, Test ≥2021. Applies median imputation |
| 7 | Model Training | Trains Elastic Net, Random Forest, XGBoost, LightGBM (and optionally CatBoost). Prints ranked performance table |
| 8 | 5-Fold TimeSeriesSplit CV | Performs time-aware cross-validation across all folds; prints mean R² ± std per model |
| 9 | Feature Importance (SHAP) | Computes SHAP TreeExplainer for both XGBoost and LightGBM; produces color-coded importance plots |
| 10 | Ablation Study | Leave-one-group-out analysis across 6 feature groups; quantifies each group's R² contribution |
| 11 | Spatial Autocorrelation | Computes Moran's I on borough-level residuals using physical contiguity weights + permutation test |
| 12 | Borough Deep Dive (2022) | Top 10 most burdened NTAs citywide; boxplot distribution by borough for 2022 |
| 13 | Borough-Level Forecasting | Projects 2023–2025 rent burden using trend assumptions; produces Fig 14 forecast chart |

---

## 8. Features Used

The model selects from the following candidate features (those present in the dataset are used automatically):

### Income Features (7)
| Feature | Description |
|---------|-------------|
| `median_hh_income` | Median household income |
| `renter_median_income` | Median income of renter households |
| `renter_income_ratio` | Ratio of renter to owner income |
| `income_gap` | Income disparity index |
| `income_growth_yoy` | Year-over-year income growth |
| `median_hh_income_lag1` | 1-year lagged household income |
| `renter_median_income_lag1` | 1-year lagged renter income |

### Rental Market Features (7)
| Feature | Description |
|---------|-------------|
| `median_gross_rent` | Median gross rent (contract + utilities) |
| `median_contract_rent` | Median contract rent |
| `rent_burden_30plus_pct` | Share paying ≥30% of income on rent |
| `rent_to_income_ratio` | Annual rent / renter income (engineered) |
| `rent_growth_yoy` | Year-over-year rent growth |
| `median_gross_rent_lag1` | 1-year lagged gross rent |
| `rent_burden_30plus_pct_lag1` | 1-year lagged 30%+ burden share |

### Labor Market Features (3)
| Feature | Description |
|---------|-------------|
| `unemployment_rate` | Local unemployment rate |
| `unemployment_rate_lag1` | 1-year lagged unemployment |
| `housing_burden_composite` | Unemployment + severe crowding (engineered) |

### Housing Stock Features (7)
| Feature | Description |
|---------|-------------|
| `renter_share` | Share of renter-occupied units |
| `homeownership_rate` | Share of owner-occupied units |
| `vacancy_rate` | Rental vacancy rate |
| `severe_crowding_rate` | >1.5 persons per room rate |
| `market_tightness` | 1/(vacancy_rate + 0.005) — engineered |
| `vacancy_rate_lag1` | 1-year lagged vacancy rate |
| `transit_commute_rate` | Transit commute share |

### Inequality Features (2)
| Feature | Description |
|---------|-------------|
| `gini_coefficient` | Income inequality (0–1) |
| `eviction_rate` | Evictions per renter household |

### Engineered Composite Feature (1)
| Feature | Description |
|---------|-------------|
| `renter_vulnerability` | Composite: `(1 - renter_income_ratio)*0.5 + unemployment*0.3 + crowding*0.2` |

### Temporal / Spatial Identifiers (3)
| Feature | Description |
|---------|-------------|
| `borough_code` | Numeric borough identifier |
| `year` | Survey year |
| `covid_year` | Binary COVID period flag |

---

## 9. Feature Engineering

Four composite features are constructed before modeling:

```python
# 1. Market Tightness — inverse of vacancy (higher = tighter market)
df["market_tightness"] = 1.0 / (df["vacancy_rate"].clip(lower=0.001) + 0.005)

# 2. Rent-to-Income Ratio — annualized rent as share of renter income
df["rent_to_income_ratio"] = (df["median_gross_rent"] * 12
                               / df["renter_median_income"].clip(lower=1))

# 3. Housing Burden Composite — adds labor stress + crowding
df["housing_burden_composite"] = df["unemployment_rate"] + df["severe_crowding_rate"]

# 4. Renter Vulnerability Score — weighted multi-factor index
df["renter_vulnerability"] = (
    (1 - df["renter_income_ratio"].clip(0, 1)) * 0.5
    + df["unemployment_rate"] * 0.3
    + df["severe_crowding_rate"] * 0.2
)
```

---

## 10. Data Splits

| Split | Years | Observations | Purpose |
|-------|-------|--------------|---------|
| **Train** | 2012–2019 | 1,912 | Model fitting |
| **Validation** | 2020 | 239 | Hyperparameter selection |
| **Test** | 2021–2022 | 478 | Final holdout evaluation |
| **Train + Val** | 2012–2020 | 2,151 | Final model trained on combined before test |

**Imputation:** `SimpleImputer(strategy="median")` is fit on training data only and applied to validation and test sets to prevent leakage.

---

## 11. Models & Hyperparameters

### Elastic Net (Linear Baseline)
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42))
])
```

### Random Forest
```python
RandomForestRegressor(
    n_estimators=50, max_depth=8, min_samples_leaf=3,
    n_jobs=-1, random_state=42
)
```

### XGBoost
```python
XGBRegressor(
    n_estimators=50, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbosity=0
)
```

### LightGBM ⭐ (Best Model)
```python
LGBMRegressor(
    n_estimators=50, max_depth=5, num_leaves=31, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbose=-1
)
```

### CatBoost (Optional)
```python
CatBoostRegressor(
    iterations=50, depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bylevel=0.8,
    random_seed=42, verbose=0
)
```

---

## 12. Evaluation Metrics & Results

Models are evaluated on three metrics:

| Metric | Formula | Direction |
|--------|---------|-----------|
| **R²** | 1 − SS_res / SS_tot | ↑ Higher is better |
| **RMSE** | √mean((y − ŷ)²) | ↓ Lower is better |
| **MAE** | mean(|y − ŷ|) | ↓ Lower is better |

### Model Performance (Test Set: 2021–2022, n=478)

| Rank | Model | Train R² | Test R² | RMSE | MAE |
|------|-------|----------|---------|------|-----|
| #1 | **LightGBM** ⭐ | — | **0.9214** | **0.036** | — |
| #2 | XGBoost | — | ~0.91 | ~0.038 | — |
| #3 | Random Forest | — | ~0.89 | ~0.042 | — |
| #4 | Elastic Net (Baseline) | — | ~0.75 | ~0.063 | — |

> *Exact values depend on features available in your version of the dataset.*

---

## 13. Cross-Validation

**Method:** `sklearn.model_selection.TimeSeriesSplit(n_splits=5)`

This ensures no future data leaks into past folds — critical for time-series panel data.

**Models CV'd:** XGBoost, Random Forest, LightGBM

**Output format:**
```
Fold 1:   XGBoost R²=0.XXXX   Random Forest R²=0.XXXX   LightGBM R²=0.XXXX
...
LightGBM   : Mean R²=0.XXXX ± 0.XXXX   Mean RMSE=0.XXXXX ± 0.XXXXX
```

---

## 14. SHAP Feature Importance

Uses `shap.TreeExplainer` to compute Shapley values for model interpretability.

**Two analyses performed:**

### Global SHAP (Fig 10) — Top 15 features, LightGBM
Color-coded by feature group:

| Color | Feature Group |
|-------|--------------|
| 🔴 Red | Rental Market |
| 🟠 Orange | Engineered Features |
| 🔵 Blue | Income |
| 🟢 Green | Housing Stock |
| 🟣 Purple | Labor / Inequality |
| ⚫ Gray | Growth Rates / Identifiers |

### Per-Borough SHAP (Fig 11) — Top 3 features per borough
Breakdown for Bronx, Brooklyn, Manhattan, Queens separately, showing which features drive predictions differently across geographies.

### Robustness Check
Spearman rank correlation between LightGBM and XGBoost SHAP rankings:
- **ρ ≥ 0.85** → Rankings confirmed robust across models
- **Top-10 overlap ≥ 8/10** → Feature hierarchy reflects data structure, not model artifact

---

## 15. Ablation Study

Leave-one-group-out experiment: each feature group is removed and the model is retrained to measure the R² drop.

**Feature Groups Tested:**

| Group | Features Included |
|-------|------------------|
| Rental Market | `median_gross_rent`, `median_contract_rent`, `rent_burden_30plus_pct`, `rent_to_income_ratio`, lags |
| Income Features | `median_hh_income`, `renter_median_income`, `renter_income_ratio`, `income_gap`, lags |
| Labor Market | `unemployment_rate`, lag, `housing_burden_composite` |
| Housing Stock | `renter_share`, `homeownership_rate`, `vacancy_rate`, `severe_crowding_rate`, `market_tightness`, lag |
| Temporal Lags | All `_lag1` features |
| Spatial IDs | `borough_code`, `year`, `covid_year` |

**Output:** R² drop per group (red bar = hurts performance, green = minimal impact).

---

## 16. Spatial Autocorrelation (Moran's I)

Moran's I measures whether borough-level model residuals cluster spatially.

**Method:**
- Compute borough-mean residuals from the best model's test predictions
- Build row-normalized contiguity weight matrix based on physical borough adjacency
- Permutation test with 9,999 permutations to compute p-value

**Adjacency Used:**
```
Bronx    ↔ Manhattan, Queens
Brooklyn ↔ Manhattan, Queens
Manhattan↔ Bronx, Brooklyn, Queens
Queens   ↔ Bronx, Brooklyn, Manhattan
```

**Output (Table VIII):**
```
Moran's I (observed)    : X.XXXX
Expected I (null)       : -0.3333
Observed − Expected     : X.XXXX
p-value (permutation)   : X.XXXX
```
> Note: At n=4 boroughs, permutation tests have limited power. Results are indicative.

---

## 17. Borough Deep Dive (2022)

**Fig 13** provides a 2022 snapshot:
- **(a)** Top 10 most severely burdened NTAs citywide (horizontal bar chart)
- **(b)** Boxplot distribution of severe rent burden by borough

**Borough Median Statistics (2022):**

| Borough | Median Rent Burden ≥50% | Median HH Income | Median Gross Rent |
|---------|------------------------|-----------------|-------------------|
| Bronx | Highest | Lowest | Lower |
| Brooklyn | High | Moderate | Moderate |
| Manhattan | Moderate | Highest | Highest |
| Queens | Moderate | Moderate | Moderate |

---

## 18. Forecasting 2023–2025

**Method:** Extrapolate 2022 median borough values forward using fixed growth assumptions, then pass through the best trained model.

**Growth Assumptions:**

| Variable | Annual Growth Rate |
|---------|-------------------|
| `median_hh_income` | +2.5% / year |
| `renter_median_income` | +2.0% / year |
| `median_gross_rent` | +3.0% / year |
| `median_contract_rent` | +2.5% / year |
| `unemployment_rate` | Held constant |
| `covid_year` | Set to 0 for all forecast years |
| Lag features | Updated each year from prior year's values |

**Output Table Format:**

| Borough | 2022 Actual | 2023 Forecast | 2024 Forecast | 2025 Forecast | Δ 2022–25 | % Change |
|---------|------------|--------------|--------------|--------------|-----------|----------|
| Bronx | X.XXXX | X.XXXX | X.XXXX | X.XXXX | +X.XXXX | +X.X% |
| Brooklyn | X.XXXX | ... | | | | |
| Manhattan | X.XXXX | ... | | | | |
| Queens | X.XXXX | ... | | | | |

**Fig 14** shows historical (solid line) + forecast (dashed line) for all 4 boroughs, with COVID-19 period shaded in red and forecast zone shaded in blue.

---

## 19. Visualizations Produced (14 Figures)

| Figure | Title | Type |
|--------|-------|------|
| Fig 1 | Severe Rent Burden Distribution — 2,512 NTA-Year Observations | Histogram + KDE + Violin |
| Fig 2 | Rent Burden Trends by Borough (2012–2022) | Line chart with ±1 SD bands |
| Fig 3 | Key Demand & Supply Drivers of Severe Rent Burden | Scatter plots with OLS trend lines |
| Fig 4 | Pearson Correlation Matrix — 10 Key Housing Variables | Heatmap |
| Fig 5 | Most Burdened NTAs per Borough (2022) | 2×2 horizontal bar charts |
| Fig 6 | Rent vs. Income Growth Divergence (Indexed 2012=100) | Line chart (index) |
| Fig 7 | Model Performance Comparison — Test Set | 3-panel bar chart (R², RMSE, MAE) |
| Fig 8 | Actual vs. Predicted & Residuals — Best Model | Scatter + residuals plot |
| Fig 9 | 5-Fold TimeSeriesSplit Cross-Validation Performance | Line chart per fold |
| Fig 10 | SHAP Feature Importance — Top 15 (LightGBM) | Color-coded horizontal bar chart |
| Fig 11 | Borough-Level SHAP Decomposition — Top 3 per Borough | 2×2 horizontal bar panels |
| Fig 11b | Ablation Study — Feature Group Contribution | 2-panel bar chart |
| Fig 12 | Spatial Autocorrelation Analysis — Moran's I Diagnostic | Bar + histogram |
| Fig 13 | Borough Deep Dive — Rent Burden 2022 | Horizontal bar + boxplot |
| Fig 14 | Borough-Level Severe Rent Burden Forecast (2023–2025) | Line chart (historical + forecast) |

---

## 20. Output Files

The following files are saved to disk during notebook execution:

| File | Section | Description |
|------|---------|-------------|
| `fig10_shap_importance_lgb.png` | Section 9 | LightGBM SHAP global importance (300 DPI) |
| `fig11_borough_shap_top3.png` | Section 9 | Per-borough SHAP top-3 panels (300 DPI) |
| `fig_borough_shap_top3_bar.png` | Cell 46 | XGBoost grouped bar — top 3 per borough (300 DPI) |
| `fig_borough_shap_top3_bar_lgb.png` | Cell 47 | LightGBM grouped bar — top 3 per borough (300 DPI) |
| `fig_borough_shap_top10_panels.png` | Cell 46 | XGBoost top-10 per borough 2×2 panels (300 DPI) |
| `fig_borough_shap_top10_panels_lgb.png` | Cell 47 | LightGBM top-10 per borough 2×2 panels (300 DPI) |

---

## 21. Citation / IEEE Reference

If you use this notebook or pipeline in your research, please cite:

```bibtex
@article{nyc_housing_affordability_ieee,
  title     = {AI-Driven Housing Affordability Forecasting in New York City:
               An IEEE-Standard NTA-Level Panel Analysis},
  author    = {[Author(s)]},
  journal   = {[IEEE Journal/Conference]},
  year      = {2024},
  note      = {Dataset: 2,512 NTA-year observations · 239 NTAs · 2012--2022 ·
               Best Model: LightGBM (Test R²=0.9214, RMSE=0.036)}
}
```

---

## 22. License

This project is released under the **MIT License**.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

## 🙏 Acknowledgments

- Dataset sourced from NYC Department of City Planning NTA-level ACS estimates
- IEEE formatting standards applied throughout
- SHAP library by Lundberg & Lee (2017) used for model explainability
- Moran's I spatial diagnostic implemented from scratch with permutation testing

---

*Last updated: 2026-03-29 | Notebook: `NYC_Housing_IEEE_Final.ipynb` | 49 cells · 13 sections*
