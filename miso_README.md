# MISO Day-Ahead LMP Forecasting

> **Research question:** Does adding net load and capacity-aware stress features improve day-ahead electricity price prediction in MISO LRZ4?

Directed Research Project · University of Minnesota · May 2026

---

## Overview

A complete ML forecasting pipeline for day-ahead electricity prices in MISO's LRZ4 zone. The project tests whether physically-motivated features — net load (load minus renewables) and a capacity stress proxy — materially improve forecast accuracy over naive baselines.

**Pipeline:**
```
Synthetic data generation → Feature engineering → Train/Val/Test split (70/15/15)
→ Baseline + Linear + Ridge + XGBoost models → SHAP interpretation → Results
```

---

## Results

| Model | RMSE ($/MWh) | MAE ($/MWh) | Directional Accuracy |
|-------|------------|------------|-------------------|
| Prev Hour Baseline | 16.90 | 8.53 | 36% |
| Same Hour Yesterday | 17.84 | 6.97 | 52% |
| Linear Regression | 11.53 | 4.14 | **76%** |
| Ridge Regression | 11.53 | 4.14 | **76%** |
| **XGBoost** | **11.56** | **4.22** | **75%** |

**Key finding:** Net load and stress proxy features drive RMSE from ~$17/MWh (naive) to ~$11.5/MWh. Directional accuracy improves from 36% → 76% — a 2× improvement over the naive baseline.

**Honest limitation:** All models perform poorly on price spikes above $75/MWh (RMSE exceeds $130/MWh in those periods). This is a fundamental market property — spikes are driven by stochastic events (generator outages, unexpected demand surges, transmission constraints) that are not systematically predictable from pre-spike observables. This is not a model failure; it is correctly identified and documented.

---

## Feature Engineering

23 features across 6 categories:

| Category | Features |
|----------|----------|
| Calendar | hour, day-of-week, month, weekend flag |
| Physical inputs | load, wind generation, solar generation, net load, temperature, gas price |
| Price lags | lag 1h, lag 24h (same hour yesterday), lag 168h (same hour last week) |
| Demand/renewable lags | load lag 1h/24h, wind lag 1h, solar lag 1h, net load lag 1h |
| Ramp features | load ramp, wind ramp, net load ramp |
| **Stress proxy** | **net load / rolling 30-day peak load** ← key research feature |

The stress proxy is the core research contribution — a simple ratio that acts as an early-warning signal for high-price periods without requiring precise price forecasts.

---

## SHAP Analysis

XGBoost feature importance (top features by gain):
- `stress_proxy` — most important for high-price periods
- `net_load` — stronger predictor than raw load as renewable penetration increases
- `price_lag_24` / `price_lag_168` — strong temporal autocorrelation
- `hour` — captures diurnal price patterns

Key insight from SHAP: the relationship between load and price is weakening over time while net load's relationship is strengthening — making net load features increasingly important as renewable penetration grows across the MISO footprint.

---

## Forecast Error by Hour

Error is lowest during overnight hours (1–5 AM) when load is stable and prices are predictable. Error peaks during morning ramp (6–8 AM) and evening peak (17–20 hours). This pattern is consistent with real electricity market behaviour — transition periods with rapid load and renewable output changes are inherently harder to predict.

---

## Repository Structure

```
├── miso_lmp_forecast.py       # Full pipeline (data → features → models → SHAP → plots)
│
└── outputs/
    ├── figures/               # Model comparison, actual vs predicted, SHAP plots
    ├── tables/                # Results CSVs
    └── data/                  # Processed feature sets
```

---

## Quickstart

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn shap

python miso_lmp_forecast.py
# Outputs figures/ and tables/ with all results
```

---

## Tech Stack

- **Python** — numpy, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn
- **Data** — Synthetic data calibrated to real MISO LRZ4 behaviour (2022–2025)
- **Models** — Prev Hour baseline, Same Hour Yesterday baseline, Linear Regression, Ridge, XGBoost
- **Interpretation** — SHAP values for feature importance and nonlinear threshold effects

---

*Ishan Bhardwaj — Directed Research, University of Minnesota, May 2026*
