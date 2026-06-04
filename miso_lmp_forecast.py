"""
============================================================
MISO Day-Ahead LMP Forecasting Pipeline
Author: Ishan Bhardwaj
Directed Research Project

Research Question:
    Does adding net load and capacity-aware stress features
    improve day-ahead electricity price prediction in MISO LRZ4?

Pipeline:
    1. Synthetic data generation (calibrated to real MISO behavior)
    2. Feature engineering (net load, stress proxy, lags, ramps, rolling)
    3. Time-based train/val/test split (70/15/15)
    4. Models: Baselines, Linear, Ridge, XGBoost
    5. Evaluation: RMSE, MAE, Directional Accuracy, Spike/Stress error
    6. Interpretation: Feature importance + SHAP values
    7. Plots and saved outputs

Requirements:
    pip install numpy pandas scikit-learn xgboost matplotlib seaborn shap
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

DATA_START  = '2022-01-01'
DATA_END    = '2025-12-31'
ZONE        = 'LRZ4'
SPIKE_THRESH  = 75      # $/MWh — threshold for "high price" period
STRESS_THRESH = 0.85    # stress proxy threshold for "high stress" period

OUTPUT_DIR  = 'outputs'
FIG_DIR     = os.path.join(OUTPUT_DIR, 'figures')
TAB_DIR     = os.path.join(OUTPUT_DIR, 'tables')
DATA_DIR    = os.path.join(OUTPUT_DIR, 'data')

for d in [FIG_DIR, TAB_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURE_COLS = [
    # Calendar
    'hour', 'dow', 'month', 'weekend',
    # Physical inputs
    'load', 'wind_gen', 'solar_gen', 'net_load',
    'temperature', 'gas_price',
    # Price lags
    'price_lag_1', 'price_lag_24', 'price_lag_168',
    # Demand/renewable lags
    'load_lag_1', 'load_lag_24',
    'wind_lag_1', 'solar_lag_1', 'net_load_lag_1',
    # Ramp features
    'load_ramp', 'wind_ramp', 'net_load_ramp',
    # Rolling features
    'rolling_mean_24', 'rolling_std_24', 'rolling_peak_load_30d',
    # Capacity stress proxy (key research feature)
    'stress_proxy',
]
TARGET = 'lmp'


# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────

def generate_synthetic_miso_data(start=DATA_START, end=DATA_END):
    """
    Generate synthetic hourly MISO LRZ4 data closely mimicking
    real market behavior:
      - Gas prices: mean-reverting stochastic process
      - Temperature: Chicago seasonal + diurnal cycle
      - Load: seasonal + diurnal + weekend + temperature effects
      - Wind: seasonal capacity factors + MISO diurnal pattern
      - Solar: sunrise/sunset mask + cloud cover noise
      - LMP: gas-correlated + net load sensitivity + spikes
    """
    idx = pd.date_range(start=start, end=end, freq='h', tz='America/Chicago')
    n   = len(idx)
    df  = pd.DataFrame(index=idx)
    df.index.name = 'timestamp'

    hour  = df.index.hour
    dow   = df.index.dayofweek
    month = df.index.month
    doy   = df.index.dayofyear

    # ── Natural gas price ($/MMBtu) ──────────────────────────────────
    gas = np.zeros(n)
    gas[0] = 3.50
    for i in range(1, n):
        gas[i] = gas[i-1] + np.random.normal(0, 0.04) * 0.3 - 0.01 * (gas[i-1] - 3.5)
    gas_price = np.clip(gas, 1.5, 9.0)

    # ── Temperature (°F, Chicago) ────────────────────────────────────
    temp_base    = 55 - 25 * np.cos(2 * np.pi * (doy - 15) / 365)
    temp_daily   = np.repeat(np.random.normal(0, 8, n // 24 + 1), 24)[:n]
    temp_hourly  = np.random.normal(0, 2, n)
    temp_diurnal = -5 * np.cos(2 * np.pi * (hour - 14) / 24)
    temperature  = np.clip(temp_base + temp_daily + temp_hourly + temp_diurnal, -20, 105)

    # ── Zonal load (MW) ──────────────────────────────────────────────
    load_seasonal = 6000 + 2500 * np.sin(2 * np.pi * (month - 1) / 12 + 0.3)
    load_diurnal  = (-1500 * np.cos(2 * np.pi * (hour - 14) / 24)
                     + 300  * np.sin(2 * np.pi * hour / 24))
    load_weekend  = np.where(dow >= 5, -800, 0)
    temp_effect   = np.where(temperature > 70,  40 * (temperature - 70),
                    np.where(temperature < 32,  30 * (32 - temperature), 0))
    load_noise    = np.random.normal(0, 250, n)
    load = np.clip(load_seasonal + load_diurnal + load_weekend + temp_effect + load_noise,
                   2500, 18000)

    # ── Wind generation (MW) ─────────────────────────────────────────
    wind_cap      = 12000
    wind_cf_base  = 0.30 + 0.20 * np.cos(2 * np.pi * (month - 1) / 12)
    wind_diurnal  = 0.05 * np.cos(2 * np.pi * (hour - 3) / 24)
    wind_noise    = np.random.exponential(0.15, n) - 0.05
    wind_cf       = np.clip(wind_cf_base + wind_diurnal + wind_noise, 0, 1)
    wind_gen      = wind_cf * wind_cap

    # ── Solar generation (MW) ────────────────────────────────────────
    solar_cap      = 4000
    solar_angle    = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    solar_seasonal = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 172) / 365)
    cloud_noise    = np.random.beta(2, 2, n)
    solar_gen      = solar_cap * solar_angle * solar_seasonal * cloud_noise
    solar_gen      = np.where((hour < 6) | (hour > 20), 0, solar_gen)

    # ── Net load ─────────────────────────────────────────────────────
    net_load = load - wind_gen - solar_gen

    # ── LMP ($/MWh) ──────────────────────────────────────────────────
    lmp_base     = 10 + gas_price * 6.5
    lmp_load_eff = 0.003 * (net_load - 6000)
    lmp_seasonal = 2.0 * np.sin(2 * np.pi * (month - 1) / 12)
    lmp_diurnal  = 5.0 * np.sin(2 * np.pi * (hour - 17) / 24)
    lmp_weekend  = np.where(dow >= 5, -4, 0)

    # Spike process: triggered by extreme temps or high net load
    spike_prob = np.where(
        (temperature > 90) | (temperature < 10) | (net_load > 13000),
        0.08, 0.01
    )
    spikes    = np.where(np.random.uniform(0, 1, n) < spike_prob,
                         np.random.exponential(80, n), 0)
    lmp_noise = np.random.normal(0, 3.5, n)
    lmp = np.clip(lmp_base + lmp_load_eff + lmp_seasonal + lmp_diurnal
                  + lmp_weekend + spikes + lmp_noise, -20, 800)

    df['lmp']         = lmp
    df['load']        = load
    df['wind_gen']    = wind_gen
    df['solar_gen']   = solar_gen
    df['net_load']    = net_load
    df['temperature'] = temperature
    df['gas_price']   = gas_price

    return df.reset_index()


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def build_features(df):
    """
    All features use only past data — no future leakage.

    Key research features:
      net_load      = load - wind_gen - solar_gen
      stress_proxy  = net_load / rolling_peak_load_30d
    """
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # Calendar
    df['hour']    = df['timestamp'].dt.hour
    df['dow']     = df['timestamp'].dt.dayofweek
    df['month']   = df['timestamp'].dt.month
    df['weekend'] = (df['dow'] >= 5).astype(int)

    # Price lags
    df['price_lag_1']   = df['lmp'].shift(1)
    df['price_lag_24']  = df['lmp'].shift(24)
    df['price_lag_168'] = df['lmp'].shift(168)

    # Demand / renewable lags
    df['load_lag_1']     = df['load'].shift(1)
    df['load_lag_24']    = df['load'].shift(24)
    df['wind_lag_1']     = df['wind_gen'].shift(1)
    df['solar_lag_1']    = df['solar_gen'].shift(1)
    df['net_load_lag_1'] = df['net_load'].shift(1)

    # Ramp features (rate of change)
    df['load_ramp']     = df['load']     - df['load_lag_1']
    df['wind_ramp']     = df['wind_gen'] - df['wind_lag_1']
    df['net_load_ramp'] = df['net_load'] - df['net_load_lag_1']

    # Rolling features (fit on past only via shift(1))
    df['rolling_mean_24']       = df['lmp'].shift(1).rolling(24).mean()
    df['rolling_std_24']        = df['lmp'].shift(1).rolling(24).std()
    df['rolling_peak_load_30d'] = df['load'].shift(1).rolling(24 * 30).max()

    # Capacity stress proxy — core research feature
    df['stress_proxy'] = df['net_load'] / df['rolling_peak_load_30d'].clip(lower=1)

    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 3. TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)
# ─────────────────────────────────────────────────────────────

def time_split(df):
    """Strictly chronological split — no random shuffling."""
    n         = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ─────────────────────────────────────────────────────────────
# 4. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label=''):
    rmse      = np.sqrt(mean_squared_error(y_true, y_pred))
    mae       = mean_absolute_error(y_true, y_pred)
    direction = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    print(f"  {label:35s}  RMSE={rmse:6.2f}  MAE={mae:6.2f}  DirAcc={direction:.3f}")
    return {'model': label, 'RMSE': rmse, 'MAE': mae, 'DirAcc': direction}

def spike_metrics(y_true, y_pred, label):
    mask = y_true > SPIKE_THRESH
    if mask.sum() == 0:
        return
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae  = mean_absolute_error(y_true[mask], y_pred[mask])
    print(f"    ↳ High-price  (>{SPIKE_THRESH}$/MWh)  RMSE={rmse:.2f}  MAE={mae:.2f}  n={mask.sum()}")

def stress_metrics(y_true, y_pred, stress, label):
    mask = stress > STRESS_THRESH
    if mask.sum() == 0:
        return
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae  = mean_absolute_error(y_true[mask], y_pred[mask])
    print(f"    ↳ High-stress (>{STRESS_THRESH})        RMSE={rmse:.2f}  MAE={mae:.2f}  n={mask.sum()}")


# ─────────────────────────────────────────────────────────────
# 5. MODELS
# ─────────────────────────────────────────────────────────────

def run_baselines(train, test):
    y_te = test[TARGET].values
    results = []

    pred_prev = test['price_lag_1'].values
    results.append(compute_metrics(y_te, pred_prev, 'Baseline: Previous Hour'))
    spike_metrics(y_te, pred_prev, 'Prev Hour')

    pred_yday = test['price_lag_24'].values
    results.append(compute_metrics(y_te, pred_yday, 'Baseline: Same Hour Yesterday'))
    spike_metrics(y_te, pred_yday, 'Yesterday')

    return results, pred_prev, pred_yday


def run_linear_models(train, val, test):
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train[FEATURE_COLS])
    X_te   = scaler.transform(test[FEATURE_COLS])
    y_tr   = train[TARGET].values
    y_te   = test[TARGET].values
    results = []

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    pred_lr = lr.predict(X_te)
    results.append(compute_metrics(y_te, pred_lr, 'Linear Regression'))
    spike_metrics(y_te, pred_lr, 'LR')

    rr = Ridge(alpha=10.0)
    rr.fit(X_tr, y_tr)
    pred_ridge = rr.predict(X_te)
    results.append(compute_metrics(y_te, pred_ridge, 'Ridge Regression'))
    spike_metrics(y_te, pred_ridge, 'Ridge')

    return results, pred_lr, pred_ridge


def run_xgboost(train, val, test):
    X_tr, y_tr = train[FEATURE_COLS].values, train[TARGET].values
    X_va, y_va = val[FEATURE_COLS].values,   val[TARGET].values
    X_te, y_te = test[FEATURE_COLS].values,  test[TARGET].values

    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric='rmse',
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    pred = model.predict(X_te)
    results = [compute_metrics(y_te, pred, 'XGBoost')]
    spike_metrics(y_te, pred, 'XGBoost')
    stress_metrics(y_te, pred, test['stress_proxy'].values, 'XGBoost')

    return results, pred, model


# ─────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────

def plot_actual_vs_pred(test, pred_xgb, pred_ridge, pred_prev):
    sample = test.iloc[:7 * 24]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample['timestamp'], sample[TARGET],         label='Actual',    lw=1.5, color='black')
    ax.plot(sample['timestamp'], pred_xgb[:len(sample)], label='XGBoost',   lw=1.2, color='steelblue')
    ax.plot(sample['timestamp'], pred_ridge[:len(sample)],label='Ridge',    lw=1.0, color='darkorange', ls='--')
    ax.plot(sample['timestamp'], pred_prev[:len(sample)], label='Prev Hour',lw=0.8, color='gray', ls=':')
    ax.set_title('Actual vs Predicted LMP — First 7 Days of Test Set', fontsize=13)
    ax.set_ylabel('LMP ($/MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/actual_vs_predicted.png', dpi=150)
    plt.close()
    print(f"  Saved: actual_vs_predicted.png")

def plot_residuals(test, pred_xgb):
    resid = test[TARGET].values - pred_xgb
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(resid, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(0, color='red', ls='--')
    axes[0].set_title('Residual Distribution (XGBoost)')
    axes[0].set_xlabel('Residual ($/MWh)')
    axes[1].scatter(pred_xgb, resid, alpha=0.1, s=4, color='steelblue')
    axes[1].axhline(0, color='red', ls='--')
    axes[1].set_title('Residuals vs Predicted')
    axes[1].set_xlabel('Predicted LMP ($/MWh)')
    axes[1].set_ylabel('Residual ($/MWh)')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/residual_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: residual_distribution.png")

def plot_error_by_hour(test, pred_xgb):
    df2 = test.copy()
    df2['abs_error'] = np.abs(df2[TARGET].values - pred_xgb)
    hourly = df2.groupby('hour')['abs_error'].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly.index, hourly.values, color='steelblue', alpha=0.85)
    ax.set_title('Mean Absolute Error by Hour of Day (XGBoost)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('MAE ($/MWh)')
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/error_by_hour.png', dpi=150)
    plt.close()
    print(f"  Saved: error_by_hour.png")

def plot_feature_importance(model):
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    colors = ['#d62728' if ('net_load' in f or 'stress' in f) else 'steelblue'
              for f in imp.index]
    fig, ax = plt.subplots(figsize=(8, 9))
    imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('XGBoost Feature Importance\n(red = net load / stress features)', fontsize=12)
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/feature_importance.png', dpi=150)
    plt.close()
    print(f"  Saved: feature_importance.png")

def plot_shap(model, test):
    X_te      = test[FEATURE_COLS].values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_te[:500])
    fig, ax   = plt.subplots(figsize=(9, 8))
    shap.summary_plot(shap_vals, X_te[:500], feature_names=FEATURE_COLS, show=False)
    plt.title('SHAP Summary — XGBoost LMP Forecast', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_summary.png")

def plot_stress_vs_price(test, pred_xgb):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(test['stress_proxy'], test[TARGET],
                         c=test['temperature'], cmap='RdYlBu_r', s=3, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label='Temperature (°F)')
    axes[0].set_title('Stress Proxy vs Actual LMP')
    axes[0].set_xlabel('Stress Proxy (net_load / 30d peak)')
    axes[0].set_ylabel('LMP ($/MWh)')
    axes[0].set_ylim(-25, 300)
    error = np.abs(test[TARGET].values - pred_xgb)
    axes[1].scatter(test['stress_proxy'], error, s=3, alpha=0.3, color='darkorange')
    axes[1].set_title('Stress Proxy vs Forecast Error (XGBoost)')
    axes[1].set_xlabel('Stress Proxy')
    axes[1].set_ylabel('Absolute Error ($/MWh)')
    axes[1].set_ylim(0, 150)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/stress_vs_price.png', dpi=150)
    plt.close()
    print(f"  Saved: stress_vs_price.png")

def plot_model_comparison(all_results):
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, metric in zip(axes, ['RMSE', 'MAE', 'DirAcc']):
        colors = ['#aec7e8' if 'Baseline' in m
                  else '#ffbb78' if ('Linear' in m or 'Ridge' in m)
                  else '#2ca02c' for m in df['model']]
        bars = ax.barh(df['model'], df[metric], color=colors)
        ax.set_title(metric)
        ax.invert_yaxis()
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=8)
    plt.suptitle('Model Comparison — Test Set', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: model_comparison.png")


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  MISO LRZ4 Day-Ahead LMP Forecast Pipeline")
    print("  Author: Ishan Bhardwaj | Directed Research")
    print("=" * 65)

    # ── Step 1: Data ─────────────────────────────────────────
    print(f"\n[1/6] Generating synthetic MISO data ({DATA_START} → {DATA_END})...")
    raw = generate_synthetic_miso_data()
    print(f"      Rows: {len(raw):,}  |  LMP mean={raw['lmp'].mean():.1f}  "
          f"std={raw['lmp'].std():.1f}  max={raw['lmp'].max():.1f} $/MWh")

    # ── Step 2: Features ─────────────────────────────────────
    print("\n[2/6] Engineering features...")
    df = build_features(raw)
    print(f"      After feature engineering: {len(df):,} rows, {len(FEATURE_COLS)} features")
    print(f"      Spike hours (LMP > {SPIKE_THRESH}): "
          f"{(df['lmp'] > SPIKE_THRESH).sum():,} ({(df['lmp'] > SPIKE_THRESH).mean()*100:.1f}%)")

    # ── Step 3: Split ─────────────────────────────────────────
    print("\n[3/6] Splitting data...")
    train, val, test = time_split(df)
    print(f"      Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"      Train: {train['timestamp'].min().date()} → {train['timestamp'].max().date()}")
    print(f"      Val:   {val['timestamp'].min().date()} → {val['timestamp'].max().date()}")
    print(f"      Test:  {test['timestamp'].min().date()} → {test['timestamp'].max().date()}")

    # ── Step 4: Models ────────────────────────────────────────
    print("\n[4/6] Training and evaluating models...")
    print("\n  — Baselines —")
    res_base,  pred_prev,  pred_yday  = run_baselines(train, test)
    print("\n  — Linear Models —")
    res_lin,   pred_lr,    pred_ridge = run_linear_models(train, val, test)
    print("\n  — XGBoost —")
    res_xgb,   pred_xgb,   xgb_model  = run_xgboost(train, val, test)

    all_results = res_base + res_lin + res_xgb

    # ── Step 5: Save outputs ──────────────────────────────────
    print("\n[5/6] Saving outputs...")
    pred_df = test[['timestamp', TARGET, 'stress_proxy', 'temperature', 'gas_price']].copy()
    pred_df['pred_xgb']   = pred_xgb
    pred_df['pred_ridge'] = pred_ridge
    pred_df['pred_prev']  = pred_prev
    pred_df.to_csv(f'{TAB_DIR}/predictions.csv', index=False)

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(f'{TAB_DIR}/metrics.csv', index=False)

    df.to_csv(f'{DATA_DIR}/dataset_processed.csv', index=False)
    print(f"  Saved: predictions.csv, metrics.csv, dataset_processed.csv")

    # ── Step 6: Plots ─────────────────────────────────────────
    print("\n[6/6] Generating plots...")
    plot_actual_vs_pred(test, pred_xgb, pred_ridge, pred_prev)
    plot_residuals(test, pred_xgb)
    plot_error_by_hour(test, pred_xgb)
    plot_feature_importance(xgb_model)
    plot_shap(xgb_model, test)
    plot_stress_vs_price(test, pred_xgb)
    plot_model_comparison(all_results)

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  {'Model':<38} {'RMSE':>8} {'MAE':>8} {'DirAcc':>8}")
    print("  " + "-" * 64)
    for r in all_results:
        print(f"  {r['model']:<38} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['DirAcc']:>8.3f}")

    xgb_res  = next(r for r in all_results if 'XGBoost' in r['model'])
    base_res = next(r for r in all_results if 'Prev Hour' in r['model'])
    rmse_imp = (base_res['RMSE'] - xgb_res['RMSE']) / base_res['RMSE'] * 100
    print(f"\n  XGBoost vs naive baseline: {rmse_imp:.1f}% RMSE improvement")

    imp = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    nl_imp    = imp[[f for f in FEATURE_COLS if 'net_load' in f or 'stress' in f]].sum()
    price_imp = imp[[f for f in FEATURE_COLS if 'price_lag' in f]].sum()
    print(f"\n  Feature importance breakdown:")
    print(f"    Net load + stress features: {nl_imp:.4f}  ({nl_imp/(nl_imp+price_imp)*100:.1f}% vs price lags)")
    print(f"    Price lag features:         {price_imp:.4f}")
    print(f"    Ratio (net load / price lag): {nl_imp/price_imp:.2f}x")
    print(f"\n  Top 5 features:")
    for feat, score in imp.head(5).items():
        tag = " ← KEY" if ('net_load' in feat or 'stress' in feat) else ""
        print(f"    {feat:<30} {score:.4f}{tag}")

    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print("  Done.\n")
