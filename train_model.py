"""
train_model.py
--------------
Reproducible, standalone ML training pipeline for E-Commerce Sales Prediction.

AICTE | IBM SkillsBuild — Data Analytics with AI Internship 2026
Author: Anirudra Gupta

Inputs  : data/store_orders.csv, data/customer_master.csv
Outputs : model/model.pkl, model/scaler.pkl, model/label_encoders.pkl,
          model/feature_names.pkl, model/model_metrics.pkl
          static/*.png  (EDA and model charts)

Usage:
  cd ecommerce_sales_prediction
  python train_model.py

No GPU required. Runs on Windows in VS Code.
"""

import warnings, pathlib, sys, time
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

# ── reproducibility
SEED = 42
np.random.seed(SEED)

# ── paths (relative to this file)
ROOT       = pathlib.Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "model"
STATIC_DIR = ROOT / "static"
MODEL_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

ORDERS_CSV   = DATA_DIR / "store_orders.csv"
CUSTOMER_CSV = DATA_DIR / "customer_master.csv"

# split boundaries
TRAIN_END = pd.Timestamp("2023-06-30")
VAL_END   = pd.Timestamp("2023-12-31")

sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    if not ORDERS_CSV.exists():
        print(f"[ERROR] {ORDERS_CSV} not found.")
        print("  Run: python data/generate_transactions.py")
        sys.exit(1)
    if not CUSTOMER_CSV.exists():
        print(f"[ERROR] {CUSTOMER_CSV} not found.")
        sys.exit(1)

    orders = pd.read_csv(ORDERS_CSV)
    cust   = pd.read_csv(CUSTOMER_CSV)

    print(f"[LOAD] Orders  : {len(orders):,} rows x {orders.shape[1]} cols")
    print(f"[LOAD] Customers: {len(cust):,} rows x {cust.shape[1]} cols")
    return orders, cust


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────
def clean(raw_orders: pd.DataFrame, raw_cust: pd.DataFrame) -> pd.DataFrame:
    df = raw_orders.copy()

    # duplicates
    before = len(df)
    df = df.drop_duplicates()
    n_dup = before - len(df)

    # dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"]  = pd.to_datetime(df["ship_date"],  errors="coerce")
    n_null_date = df["order_date"].isna().sum()
    df = df.dropna(subset=["order_date"])

    # zero/neg sales
    n_bad_sales = (df["sales"] <= 0).sum()
    df = df[df["sales"] > 0].copy()

    # discount clip
    df["discount"] = df["discount"].clip(0, 0.9)

    # outlier cap (99.9th pct)
    cap = df["sales"].quantile(0.999)
    n_cap = (df["sales"] > cap).sum()
    df["sales"] = df["sales"].clip(upper=cap)

    # fill ship_date nulls
    mask = df["ship_date"].isna()
    df.loc[mask, "ship_date"] = df.loc[mask, "order_date"] + pd.Timedelta(days=5)

    # join customer dimension
    cust_slim = raw_cust[["customer_id","customer_age","customer_acquisition_cost"]].copy()
    df = df.merge(cust_slim, on="customer_id", how="left")
    for col in ["customer_age","customer_acquisition_cost"]:
        df[col] = df[col].fillna(df[col].median())

    df = df.sort_values("order_date").reset_index(drop=True)

    print(f"[CLEAN] Removed: {n_dup} dups, {n_null_date} null-date rows, {n_bad_sales} bad-sales rows, "
          f"{n_cap} outliers capped")
    print(f"[CLEAN] Final  : {len(df):,} rows | "
          f"{df['order_date'].min().date()} to {df['order_date'].max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()

    # temporal
    d["year"]         = d["order_date"].dt.year
    d["month"]        = d["order_date"].dt.month
    d["quarter"]      = d["order_date"].dt.quarter
    d["day_of_week"]  = d["order_date"].dt.dayofweek
    d["week_of_year"] = d["order_date"].dt.isocalendar().week.astype(int)
    d["is_weekend"]   = (d["day_of_week"] >= 5).astype(int)
    d["month_sin"]    = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"]    = np.cos(2 * np.pi * d["month"] / 12)
    d["dow_sin"]      = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["dow_cos"]      = np.cos(2 * np.pi * d["day_of_week"] / 7)

    # transaction
    d["ship_lag"]       = (d["ship_date"] - d["order_date"]).dt.days.clip(0, 14).fillna(5)
    d["discount_qty"]   = d["discount"] * d["quantity"]
    d["log_unit_price"] = np.log1p(d["unit_price"])

    # prior-month revenue (no leakage: using t-1 month)
    d["ym"] = d["order_date"].dt.to_period("M")
    monthly = (d.groupby("ym")["sales"].sum()
                .reset_index()
                .rename(columns={"sales":"monthly_sales_total"}))
    monthly["ym_prev"] = monthly["ym"] - 1
    monthly = monthly[["ym_prev","monthly_sales_total"]].rename(columns={"ym_prev":"ym"})
    monthly.columns = ["ym","prior_month_sales"]
    d = d.merge(monthly, on="ym", how="left")
    d["prior_month_sales"] = d["prior_month_sales"].fillna(d["sales"].mean())
    d.drop(columns=["ym"], inplace=True)

    return d


CAT_COLS = ["category","sub_category","region","segment","ship_mode"]

FEATURE_COLS = [
    "quantity","discount","unit_price","log_unit_price",
    "year","month","quarter","day_of_week","week_of_year","is_weekend",
    "month_sin","month_cos","dow_sin","dow_cos",
    "ship_lag","discount_qty",
    "customer_age","customer_acquisition_cost","prior_month_sales",
    "category","sub_category","region","segment","ship_mode",
]
TARGET = "sales"


def encode(df_feat: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df_feat.copy()
    if encoders is None:
        encoders = {}
    for col in CAT_COLS:
        df[col] = df[col].astype(str).fillna("Unknown")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
    return df, encoders


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def report(name, split, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    w    = wape(y_true, y_pred)
    print(f"  {name:35s} | {split:5s} | MAE=${mae:8.2f} | RMSE=${rmse:8.2f} | R2={r2:.4f} | WAPE={w:.2f}%")
    return {"model": name, "split": split, "mae": mae, "rmse": rmse, "r2": r2, "wape": w}


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def save_eda_charts(df: pd.DataFrame):
    plt.rcParams.update({"figure.dpi": 110})

    # monthly trend
    monthly = (df.assign(m=df["order_date"].dt.to_period("M"))
                 .groupby("m")["sales"].sum().reset_index())
    monthly["dt"] = monthly["m"].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly["dt"], monthly["sales"], marker="o", markersize=3, linewidth=1.6)
    ax.fill_between(monthly["dt"], monthly["sales"], alpha=0.13)
    ax.set_title("Monthly Sales Revenue", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
    fig.tight_layout(); fig.savefig(STATIC_DIR/"eda_monthly_sales.png", bbox_inches="tight"); plt.close(fig)

    # category bar
    cat = df.groupby("category")[["sales","profit"]].sum().reset_index()
    x   = np.arange(len(cat)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x-w/2, cat["sales"],  w, label="Sales",  color="#4C72B0")
    ax.bar(x+w/2, cat["profit"], w, label="Profit", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(cat["category"])
    ax.set_title("Sales & Profit by Category", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
    ax.legend(); fig.tight_layout(); fig.savefig(STATIC_DIR/"eda_category.png", bbox_inches="tight"); plt.close(fig)

    # correlation
    num_cols = [c for c in ["sales","profit","quantity","discount","unit_price",
                             "customer_age","customer_acquisition_cost"] if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.tight_layout(); fig.savefig(STATIC_DIR/"eda_correlation.png", bbox_inches="tight"); plt.close(fig)

    print("[CHARTS] EDA charts saved to static/")


def save_model_charts(best_name, best_model, best_X, y_test, feat_cols):
    # feature importance
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feat_cols).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(imp.index, imp.values, color=sns.color_palette("Blues_r", len(imp)))
        ax.set_title(f"Feature Importances — {best_name}", fontweight="bold")
        ax.set_xlabel("Importance")
        fig.tight_layout(); fig.savefig(STATIC_DIR/"feature_importance.png", bbox_inches="tight"); plt.close(fig)

    # pred vs actual
    y_pred = best_model.predict(best_X)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.25, s=6, color="#4C72B0")
    lim = [0, max(float(y_test.max()), float(y_pred.max()))]
    ax.plot(lim, lim, "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual ($)"); ax.set_ylabel("Predicted ($)")
    ax.set_title(f"Predicted vs Actual — {best_name}", fontweight="bold")
    ax.legend(); fig.tight_layout()
    fig.savefig(STATIC_DIR/"pred_vs_actual.png", bbox_inches="tight"); plt.close(fig)
    print("[CHARTS] Model charts saved to static/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*65)
    print("  E-Commerce Sales Prediction — Training Pipeline")
    print("  AICTE | IBM SkillsBuild 2026  |  Author: Anirudra Gupta")
    print("="*65 + "\n")

    # ── 1. Load
    print("[STEP 1] Loading data ...")
    raw_orders, raw_cust = load_data()

    # ── 2. Clean
    print("\n[STEP 2] Cleaning data ...")
    df = clean(raw_orders, raw_cust)

    # ── 3. EDA charts
    print("\n[STEP 3] Generating EDA charts ...")
    save_eda_charts(df)

    # ── 4. Features
    print("\n[STEP 4] Engineering features ...")
    df_feat = engineer_features(df)
    df_enc, encoders = encode(df_feat, fit=True)
    feat_cols = [c for c in FEATURE_COLS if c in df_enc.columns]
    print(f"  Feature columns ({len(feat_cols)}): {feat_cols}")

    # ── 5. Split
    print("\n[STEP 5] Splitting data (time-based) ...")
    train_df = df_enc[df_enc["order_date"] <= TRAIN_END].copy()
    val_df   = df_enc[(df_enc["order_date"] > TRAIN_END) & (df_enc["order_date"] <= VAL_END)].copy()
    test_df  = df_enc[df_enc["order_date"] > VAL_END].copy()

    X_train, y_train = train_df[feat_cols].values, train_df[TARGET].values
    X_val,   y_val   = val_df[feat_cols].values,   val_df[TARGET].values
    X_test,  y_test  = test_df[feat_cols].values,  test_df[TARGET].values

    print(f"  Train : {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── 6. Models
    print("\n[STEP 6] Training models ...")
    print(f"  {'Model':35s} | split | {'MAE':>10} | {'RMSE':>10} | {'R2':>7} | WAPE")
    print("  " + "-"*80)

    results = []

    # Baseline
    sub_median  = train_df.groupby("sub_category")[TARGET].median().to_dict()
    global_med  = float(y_train.mean())
    le_sub      = encoders["sub_category"]

    def bl_pred(dfs):
        decoded = le_sub.inverse_transform(dfs["sub_category"].values.astype(int))
        return np.array([sub_median.get(d, global_med) for d in decoded])

    results.append(report("Baseline (SubCat Median)", "val",  y_val,  bl_pred(val_df)))
    results.append(report("Baseline (SubCat Median)", "test", y_test, bl_pred(test_df)))

    # Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_s, y_train)
    results.append(report("Ridge Regression", "val",  y_val,  ridge.predict(X_val_s)))
    results.append(report("Ridge Regression", "test", y_test, ridge.predict(X_test_s)))

    # Random Forest
    print("  [training RandomForest — may take 30s]")
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=4,
                               max_features=0.6, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    results.append(report("Random Forest", "val",  y_val,  rf.predict(X_val)))
    results.append(report("Random Forest", "test", y_test, rf.predict(X_test)))

    # Gradient Boosting
    print("  [training GradientBoosting — may take 60s]")
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                   subsample=0.8, min_samples_leaf=5, random_state=SEED)
    gb.fit(X_train, y_train)
    results.append(report("Gradient Boosting", "val",  y_val,  gb.predict(X_val)))
    results.append(report("Gradient Boosting", "test", y_test, gb.predict(X_test)))

    # ── 7. Best model
    test_res  = [r for r in results if r["split"]=="test" and not r["model"].startswith("Baseline")]
    best_rec  = min(test_res, key=lambda r: r["rmse"])
    best_name = best_rec["model"]
    model_map = {
        "Ridge Regression":  (ridge, X_test_s, True),
        "Random Forest":     (rf,    X_test,   False),
        "Gradient Boosting": (gb,    X_test,   False),
    }
    best_model, best_X, needs_scaling = model_map[best_name]
    y_pred_best = best_model.predict(best_X)

    print(f"\n[BEST]  {best_name}")
    print(f"  Test MAE  : ${mean_absolute_error(y_test, y_pred_best):.2f}")
    print(f"  Test RMSE : ${np.sqrt(mean_squared_error(y_test, y_pred_best)):.2f}")
    print(f"  Test R2   : {r2_score(y_test, y_pred_best):.4f}")
    print(f"  Test WAPE : {wape(y_test, y_pred_best):.2f}%")

    # ── 8. Charts
    print("\n[STEP 7] Saving model charts ...")
    save_model_charts(best_name, best_model, best_X, y_test, feat_cols)

    # ── 9. Save artifacts
    print("\n[STEP 8] Saving model artifacts ...")
    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    joblib.dump(scaler,     MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders,   MODEL_DIR / "label_encoders.pkl")
    joblib.dump(feat_cols,  MODEL_DIR / "feature_names.pkl")
    joblib.dump({
        "best_model":    best_name,
        "best_rmse":     float(np.sqrt(mean_squared_error(y_test, y_pred_best))),
        "best_mae":      float(mean_absolute_error(y_test, y_pred_best)),
        "best_r2":       float(r2_score(y_test, y_pred_best)),
        "best_wape":     float(wape(y_test, y_pred_best)),
        "train_rows":    int(len(X_train)),
        "val_rows":      int(len(X_val)),
        "test_rows":     int(len(X_test)),
        "features":      feat_cols,
        "needs_scaling": needs_scaling,
        "results":       results,
        "train_end":     str(TRAIN_END.date()),
        "val_end":       str(VAL_END.date()),
    }, MODEL_DIR / "model_metrics.pkl")

    for f in sorted(MODEL_DIR.iterdir()):
        print(f"  model/{f.name}  ({f.stat().st_size:,} bytes)")

    elapsed = time.time() - t0
    print(f"\n[DONE] Total time: {elapsed:.1f}s")
    print("="*65)


if __name__ == "__main__":
    main()
