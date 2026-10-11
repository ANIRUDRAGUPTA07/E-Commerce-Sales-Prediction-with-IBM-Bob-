"""
eda_explore.py
--------------
Exploratory Data Analysis for the E-Commerce Sales Prediction project.

Outputs (saved to static/):
  - eda_monthly_sales.png
  - eda_category_sales.png
  - eda_segment_sales.png
  - eda_region_sales.png
  - eda_correlation.png
  - eda_top_products.png
  - eda_summary.txt

Usage:
  python eda_explore.py
"""

import sys, pathlib, textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

ORDERS_CSV   = DATA_DIR / "store_orders.csv"
CUSTOMER_CSV = DATA_DIR / "customer_master.csv"

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (10, 5)


# ── helpers ──────────────────────────────────────────────────────────────────
def save(fig, name: str):
    path = STATIC_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [saved] {path.name}")


def load_orders() -> pd.DataFrame:
    """Load and minimally clean the Superstore orders CSV."""
    if not ORDERS_CSV.exists():
        print(f"[ERROR] {ORDERS_CSV} not found.")
        print("  Run:  python data/download_data.py   OR place the CSV manually.")
        sys.exit(1)

    df = pd.read_csv(ORDERS_CSV, encoding="latin-1")

    # normalise column names
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )

    # required columns (Superstore standard names after normalisation)
    required = {"order_date", "sales", "quantity", "profit", "category", "region", "segment"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing expected columns: {missing}")
        print(f"  Found: {list(df.columns)}")
        sys.exit(1)

    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=False, errors="coerce")
    df.dropna(subset=["order_date", "sales"], inplace=True)
    df["sales"]    = pd.to_numeric(df["sales"],    errors="coerce").fillna(0)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["profit"]   = pd.to_numeric(df["profit"],   errors="coerce").fillna(0)
    df["discount"] = pd.to_numeric(df.get("discount", 0), errors="coerce").fillna(0)

    print(f"[OK] Orders loaded: {len(df):,} rows | {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    return df


def merge_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Join customer dimension if available."""
    if not CUSTOMER_CSV.exists():
        print("[WARN] customer_master.csv not found – skipping customer enrichment.")
        return df
    cust = pd.read_csv(CUSTOMER_CSV)
    cust.columns = cust.columns.str.lower().str.replace(" ", "_")
    # Superstore uses 'customer_id'; our master uses 'customer_id'
    if "customer_id" in df.columns and "customer_id" in cust.columns:
        cust_cols = [c for c in ["customer_id", "customer_age", "gender",
                                  "customer_segment", "customer_acquisition_cost",
                                  "region"] if c in cust.columns]
        # avoid column collision with existing region
        cust_slim = cust[cust_cols].copy()
        if "region" in cust_slim.columns:
            cust_slim = cust_slim.rename(columns={"region": "cust_region"})
        df = df.merge(cust_slim, on="customer_id", how="left")
        matched = df["customer_age"].notna().sum() if "customer_age" in df.columns else 0
        print(f"[OK] Customer enrichment: {matched:,} rows matched.")
    return df


# ── plots ────────────────────────────────────────────────────────────────────
def plot_monthly_sales(df: pd.DataFrame):
    monthly = (
        df.assign(month=df["order_date"].dt.to_period("M"))
          .groupby("month")["sales"].sum()
          .reset_index()
    )
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(monthly["month_dt"], monthly["sales"], marker="o", linewidth=1.5, markersize=3)
    ax.fill_between(monthly["month_dt"], monthly["sales"], alpha=0.15)
    ax.set_title("Monthly Sales Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    save(fig, "eda_monthly_sales.png")


def plot_category_sales(df: pd.DataFrame):
    cat = df.groupby("category")[["sales", "profit"]].sum().reset_index().sort_values("sales", ascending=False)
    x = np.arange(len(cat))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, cat["sales"],  width, label="Sales",  color="#4C72B0")
    ax.bar(x + width/2, cat["profit"], width, label="Profit", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(cat["category"])
    ax.set_title("Sales & Profit by Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("USD ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    fig.tight_layout()
    save(fig, "eda_category_sales.png")


def plot_segment_sales(df: pd.DataFrame):
    seg = df.groupby("segment")["sales"].sum().reset_index().sort_values("sales", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(seg["segment"], seg["sales"], color=sns.color_palette("muted", len(seg)))
    ax.set_title("Sales by Customer Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        ax.text(bar.get_width() + 2000, bar.get_y() + bar.get_height()/2,
                f"${bar.get_width():,.0f}", va="center", fontsize=9)
    fig.tight_layout()
    save(fig, "eda_segment_sales.png")


def plot_region_sales(df: pd.DataFrame):
    reg = df.groupby("region")["sales"].sum().reset_index().sort_values("sales", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        reg["sales"], labels=reg["region"], autopct="%1.1f%%",
        colors=sns.color_palette("pastel"), startangle=140
    )
    ax.set_title("Sales Distribution by Region", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "eda_region_sales.png")


def plot_correlation(df: pd.DataFrame):
    num_cols = ["sales", "quantity", "discount", "profit"]
    num_cols = [c for c in num_cols if c in df.columns]
    if "customer_age" in df.columns:
        num_cols.append("customer_age")
    if "customer_acquisition_cost" in df.columns:
        num_cols.append("customer_acquisition_cost")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "eda_correlation.png")


def plot_top_products(df: pd.DataFrame):
    col = "sub_category" if "sub_category" in df.columns else "category"
    top = (
        df.groupby(col)["sales"].sum()
          .sort_values(ascending=True)
          .tail(12)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index, top.values, color=sns.color_palette("Blues_r", len(top)))
    ax.set_title(f"Top {col.replace('_',' ').title()} by Sales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    save(fig, "eda_top_products.png")


def print_summary(df: pd.DataFrame):
    lines = []
    lines.append("=" * 60)
    lines.append("   E-COMMERCE SALES — EDA SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total records          : {len(df):,}")
    lines.append(f"Date range             : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    lines.append(f"Total revenue          : ${df['sales'].sum():,.2f}")
    lines.append(f"Total profit           : ${df['profit'].sum():,.2f}")
    lines.append(f"Avg order value        : ${df['sales'].mean():.2f}")
    lines.append(f"Unique customers       : {df['customer_id'].nunique() if 'customer_id' in df.columns else 'N/A'}")
    lines.append(f"Categories             : {', '.join(df['category'].unique())}")
    lines.append(f"Regions                : {', '.join(df['region'].unique())}")
    lines.append(f"Segments               : {', '.join(df['segment'].unique())}")
    lines.append(f"Missing values         :\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")
    lines.append("=" * 60)
    summary = "\n".join(lines)
    print("\n" + summary)
    (STATIC_DIR / "eda_summary.txt").write_text(summary)
    print(f"  [saved] eda_summary.txt")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[EDA] Loading data …")
    df = load_orders()
    df = merge_customers(df)

    print("\n[EDA] Generating plots …")
    plot_monthly_sales(df)
    plot_category_sales(df)
    plot_segment_sales(df)
    plot_region_sales(df)
    plot_correlation(df)
    plot_top_products(df)
    print_summary(df)

    print("\n[EDA] Complete. Plots saved to static/")
