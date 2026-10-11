"""
ui.py  —  Streamlit Dashboard for E-Commerce Sales Prediction
--------------------------------------------------------------
Pages:
  🏠 Home / Overview
  📊 EDA Explorer
  🤖 Sales Predictor (calls Flask API)
  📈 Model Performance
  📋 Data Preview

Usage (after starting app.py in a separate terminal):
  streamlit run ui.py

Or standalone (loads model directly — no Flask needed):
  streamlit run ui.py -- --standalone
"""

import sys, pathlib, datetime, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ── config ────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent
MODEL_DIR  = ROOT / "model"
STATIC_DIR = ROOT / "static"
DATA_DIR   = ROOT / "data"
API_BASE   = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="E-Commerce Sales Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── css ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-card h3 { margin: 0; font-size: 1.8rem; color: #3b82d4; }
  .metric-card p  { margin: 0; color: #57606a; font-size: 0.85rem; }
  .section-header { color: #1f2328; font-size: 1.25rem; font-weight: 600; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_orders():
    path = DATA_DIR / "store_orders.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = (df.columns.str.strip().str.lower()
                             .str.replace(" ", "_").str.replace("-", "_"))
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for c in ["sales", "quantity", "profit", "discount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("order_date")


@st.cache_resource
def load_artifacts():
    files = ["model.pkl", "scaler.pkl", "label_encoders.pkl",
             "feature_names.pkl", "model_metrics.pkl"]
    if not all((MODEL_DIR / f).exists() for f in files):
        return None
    return {
        "model":    joblib.load(MODEL_DIR / "model.pkl"),
        "scaler":   joblib.load(MODEL_DIR / "scaler.pkl"),
        "encoders": joblib.load(MODEL_DIR / "label_encoders.pkl"),
        "features": joblib.load(MODEL_DIR / "feature_names.pkl"),
        "metrics":  joblib.load(MODEL_DIR / "model_metrics.pkl"),
    }


def api_available():
    try:
        r = requests.get(f"{API_BASE}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/200px-IBM_logo.svg.png",
                 width=80)
st.sidebar.title("🛒 Sales Prediction")
st.sidebar.caption("AICTE | IBM SkillsBuild\nData Analytics with AI — 2026")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA Explorer", "🤖 Sales Predictor",
     "📈 Model Performance", "📋 Data Preview"],
    index=0,
)

# API status badge
if api_available():
    st.sidebar.success("Flask API: 🟢 Online")
else:
    st.sidebar.warning("Flask API: 🔴 Offline\n(Predictor uses local model)")

artifacts = load_artifacts()
df_orders = load_orders()

# ── page: Overview ───────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🛒 E-Commerce Sales Prediction")
    st.markdown(
        "**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**  \n"
        "A complete Machine Learning pipeline — from raw Superstore orders to sales forecasting."
    )

    if df_orders is not None:
        c1, c2, c3, c4 = st.columns(4)
        total_sales   = df_orders["sales"].sum()
        total_profit  = df_orders["profit"].sum() if "profit" in df_orders.columns else 0
        total_orders  = len(df_orders)
        date_range    = f"{df_orders['order_date'].min().year} – {df_orders['order_date'].max().year}"

        c1.markdown(f'<div class="metric-card"><h3>${total_sales:,.0f}</h3><p>Total Revenue</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><h3>${total_profit:,.0f}</h3><p>Total Profit</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><h3>{total_orders:,}</h3><p>Total Orders</p></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><h3>{date_range}</h3><p>Date Range</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        # Monthly trend
        monthly = (df_orders.assign(month=df_orders["order_date"].dt.to_period("M"))
                             .groupby("month")["sales"].sum()
                             .reset_index())
        monthly["month_dt"] = monthly["month"].dt.to_timestamp()
        fig = px.area(monthly, x="month_dt", y="sales",
                      title="📅 Monthly Sales Trend",
                      labels={"month_dt": "Month", "sales": "Sales ($)"},
                      color_discrete_sequence=["#3b82d4"])
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Dataset not found. Run `python data/download_data.py` to fetch the Superstore dataset.")

    if artifacts:
        m = artifacts["metrics"]
        st.success(f"✅ Model trained: **{m.get('best_model')}** | "
                   f"R²={m.get('best_r2', 0):.4f} | "
                   f"RMSE=${m.get('best_rmse', 0):.2f}")
    else:
        st.warning("⚠️ Model not yet trained. Run `python train_model.py`.")


# ── page: EDA Explorer ───────────────────────────────────────────────────────
elif page == "📊 EDA Explorer":
    st.title("📊 Exploratory Data Analysis")

    if df_orders is None:
        st.error("Dataset not found.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Sales Trend", "Category Analysis", "Regional View", "Saved Charts"]
    )

    with tab1:
        monthly = (df_orders.assign(month=df_orders["order_date"].dt.to_period("M"))
                             .groupby("month")["sales"].sum()
                             .reset_index())
        monthly["month_dt"] = monthly["month"].dt.to_timestamp()
        fig = px.line(monthly, x="month_dt", y="sales",
                      title="Monthly Sales Revenue",
                      labels={"month_dt": "Month", "sales": "Sales ($)"},
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # Year-over-year
        if df_orders["order_date"].dt.year.nunique() > 1:
            yoy = df_orders.copy()
            yoy["year"]  = yoy["order_date"].dt.year
            yoy["month"] = yoy["order_date"].dt.month
            yoy_grp = yoy.groupby(["year", "month"])["sales"].sum().reset_index()
            fig2 = px.line(yoy_grp, x="month", y="sales", color="year",
                           title="Year-over-Year Sales Comparison",
                           labels={"month": "Month", "sales": "Sales ($)", "year": "Year"})
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            cat = df_orders.groupby("category")[["sales", "profit"]].sum().reset_index()
            fig = px.bar(cat, x="category", y=["sales", "profit"],
                         barmode="group", title="Sales & Profit by Category",
                         labels={"value": "USD ($)", "variable": "Metric"})
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "sub_category" in df_orders.columns:
                sub = df_orders.groupby("sub_category")["sales"].sum().sort_values(ascending=False).head(10)
                fig2 = px.bar(sub.reset_index(), x="sub_category", y="sales",
                              title="Top 10 Sub-Categories by Sales",
                              labels={"sub_category": "Sub-Category", "sales": "Sales ($)"},
                              color="sales", color_continuous_scale="Blues")
                st.plotly_chart(fig2, use_container_width=True)

        if "segment" in df_orders.columns:
            seg = df_orders.groupby("segment")["sales"].sum().reset_index()
            fig3 = px.pie(seg, names="segment", values="sales",
                          title="Sales Share by Customer Segment")
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        reg = df_orders.groupby("region")[["sales", "profit"]].sum().reset_index()
        fig = px.bar(reg, x="region", y="sales", color="profit",
                     title="Sales & Profit by Region",
                     labels={"sales": "Sales ($)", "profit": "Profit ($)"},
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter: discount vs sales
        sample = df_orders.sample(min(1000, len(df_orders)), random_state=42)
        fig2   = px.scatter(sample, x="discount", y="sales",
                            color="category" if "category" in sample.columns else None,
                            title="Discount vs Sales",
                            opacity=0.5,
                            labels={"discount": "Discount Rate", "sales": "Sales ($)"})
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.markdown("#### Saved EDA Charts (generated by `eda_explore.py`)")
        charts = {
            "Monthly Sales Trend":       "eda_monthly_sales.png",
            "Category Sales":            "eda_category_sales.png",
            "Customer Segment Sales":    "eda_segment_sales.png",
            "Regional Sales Share":      "eda_region_sales.png",
            "Correlation Heatmap":       "eda_correlation.png",
            "Top Products / Sub-Cat":    "eda_top_products.png",
            "Feature Importance":        "feature_importance.png",
            "Predicted vs Actual":       "pred_vs_actual.png",
        }
        for label, fname in charts.items():
            path = STATIC_DIR / fname
            if path.exists():
                st.subheader(label)
                st.image(str(path))
            else:
                st.caption(f"⬜ {label} — not yet generated ({fname})")


# ── page: Sales Predictor ────────────────────────────────────────────────────
elif page == "🤖 Sales Predictor":
    st.title("🤖 Sales Predictor")
    st.markdown("Fill in order details and get an instant predicted sales value.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            order_date = st.date_input("Order Date", value=datetime.date.today())
            quantity   = st.number_input("Quantity", min_value=1, max_value=100, value=3)
            discount   = st.slider("Discount Rate", 0.0, 0.9, 0.1, 0.05)

        with col2:
            # populate from encoders if available
            cats    = ["Furniture", "Office Supplies", "Technology"]
            subcats = ["Chairs", "Binders", "Phones", "Tables", "Storage", "Art", "Labels",
                       "Accessories", "Copiers", "Bookcases", "Appliances", "Supplies",
                       "Paper", "Machines", "Fasteners", "Envelopes", "Furnishings"]
            regions  = ["West", "East", "South", "Central"]
            segments = ["Consumer", "Corporate", "Home Office"]
            modes    = ["Standard Class", "Second Class", "First Class", "Same Day"]

            if artifacts:
                enc = artifacts["encoders"]
                cats    = list(enc.get("category",    type("_", (), {"classes_": cats})()).classes_)
                subcats = list(enc.get("sub_category", type("_", (), {"classes_": subcats})()).classes_)
                regions  = list(enc.get("region",   type("_", (), {"classes_": regions})()).classes_)
                segments = list(enc.get("segment",  type("_", (), {"classes_": segments})()).classes_)
                modes    = list(enc.get("ship_mode", type("_", (), {"classes_": modes})()).classes_)

            category    = st.selectbox("Category",     cats)
            sub_category = st.selectbox("Sub-Category", subcats)
            region      = st.selectbox("Region",       regions)

        with col3:
            segment   = st.selectbox("Customer Segment", segments)
            ship_mode = st.selectbox("Ship Mode",        modes)
            cust_age  = st.slider("Customer Age", 18, 80, 40)
            acq_cost  = st.number_input("Acquisition Cost ($)", 5.0, 200.0, 30.0)

        submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)

    if submitted:
        payload = {
            "order_date":                str(order_date),
            "quantity":                  quantity,
            "discount":                  discount,
            "category":                  category,
            "sub_category":              sub_category,
            "region":                    region,
            "segment":                   segment,
            "ship_mode":                 ship_mode,
            "customer_age":              cust_age,
            "customer_acquisition_cost": acq_cost,
            "monthly_sales_total":       15000.0,
        }

        prediction = None
        error_msg  = None

        # 1) try Flask API
        if api_available():
            try:
                resp = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    prediction = data.get("predicted_sales")
                else:
                    error_msg = f"API error {resp.status_code}: {resp.text}"
            except Exception as e:
                error_msg = str(e)

        # 2) fallback: direct model call
        if prediction is None and artifacts:
            try:
                # inline feature builder (same logic as app.py)
                from train_model import build_features as _bf
            except ImportError:
                _bf = None

            if _bf:
                try:
                    X = _bf(payload, artifacts)
                    needs_sc = artifacts["metrics"].get("needs_scaling", False)
                    if needs_sc:
                        X = artifacts["scaler"].transform(X)
                    prediction = float(artifacts["model"].predict(X)[0])
                    prediction = max(0.0, round(prediction, 2))
                except Exception as e:
                    error_msg = f"Local model error: {e}"
            else:
                error_msg = "Train the model first: python train_model.py"

        if prediction is not None:
            st.success(f"### 💰 Predicted Sales: **${prediction:,.2f}**")
            # simple gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"prefix": "$"},
                title={"text": "Predicted Sales"},
                gauge={
                    "axis": {"range": [0, 5000]},
                    "bar":  {"color": "#3b82d4"},
                    "steps": [
                        {"range": [0,    500],  "color": "#fee2e2"},
                        {"range": [500,  2000], "color": "#fef9c3"},
                        {"range": [2000, 5000], "color": "#dcfce7"},
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Input details"):
                st.json(payload)
        else:
            st.error(f"Prediction failed: {error_msg or 'Unknown error'}")


# ── page: Model Performance ──────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    if not artifacts:
        st.warning("Model not trained yet. Run `python train_model.py` first.")
        st.stop()

    m = artifacts["metrics"]
    results = m.get("results", [])

    st.markdown("#### Comparison: All Models vs Baseline")
    res_df = pd.DataFrame(results)
    st.dataframe(res_df.style.format({
        "rmse": "${:.2f}", "mae": "${:.2f}", "r2": "{:.4f}"
    }).highlight_min(subset=["rmse", "mae"], color="#dcfce7")
      .highlight_max(subset=["r2"], color="#dcfce7"),
      use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best RMSE",  f"${m.get('best_rmse',0):.2f}")
    col2.metric("Best MAE",   f"${m.get('best_mae',0):.2f}")
    col3.metric("Best R²",    f"{m.get('best_r2',0):.4f}")

    st.markdown(f"**Winning model:** `{m.get('best_model')}`  |  "
                f"Train: {m.get('train_rows',0):,} rows  |  Test: {m.get('test_rows',0):,} rows")

    # bar chart of RMSE comparison
    fig = px.bar(res_df, x="model", y="rmse",
                 title="RMSE by Model (lower = better)",
                 color="rmse", color_continuous_scale="RdYlGn_r",
                 labels={"rmse": "RMSE ($)", "model": "Model"})
    st.plotly_chart(fig, use_container_width=True)

    # feature importance + pred vs actual
    col_l, col_r = st.columns(2)
    for img_name, col, title in [
        ("feature_importance.png", col_l, "Feature Importance"),
        ("pred_vs_actual.png",     col_r, "Predicted vs Actual"),
    ]:
        path = STATIC_DIR / img_name
        with col:
            st.subheader(title)
            if path.exists():
                st.image(str(path))
            else:
                st.caption(f"Not yet generated. Run `python train_model.py`.")


# ── page: Data Preview ───────────────────────────────────────────────────────
elif page == "📋 Data Preview":
    st.title("📋 Data Preview")

    if df_orders is None:
        st.info("Dataset not found. Run `python data/download_data.py`.")
    else:
        st.subheader("Superstore Orders")
        st.write(f"{len(df_orders):,} rows × {len(df_orders.columns)} columns")
        st.dataframe(df_orders.head(200), use_container_width=True)

        st.subheader("Basic Statistics")
        num_cols = df_orders.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(df_orders[num_cols].describe().T.style.format("{:.2f}"),
                     use_container_width=True)

    cust_path = DATA_DIR / "customer_master.csv"
    if cust_path.exists():
        st.subheader("Customer Master (Dimension Table)")
        cust = pd.read_csv(cust_path)
        st.write(f"{len(cust):,} rows × {len(cust.columns)} columns")
        st.dataframe(cust.head(100), use_container_width=True)
