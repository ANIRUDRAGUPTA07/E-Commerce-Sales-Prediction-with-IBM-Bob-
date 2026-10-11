# 🛒 E-Commerce Sales Prediction
### AICTE | IBM SkillsBuild — Data Analytics with AI Internship 2026

A production-ready end-to-end Machine Learning project that predicts order-level
sales revenue for an e-commerce business using a Superstore transactions dataset
enriched with CRM customer data.

---

## 📁 Project Structure

```
ecommerce_sales_prediction/
├── data/
│   ├── download_data.py          # Kaggle dataset downloader
│   ├── store_orders.csv          # [place here] Superstore orders
│   └── customer_master.csv       # CRM customer dimension (provided)
├── model/
│   ├── model.pkl                 # Trained best model (auto-generated)
│   ├── scaler.pkl                # StandardScaler artifact
│   ├── label_encoders.pkl        # Categorical encoders
│   ├── feature_names.pkl         # Ordered feature list
│   └── model_metrics.pkl         # Evaluation results dict
├── static/
│   ├── eda_*.png                 # EDA charts (auto-generated)
│   ├── feature_importance.png
│   └── pred_vs_actual.png
├── eda_explore.py                # Exploratory Data Analysis
├── train_model.py                # ML training pipeline
├── app.py                        # Flask REST API (backend)
├── ui.py                         # Streamlit dashboard (frontend)
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

### Primary: Superstore Sales Dataset (Kaggle)
- **URL:** https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
- **File needed:** `SampleSuperstore.csv` → rename to `data/store_orders.csv`
- **Records:** ~9,994 orders | **Years:** 2014–2017 | **Target:** `Sales` ($)
- **Key columns:** `Order Date`, `Category`, `Sub-Category`, `Region`,
  `Segment`, `Sales`, `Quantity`, `Discount`, `Profit`, `Ship Mode`

### Secondary: customer_master.csv (provided)
- CRM dimension table — joined on `Customer ID`
- Adds: `customer_age`, `gender`, `customer_segment`, `customer_acquisition_cost`

> ⚠️ `customer_master.csv` alone has **no sales target** and cannot be used
> for prediction without the Superstore transactions file.

---

## ⚙️ Setup

### 1. Clone / open project
```powershell
cd "ecommerce_sales_prediction"
```

### 2. Create virtual environment (recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Get the dataset

**Option A — Kaggle CLI (automatic):**
```powershell
# First place kaggle.json in %USERPROFILE%\.kaggle\
python data/download_data.py
```

**Option B — Manual download:**
1. Go to https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
2. Download `SampleSuperstore.csv`
3. Copy to `data/` and rename: `store_orders.csv`

---

## 🚀 Running the Project

### Step 1 — EDA (optional but recommended)
```powershell
python eda_explore.py
```
Generates 6 charts + summary in `static/`.

### Step 2 — Train the model
```powershell
python train_model.py
```
Trains Ridge, Random Forest, and Gradient Boosting.  
Saves the best model + artifacts to `model/`.

### Step 3 — Start the Flask API (Terminal 1)
```powershell
python app.py
```
API available at: http://127.0.0.1:5000

### Step 4 — Launch the Streamlit UI (Terminal 2)
```powershell
streamlit run ui.py
```
Dashboard at: http://localhost:8501

---

## 🔌 Flask API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Health check |
| GET  | `/api/metrics` | Model performance (RMSE, MAE, R²) |
| POST | `/api/predict` | Predict sales for an order |
| GET  | `/api/categories` | List valid category/segment values |
| GET  | `/api/eda/<filename>` | Serve a saved chart PNG |

### Example — Predict Sales
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_date": "2024-11-15",
    "quantity": 3,
    "discount": 0.2,
    "category": "Technology",
    "sub_category": "Phones",
    "region": "West",
    "segment": "Consumer",
    "ship_mode": "Standard Class",
    "customer_age": 35,
    "customer_acquisition_cost": 25.0
  }'
```

### Response
```json
{
  "input": { "..." : "..." },
  "predicted_sales": 742.50
}
```

---

## 🧠 ML Pipeline

```
Raw CSV
  ↓ Clean + parse dates
  ↓ Join customer_master (age, acq_cost)
  ↓ Feature Engineering
      temporal: year, month, quarter, day_of_week, month_sin/cos, is_weekend
      interaction: discount × quantity
      aggregation: monthly_sales_total (rolling)
      customer: customer_age, acquisition_cost
      categorical → LabelEncoder
  ↓ Time-based split (last 20% months = test)
  ↓ StandardScaler (Ridge only)
  ↓ Train 3 models + Baseline
  ↓ Evaluate: RMSE, MAE, R²
  ↓ Save best model + artifacts
```

### Models compared

| Model | Notes |
|---|---|
| Baseline (Sub-Cat Median) | Naive benchmark — zero features |
| Ridge Regression | Linear with L2 regularisation |
| Random Forest | 200 estimators, max_depth=12 |
| **Gradient Boosting** | 300 estimators, lr=0.05 (typically best) |

---

## 📊 Streamlit Dashboard Pages

| Page | Content |
|---|---|
| 🏠 Overview | KPI cards + monthly sales trend |
| 📊 EDA Explorer | Interactive Plotly charts — trend, category, region, saved PNGs |
| 🤖 Sales Predictor | Form → calls Flask API → gauge + result |
| 📈 Model Performance | Comparison table, RMSE bar chart, feature importance |
| 📋 Data Preview | Raw orders + customer master tables |

---

## 🧪 Testing

```powershell
# Health check
curl http://127.0.0.1:5000/

# Metrics
curl http://127.0.0.1:5000/api/metrics

# Category list
curl http://127.0.0.1:5000/api/categories
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| pandas, numpy | Data processing |
| scikit-learn | ML models, preprocessing, metrics |
| matplotlib, seaborn | EDA static charts |
| plotly | Interactive Streamlit charts |
| flask | REST API backend |
| streamlit | Interactive dashboard frontend |
| joblib | Model serialisation |
| requests | Streamlit → Flask HTTP calls |
| kaggle | Dataset download helper |

---

## 👤 Project Information

- **Internship:** AICTE | IBM SkillsBuild — Data Analytics with AI 2026
- **Tech Stack:** Python 3.10+ · scikit-learn · Flask · Streamlit · Plotly
- **Platform:** Windows 10/11 · VS Code · No GPU required

---

*Last updated: 2026 | IBM SkillsBuild Internship Project*
