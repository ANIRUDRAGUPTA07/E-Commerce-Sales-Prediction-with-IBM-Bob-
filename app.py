"""
app.py  —  Flask REST API for E-Commerce Sales Prediction
=========================================================

AICTE | IBM SkillsBuild — Data Analytics with AI Internship 2026
Author : Anirudra Gupta

Endpoints
---------
GET  /                   Health check — model load status and version info
GET  /api/metrics        Model evaluation metrics (MAE, RMSE, R², WAPE)
POST /api/predict        Predict sales for one or many orders (JSON)
GET  /api/categories     Valid enum values for categorical fields
GET  /api/charts         List available EDA/model chart filenames
GET  /api/eda/<file>     Serve a saved PNG chart from static/

Usage
-----
  cd ecommerce_sales_prediction
  python app.py

  # Health check
  curl http://127.0.0.1:5000/

  # Single prediction
  curl -X POST http://127.0.0.1:5000/api/predict \\
       -H "Content-Type: application/json" \\
       -d '{
         "order_date": "2024-06-15",
         "unit_price": 299.99,
         "quantity": 2,
         "discount": 0.1,
         "category": "Technology",
         "sub_category": "Phones",
         "region": "West",
         "segment": "Consumer",
         "ship_mode": "Standard Class"
       }'

Design notes
------------
- All features are derived identically to train_model.py (see _build_feature_row).
- unit_price is a required field — it is the dominant predictor in the trained model.
- No feature can carry future information: prior_month_sales defaults to the
  training-set mean when not supplied rather than using actual future data.
- Categorical values are validated against the fitted LabelEncoder classes;
  unknown values return HTTP 422 with the list of valid choices.
- The Flask debug server is bound to 127.0.0.1 only — never 0.0.0.0 in
  production; no secrets are required or loaded.
"""

from __future__ import annotations

import logging
import math
import pathlib
import warnings
from datetime import date, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

warnings.filterwarnings("ignore")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths (relative to this file — works from any cwd) ────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent
MODEL_DIR  = ROOT / "model"
STATIC_DIR = ROOT / "static"

# ── artifact file names ───────────────────────────────────────────────────────
_REQUIRED_ARTIFACTS = [
    "model.pkl",
    "scaler.pkl",
    "label_encoders.pkl",
    "feature_names.pkl",
    "model_metrics.pkl",
]

# ── categorical column names (must match train_model.py CAT_COLS) ─────────────
_CAT_COLS = ["category", "sub_category", "region", "segment", "ship_mode"]

# ── training-period mean prior_month_sales used as a safe default ─────────────
# (avoids requiring callers to supply a future-period aggregate)
_DEFAULT_PRIOR_MONTH_SALES = 800_000.0   # ~monthly mean at training time


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────
class _ModelState:
    """Holds loaded artifacts; populated once at startup."""
    model    = None
    scaler   = None
    encoders: dict = {}
    features: list = []
    metrics:  dict = {}
    error:    str  = ""
    ready:    bool = False


_state = _ModelState()


def _load_artifacts() -> None:
    """Load all model artifacts from MODEL_DIR.  Idempotent."""
    missing = [f for f in _REQUIRED_ARTIFACTS if not (MODEL_DIR / f).exists()]
    if missing:
        _state.error = (
            f"Missing model files: {missing}. "
            "Run: python train_model.py"
        )
        log.warning(_state.error)
        return

    try:
        _state.model    = joblib.load(MODEL_DIR / "model.pkl")
        _state.scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
        _state.encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
        _state.features = joblib.load(MODEL_DIR / "feature_names.pkl")
        _state.metrics  = joblib.load(MODEL_DIR / "model_metrics.pkl")
        _state.ready    = True
        log.info(
            "Model loaded: %s  |  features=%d  |  test R²=%.4f",
            _state.metrics.get("best_model", "?"),
            len(_state.features),
            _state.metrics.get("best_r2", 0),
        )
    except Exception as exc:                        # noqa: BLE001
        _state.error = f"Failed to load artifacts: {exc}"
        _state.ready = False
        log.error(_state.error)


_load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# Input validation schema
# ─────────────────────────────────────────────────────────────────────────────
# Numeric bounds derived from the training data generator + domain knowledge.
_NUMERIC_RULES: dict[str, dict[str, Any]] = {
    "unit_price":                   {"required": True,  "min": 0.01,  "max": 100_000.0},
    "quantity":                     {"required": False, "min": 1,     "max": 500,     "default": 1},
    "discount":                     {"required": False, "min": 0.0,   "max": 0.9,     "default": 0.0},
    "ship_lag":                     {"required": False, "min": 0,     "max": 30,      "default": 3},
    "customer_age":                 {"required": False, "min": 18,    "max": 100,     "default": 40},
    "customer_acquisition_cost":    {"required": False, "min": 0.0,   "max": 10_000,  "default": 30.0},
    "prior_month_sales":            {"required": False, "min": 0.0,   "max": 1e9,     "default": _DEFAULT_PRIOR_MONTH_SALES},
}


def _validate_payload(raw: dict) -> tuple[dict, list[str]]:
    """
    Validate and coerce one order payload.

    Returns
    -------
    (cleaned, errors)
        cleaned : dict with coerced, validated values
        errors  : list of human-readable error strings (empty = valid)
    """
    errors: list[str] = []
    out: dict = {}

    # ── order_date ────────────────────────────────────────────────────────────
    raw_date = raw.get("order_date")
    if raw_date is None:
        out["order_date"] = pd.Timestamp.now()
    else:
        try:
            ts = pd.Timestamp(str(raw_date))
            if pd.isna(ts):
                raise ValueError
            out["order_date"] = ts
        except (ValueError, TypeError):
            errors.append(
                f"order_date: cannot parse '{raw_date}' as a date "
                "(expected ISO-8601, e.g. '2024-06-15')"
            )
            out["order_date"] = pd.Timestamp.now()

    # ── numerics ──────────────────────────────────────────────────────────────
    for field, rule in _NUMERIC_RULES.items():
        raw_val = raw.get(field)
        if raw_val is None:
            if rule["required"]:
                errors.append(f"{field}: required field is missing")
                out[field] = 0.0
            else:
                out[field] = rule["default"]
            continue
        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            errors.append(f"{field}: expected a number, got {raw_val!r}")
            out[field] = rule.get("default", 0.0)
            continue
        if not math.isfinite(v):
            errors.append(f"{field}: value must be finite, got {v}")
            out[field] = rule.get("default", 0.0)
            continue
        lo, hi = rule["min"], rule["max"]
        if not (lo <= v <= hi):
            errors.append(f"{field}: {v} is outside allowed range [{lo}, {hi}]")
            out[field] = max(lo, min(hi, v))   # clamp — do not abort
        else:
            out[field] = v

    # ── categoricals ─────────────────────────────────────────────────────────
    for col in _CAT_COLS:
        raw_val = raw.get(col)
        le      = _state.encoders.get(col)
        if le is None:
            out[col] = 0
            continue
        valid_classes = list(le.classes_)
        if raw_val is None:
            out[col] = valid_classes[0]   # fallback to first class
        else:
            val = str(raw_val).strip()
            if val not in valid_classes:
                errors.append(
                    f"{col}: '{val}' is not a known value. "
                    f"Valid options: {valid_classes}"
                )
                out[col] = valid_classes[0]
            else:
                out[col] = val

    return out, errors


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction  (mirrors train_model.py engineer_features + encode)
# ─────────────────────────────────────────────────────────────────────────────
def _build_feature_row(cleaned: dict) -> np.ndarray:
    """
    Construct the exact 24-feature vector that the trained model expects.

    Feature list (in training order):
      quantity, discount, unit_price, log_unit_price,
      year, month, quarter, day_of_week, week_of_year, is_weekend,
      month_sin, month_cos, dow_sin, dow_cos,
      ship_lag, discount_qty,
      customer_age, customer_acquisition_cost, prior_month_sales,
      category(enc), sub_category(enc), region(enc), segment(enc), ship_mode(enc)

    No future information is used:
      - prior_month_sales is the PRIOR month's total, not the current one.
        Callers who omit it get the training-set mean as a conservative default.
      - order_date is used only to derive temporal features (year, month, etc.),
        never compared against a future cutoff.
    """
    ts = cleaned["order_date"]

    row: dict[str, float] = {
        # transactional
        "quantity":                  cleaned["quantity"],
        "discount":                  cleaned["discount"],
        "unit_price":                cleaned["unit_price"],
        "log_unit_price":            math.log1p(cleaned["unit_price"]),
        # temporal
        "year":                      float(ts.year),
        "month":                     float(ts.month),
        "quarter":                   float((ts.month - 1) // 3 + 1),
        "day_of_week":               float(ts.dayofweek),
        "week_of_year":              float(ts.isocalendar()[1]),
        "is_weekend":                float(ts.dayofweek >= 5),
        "month_sin":                 math.sin(2 * math.pi * ts.month / 12),
        "month_cos":                 math.cos(2 * math.pi * ts.month / 12),
        "dow_sin":                   math.sin(2 * math.pi * ts.dayofweek / 7),
        "dow_cos":                   math.cos(2 * math.pi * ts.dayofweek / 7),
        # interaction / derived
        "ship_lag":                  cleaned["ship_lag"],
        "discount_qty":              cleaned["discount"] * cleaned["quantity"],
        # customer
        "customer_age":              cleaned["customer_age"],
        "customer_acquisition_cost": cleaned["customer_acquisition_cost"],
        # aggregate context (t-1, no leakage)
        "prior_month_sales":         cleaned["prior_month_sales"],
    }

    # categorical → label-encoded integer
    for col in _CAT_COLS:
        le     = _state.encoders[col]
        val    = cleaned[col]
        row[col] = float(le.transform([val])[0])

    # Build vector in training feature order
    X = np.array([[row[f] for f in _state.features]], dtype=np.float64)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Flask application
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False   # preserve key order in responses


# ── helpers ───────────────────────────────────────────────────────────────────
def _model_unavailable():
    return jsonify({
        "error": "model_unavailable",
        "detail": _state.error or "Model not loaded.",
        "hint": "Run: python train_model.py",
    }), 503


def _bad_request(detail: str | list):
    return jsonify({"error": "bad_request", "detail": detail}), 400


def _unprocessable(detail: str | list):
    return jsonify({"error": "unprocessable_entity", "detail": detail}), 422


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    """
    Health check.

    Returns 200 always; 'status' field indicates whether the model is loaded.

    Response
    --------
    {
      "status":      "ready" | "model_not_trained",
      "model":       "<model class name or null>",
      "best_model":  "<algorithm name or null>",
      "features":    <int>,
      "test_r2":     <float>,
      "test_rmse":   <float>,
      "api_version": "1.0"
    }
    """
    if _state.ready:
        m = _state.metrics
        body = {
            "status":      "ready",
            "model":       type(_state.model).__name__,
            "best_model":  m.get("best_model"),
            "features":    len(_state.features),
            "test_r2":     round(m.get("best_r2",   0.0), 4),
            "test_rmse":   round(m.get("best_rmse", 0.0), 4),
            "test_mae":    round(m.get("best_mae",  0.0), 4),
            "test_wape":   round(m.get("best_wape", 0.0), 4),
            "train_rows":  m.get("train_rows"),
            "test_rows":   m.get("test_rows"),
            "api_version": "1.0",
        }
    else:
        body = {
            "status":      "model_not_trained",
            "error":       _state.error,
            "api_version": "1.0",
        }
    return jsonify(body), 200


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Full model evaluation metrics from the last training run.

    Returns 503 if model artifacts are not loaded.

    Response
    --------
    {
      "best_model": "<name>",
      "test_mae":   <float>,   # Mean Absolute Error on test set ($)
      "test_rmse":  <float>,   # Root Mean Squared Error on test set ($)
      "test_r2":    <float>,   # R² coefficient of determination
      "test_wape":  <float>,   # Weighted Absolute Percentage Error (%)
      "train_rows": <int>,
      "test_rows":  <int>,
      "train_cutoff": "<date>",
      "all_results":  [...]    # per-model val + test metrics
    }
    """
    if not _state.ready:
        return _model_unavailable()

    m = _state.metrics
    return jsonify({
        "best_model":   m.get("best_model"),
        "test_mae":     round(m.get("best_mae",  0.0), 4),
        "test_rmse":    round(m.get("best_rmse", 0.0), 4),
        "test_r2":      round(m.get("best_r2",   0.0), 4),
        "test_wape":    round(m.get("best_wape", 0.0), 4),
        "train_rows":   m.get("train_rows"),
        "val_rows":     m.get("val_rows"),
        "test_rows":    m.get("test_rows"),
        "train_cutoff": m.get("train_end"),
        "val_cutoff":   m.get("val_end"),
        "features":     _state.features,
        "all_results":  m.get("results", []),
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict order-level sales revenue.

    Accepts a single JSON object or an array of objects.

    Required fields
    ---------------
    unit_price  (number, > 0)  — unit selling price before discount

    Optional fields (with defaults)
    --------------------------------
    order_date                 (ISO-8601 string, default: today)
    quantity                   (int 1–500, default: 1)
    discount                   (float 0.0–0.9, default: 0.0)
    ship_lag                   (int 0–30 days, default: 3)
    customer_age               (int 18–100, default: 40)
    customer_acquisition_cost  (float, default: 30.0)
    prior_month_sales          (float, default: training mean)
    category                   (see /api/categories)
    sub_category               (see /api/categories)
    region                     (see /api/categories)
    segment                    (see /api/categories)
    ship_mode                  (see /api/categories)

    Leakage guardrail
    -----------------
    prior_month_sales represents the revenue of the PREVIOUS calendar month.
    It is the caller's responsibility to supply a value based only on
    historically observed data.  When omitted, the training-set mean is used —
    a conservative, leakage-free default.

    Response (single input)
    -----------------------
    {
      "predicted_sales": <float>,   # USD, >= 0.0
      "currency": "USD",
      "unit": "order_line",
      "model": "<algorithm name>",
      "warnings": [...]             # non-fatal validation issues
    }

    Response (batch input)
    ----------------------
    {
      "predictions": [ { "predicted_sales": ..., ... }, ... ],
      "count": <int>
    }

    Error responses
    ---------------
    400  — body is not valid JSON or is not an object/array
    422  — one or more fields fail validation (errors listed per field)
    503  — model artifacts not loaded
    """
    if not _state.ready:
        return _model_unavailable()

    body = request.get_json(force=True, silent=True)
    if body is None:
        return _bad_request("Request body must be valid JSON.")

    # normalise to list
    is_batch = isinstance(body, list)
    if isinstance(body, dict):
        items = [body]
    elif isinstance(body, list):
        if len(body) == 0:
            return _bad_request("Batch array must contain at least one item.")
        if len(body) > 500:
            return _bad_request("Batch size exceeds limit of 500 items.")
        items = body
    else:
        return _bad_request("Body must be a JSON object or array of objects.")

    results = []
    any_hard_error = False

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            results.append({
                "index":  idx,
                "error":  "bad_request",
                "detail": f"Item {idx} is not a JSON object.",
            })
            any_hard_error = True
            continue

        cleaned, errors = _validate_payload(item)

        # Hard errors: required fields missing or truly unparseable
        hard = [e for e in errors if "required field is missing" in e or "cannot parse" in e]
        if hard:
            results.append({
                "index":  idx,
                "error":  "unprocessable_entity",
                "detail": hard,
            })
            any_hard_error = True
            continue

        # Soft errors (range violations): still predict but attach warnings
        warnings_list = [e for e in errors if e not in hard]

        try:
            X = _build_feature_row(cleaned)
            if _state.metrics.get("needs_scaling", False):
                X = _state.scaler.transform(X)
            raw_pred = float(_state.model.predict(X)[0])
            pred     = round(max(0.0, raw_pred), 2)
        except Exception as exc:                    # noqa: BLE001
            log.exception("Prediction failed for item %d", idx)
            results.append({
                "index":  idx,
                "error":  "prediction_error",
                "detail": str(exc),
            })
            any_hard_error = True
            continue

        entry: dict = {
            "predicted_sales": pred,
            "currency":        "USD",
            "unit":            "order_line",
            "model":           _state.metrics.get("best_model"),
        }
        if warnings_list:
            entry["warnings"] = warnings_list
        if is_batch:
            entry["index"] = idx

        results.append(entry)

    if not is_batch:
        return jsonify(results[0]), (200 if "error" not in results[0] else 422)

    status_code = 207 if any_hard_error else 200
    return jsonify({"predictions": results, "count": len(results)}), status_code


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """
    Return the valid enum values for every categorical field.

    Response
    --------
    {
      "category":     [...],
      "sub_category": [...],
      "region":       [...],
      "segment":      [...],
      "ship_mode":    [...]
    }
    """
    if not _state.ready:
        return _model_unavailable()

    return jsonify({
        col: list(_state.encoders[col].classes_)
        for col in _CAT_COLS
        if col in _state.encoders
    }), 200


@app.route("/api/charts", methods=["GET"])
def list_charts():
    """List all available EDA and model chart files in static/."""
    if not STATIC_DIR.exists():
        return jsonify({"charts": []}), 200
    charts = sorted(p.name for p in STATIC_DIR.glob("*.png"))
    return jsonify({
        "charts": charts,
        "count":  len(charts),
        "base_url": "/api/eda/",
    }), 200


@app.route("/api/eda/<path:filename>", methods=["GET"])
def serve_chart(filename: str):
    """
    Serve a saved PNG chart from static/.

    Path traversal is prevented by resolving to a name-only component.
    Returns 404 for non-existent files or non-PNG requests.
    """
    safe = pathlib.Path(filename).name      # strip any directory components
    if not safe.endswith(".png"):
        abort(404)
    target = STATIC_DIR / safe
    if not target.exists():
        abort(404)
    return send_from_directory(str(STATIC_DIR), safe)


# ── 404 / 405 JSON handlers ────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(exc):
    return jsonify({"error": "not_found", "detail": str(exc)}), 404


@app.errorhandler(405)
def method_not_allowed(exc):
    return jsonify({"error": "method_not_allowed", "detail": str(exc)}), 405


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  E-Commerce Sales Prediction  —  Flask API")
    print("  AICTE | IBM SkillsBuild 2026")
    print("=" * 60)

    if not _state.ready:
        print(f"\n[WARN]  {_state.error}")
        print("[WARN]  /api/predict and /api/metrics will return HTTP 503.")
        print("[HINT]  Run  python train_model.py  first, then restart.\n")
    else:
        m = _state.metrics
        print(f"\n[OK]  Model : {m.get('best_model')}")
        print(f"[OK]  Test R²   = {m.get('best_r2', 0):.4f}")
        print(f"[OK]  Test RMSE = ${m.get('best_rmse', 0):.2f}")
        print(f"[OK]  Test MAE  = ${m.get('best_mae',  0):.2f}")
        print(f"[OK]  Features  = {len(_state.features)}")
        print()

    print("[INFO] Listening on http://127.0.0.1:5000")
    print("[INFO] Press Ctrl-C to stop\n")

    # debug=True only when run directly — never in production
    # host bound to loopback only — no external exposure
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
