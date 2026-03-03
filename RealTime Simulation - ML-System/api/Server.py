"""
api/server.py

HTTP API server that exposes:
  GET /api/predictions  — returns all rows from predictions.csv, flagging fraud (Class=1)
  GET /api/drift        — returns Amount drift alerts only (empty list if no drift detected)
  GET /                 — serves the dashboard HTML
"""

import os
import sys
import json
import threading
import time

import pandas as pd
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from scipy.stats import ks_2samp

# Allow imports from project root (utils.config, etc.)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import Config
from utils.logger import get_logger

app = Flask(__name__, static_folder=".")
CORS(app)

logger = get_logger("api_server")

# ── Paths ────────────────────────────────────────────────────────────────────

PREDICTIONS_PATH = os.path.join(Config.DATA_PROCESSED_DIR, "predictions.csv")
TRAIN_PATH       = os.path.join(Config.DATA_PROCESSED_DIR, "X_train.csv")
BATCH_PATH       = os.path.join(Config.DATA_PROCESSED_DIR, "current_batch.csv")

# ── In-memory drift state (updated by background watcher) ────────────────────

_drift_state = {
    "alerts": [],          # list of alert dicts for Amount column
    "last_batch_mtime": None,
}
_drift_lock = threading.Lock()


def _run_drift_check():
    """Background thread: watches current_batch.csv and checks Amount for drift."""
    if not os.path.exists(TRAIN_PATH):
        logger.warning("Training data not found; drift watcher idle.")
        return

    train_data = pd.read_csv(TRAIN_PATH)

    while True:
        try:
            if os.path.exists(BATCH_PATH):
                mtime = os.path.getmtime(BATCH_PATH)

                with _drift_lock:
                    already_processed = (mtime == _drift_state["last_batch_mtime"])

                if not already_processed:
                    recent = pd.read_csv(BATCH_PATH)

                    new_alerts = []

                    if "Amount" in train_data.columns and "Amount" in recent.columns:
                        stat, p_value = ks_2samp(
                            train_data["Amount"].values[:Config.DRIFT_WINDOW_SIZE],
                            recent["Amount"].values,
                        )
                        if p_value < Config.DRIFT_SIGNIFICANCE_LEVEL:
                            new_alerts.append({
                                "feature": "Amount",
                                "p_value": round(p_value, 6),
                                "ks_stat": round(stat, 6),
                                "threshold": Config.DRIFT_SIGNIFICANCE_LEVEL,
                            })

                    with _drift_lock:
                        _drift_state["alerts"] = new_alerts
                        _drift_state["last_batch_mtime"] = mtime

        except Exception as exc:
            logger.error(f"Drift watcher error: {exc}")

        time.sleep(1)


# Start drift watcher in daemon thread
_watcher = threading.Thread(target=_run_drift_check, daemon=True)
_watcher.start()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "dashboard.html")


@app.route("/api/predictions")
def get_predictions():
    """Return all predictions. Rows where prediction==1 are flagged as fraud."""
    if not os.path.exists(PREDICTIONS_PATH):
        return jsonify({"error": "predictions.csv not found", "rows": []}), 404

    try:
        df = pd.read_csv(PREDICTIONS_PATH)

        logger.info(f"predictions.csv columns: {df.columns.tolist()}")

        # Normalize column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Find the prediction column — could be 'prediction' or last column
        if "prediction" not in df.columns:
            # Fallback: assume last column is predictions
            df = df.rename(columns={df.columns[-1]: "prediction"})
            logger.warning(f"'prediction' column not found, using last column as prediction")

        # Convert to numeric, coerce errors
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce").fillna(0).astype(int)

        # Keep only relevant display columns if they exist
        keep = ["Time", "Amount", "prediction"]
        cols = [c for c in keep if c in df.columns]
        df_display = df[cols].copy()
        df_display["is_fraud"] = df_display["prediction"] == 1

        rows = df_display.tail(500).to_dict(orient="records")  # latest 500 rows
        total       = len(df)
        fraud_count = int((df["prediction"] == 1).sum())

        return jsonify({
            "total": total,
            "fraud_count": fraud_count,
            "rows": rows,
        })

    except Exception as exc:
        logger.error(f"/api/predictions error: {exc}")
        return jsonify({"error": str(exc), "rows": []}), 500


@app.route("/api/drift")
def get_drift():
    """Return Amount drift alerts. Empty list when no drift is detected."""
    with _drift_lock:
        alerts = list(_drift_state["alerts"])

    return jsonify({"drift_detected": len(alerts) > 0, "alerts": alerts})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting API server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)