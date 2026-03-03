# Real-Time-like Credit Card Fraud Detection

A machine learning pipeline that simulates a live transaction stream, runs fraud predictions in real time, monitors for data drift, and displays everything on a live web dashboard.

---

## What It Does

- Simulates a stream of credit card transactions batch by batch
- Predicts whether each transaction is fraudulent using a trained ML model
- Detects if the `Amount` feature distribution has shifted from training data (data drift)
- Displays predictions and drift alerts on a live dashboard that auto-refreshes every 3 seconds

---

## Project Structure

```
RealTimeSimulation-ML-Sys/
├── api/
│   ├── __init__.py
│   ├── server.py             # Flask API server
│   └── dashboard.html        # Live monitoring dashboard
├── data/
│   ├── raw/                  # Place creditcard.csv here
│   └── processed/            # Auto-generated splits and live batch files
├── models/                   # Saved model and scaler after training
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Splits raw data into train/test sets
│   ├── train_model.py        # Trains and saves the ML model
│   ├── stream_simulator.py   # Simulates live transaction stream
│   ├── online_inference.py   # Runs predictions on each new batch
│   └── drift_detection.py    # Monitors Amount column for distribution shift
├── utils/
│   ├── __init__.py
│   ├── config.py             # Central configuration (paths, thresholds, etc.)
│   └── logger.py             # Shared logger
├── README.md
└── requirements.txt
```

---

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

Download `creditcard.csv` and place it in `data/raw/`.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```


**2. Prepare the data**
```bash
python -m src.data_loader
```

**3. Train the model**
```bash
# Random Forest (default)
python -m src.train_model

# Or XGBoost
python -m src.train_model --model_type xgboost
```

Steps 2 and 3 only need to be run once.

> All commands must be run from the **project root** (`RealTime Simulation-ML-System/`).

---

## Running the Pipeline

Open four separate terminals, all from the project root:

**Terminal 1 — Stream simulator** (writes batches to `current_batch.csv`)
```bash
python -m src.stream_simulator
```

**Terminal 2 — Online inference** (reads batches, writes predictions)
```bash
python -m src.online_inference
```

**Terminal 3 — Drift detection** (monitors Amount distribution)
```bash
python -m src.drift_detection
```

**Terminal 4 — Dashboard server**
```bash
python -m api.server
```

Then open `http://localhost:5000` in your browser.

---

## Dashboard

The dashboard shows:

- **Total predictions** and **fraud count** with fraud rate
- **Prediction feed** — live table of transactions, with fraud rows highlighted in red
- **Drift alerts** — appears only when the `Amount` distribution shifts significantly from training data, with a history log of all past events

---

## Configuration

All settings are in `utils/config.py`:

| Setting | Default | Description |
|---|---|---|
| `STREAM_BATCH_SIZE` | `1` | Transactions per batch |
| `STREAM_DELAY` | `0.5` | Seconds between batches |
| `DRIFT_WINDOW_SIZE` | `500` | Training samples used as drift reference |
| `DRIFT_SIGNIFICANCE_LEVEL` | `0.05` | p-value threshold for drift detection |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

## How Drift Detection Works

Each time a new batch arrives, a [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) compares the `Amount` values in the batch against a reference window from the training data. If the p-value falls below the significance threshold, a drift alert is raised — meaning the live data no longer looks like what the model was trained on, which can degrade prediction quality.