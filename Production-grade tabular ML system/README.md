# Production-grade Tabular ML System

An end-to-end tabular ML project that mimics how models are built and operated in realistic workflows:
- programmatic dataset ingestion
- data validation (“data contracts”)
- reproducible training with saved artifacts
- evaluation with plots (ROC / PR / calibration / confusion matrix / feature importance)
- batch inference to Parquet
- drift monitoring (PSI) + synthetic drift simulation
- baseline model comparison (Logistic Regression vs HistGradientBoosting)

The goal is not just a model, but a **repeatable pipeline** with clear outputs that could run as a scheduled job.

---

## Project structure

├── make_dataset.py
├── validate_data.py
├── train.py
├── evaluate.py
├── predict_batch.py
├── monitor_drift.py
├── simulate_drift.py
├── compare_models.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
├── reports/
└── predictions/






Folders like `data/`, `reports/`, `models/`, `predictions/` are created automatically by scripts if missing.

---

## Dataset

This project uses a tabular binary classification dataset downloaded from **OpenML**.

- Each row corresponds to one sample (client record)
- The target is binary (0/1) and is stored as `target` after preprocessing
- In this OpenML version, the target column appeared as `y`, so dataset creation uses `--target_col y` and the pipeline stores it internally as `target`

> Note: The dataset used here is roughly class-balanced (~50/50). This is fine for demonstrating the pipeline and model comparison.

---

## Setup

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn openml matplotlib joblib pyarrow
# or
pip install -r requirements.txt


# Deactivate venv
deactivate



# How to run (end-to-end)
# Step 1 — Download + prepare dataset
python make_dataset.py --openml_id 45020 --target_col y


# Step 2 — Validate data
python validate_data.py


# Step 3 — Train models (baseline + stronger model)
# Train Logistic Regression baseline:
python train.py --model logreg

# Train HistGradientBoosting model:
python train.py --model hgb



# Step 4 — Evaluate models + save plots
# Evaluate LogReg:
python evaluate.py --model_path models/model_logreg.joblib --out_dir reports/logreg

# Evaluate HGB:
python evaluate.py --model_path models/model_hgb.joblib --out_dir reports/hgb



# Step 5 — Compare models
python compare_models.py


# Step 6 — Batch inference (Parquet-only)
python predict_batch.py --input_path data/processed/test.parquet --threshold 0.5


# Step 7 — Drift monitoring (PSI)
python monitor_drift.py


# Step 8 — Demonstrate drift detection (synthetic drift)
python simulate_drift.py
python monitor_drift.py --current_path data/processed/current_drifted.parquet
