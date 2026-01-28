# ============================================================
# Dataset: Breast Cancer Wisconsin (Diagnostic) from scikit-learn
#
# What this script does:
# 1) Load dataset into a pandas DataFrame
# 2) Simple EDA (shape, counts, describe, missing, correlation)
# 3) Train/Val/Test split (stratified)
# 4) Standard scaling (fit on train only)
# 5) Train 3 models (LogReg, RandomForest, GradientBoosting)
# 6) Evaluate on validation, select best, evaluate on test
# 7) Calibration curve + Brier score for the best model
#
# Outputs:
# - Prints metrics to terminal
# - Saves a few plots into ./outputs_tabular_ml/
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# -----------------------------
# Output directory
# -----------------------------
OUTPUT_DIR = "outputs_tabular_ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load dataset into DataFrame
# -----------------------------
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign in this dataset

print("df.shape:", df.shape)
print("\nHead:")
print(df.head())


# -----------------------------
# Basic EDA
# -----------------------------

print("\nTarget counts:")
print(df["target"].value_counts())

print("\nTarget ratios:")
print(df["target"].value_counts(normalize=True))

print("\nDescribe (summary stats):")
print(df.describe())

print("\nMissing values per column (top):")
print(df.isna().sum().sort_values(ascending=False).head(10))

# Target balance plot (save to file)
plt.figure(figsize=(6, 4))
counts = df["target"].value_counts().sort_index()
plt.bar(counts.index, counts.values)
plt.xticks([0, 1], ["Malignant (0)", "Benign (1)"])
plt.xlabel("Target class")
plt.ylabel("Count")
plt.title("Target Balance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "target_balance.png"), dpi=200)
plt.close()

# Correlation with target (save top correlations to inspect)
corr_with_target = df.corr(numeric_only=True)["target"].sort_values()
print("\nCorrelation with target (lowest -> highest):")
print(corr_with_target.head(8))
print(corr_with_target.tail(8))

# Save correlations to CSV (handy for report/README)
corr_with_target.to_csv(os.path.join(OUTPUT_DIR, "corr_with_target.csv"), header=["corr_with_target"])


# -----------------------------
# Train/Val/Test split
# -----------------------------

X = df.drop(columns=["target"])
y = df["target"]

# First split: train+val vs test (80/20), stratified
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Second split: train vs val from trainval (75/25 of trainval)
# Overall: 60% train, 20% val, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=42,
    stratify=y_trainval,
)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

print("\nClass ratios:")
print("Train ratios:\n", y_train.value_counts(normalize=True))
print("Val ratios:\n",   y_val.value_counts(normalize=True))
print("Test ratios:\n",  y_test.value_counts(normalize=True))


# -----------------------------
# Preprocessing: StandardScaler
# -----------------------------

# Fit scaler on training only -> avoid data leakage
scaler = StandardScaler()
scaler.fit(X_train)

X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print("Scaled shapes:")
print("X_train_s:", X_train_s.shape, "X_val_s:", X_val_s.shape, "X_test_s:", X_test_s.shape)


# -----------------------------
# Train 3 models
# -----------------------------

# Logistic Regression: strong baseline; scaling is important
log_reg = LogisticRegression(max_iter=2000, random_state=42)

# Random Forest: robust non-linear baseline
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

# Gradient Boosting: classic boosting baseline
gb = GradientBoostingClassifier(random_state=42)

models = {
    "LogReg": log_reg,
    "RandomForest": rf,
    "GradBoost": gb,
}

# Fit models on training data
for name, model in models.items():
    model.fit(X_train_s, y_train)
    print(f"Trained: {name}")


# -----------------------------
# Evaluate on validation, pick best, test evaluation
# -----------------------------

threshold = 0.5  # default decision threshold
val_probas = {}
test_probas = {}

# Get probabilities for each model
for name, model in models.items():
    val_probas[name] = model.predict_proba(X_val_s)[:, 1]
    test_probas[name] = model.predict_proba(X_test_s)[:, 1]

# Compute metrics on validation
rows = []
for name, probas in val_probas.items():
    preds = (probas >= threshold).astype(int)

    acc = accuracy_score(y_val, preds)
    auc_val = roc_auc_score(y_val, probas)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    cm = confusion_matrix(y_val, preds)  # [[TN, FP], [FN, TP]]

    rows.append({
        "model": name,
        "val_accuracy": acc,
        "val_roc_auc": auc_val,
        "val_precision": prec,
        "val_recall": rec,
        "val_TN": int(cm[0, 0]),
        "val_FP": int(cm[0, 1]),
        "val_FN": int(cm[1, 0]),
        "val_TP": int(cm[1, 1]),
    })

results_val = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)

print("\nValidation results (sorted by ROC-AUC):")
print(results_val)

# Save validation results to CSV
results_val.to_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False)

# Pick best model by validation ROC-AUC
best_model_name = results_val.loc[0, "model"]
print(f"\nBest model by VAL ROC-AUC: {best_model_name}")

# Evaluate best model on test
best_test_probas = test_probas[best_model_name]
best_test_preds = (best_test_probas >= threshold).astype(int)

test_acc = accuracy_score(y_test, best_test_preds)
test_auc = roc_auc_score(y_test, best_test_probas)
test_prec = precision_score(y_test, best_test_preds, zero_division=0)
test_rec = recall_score(y_test, best_test_preds, zero_division=0)
test_cm = confusion_matrix(y_test, best_test_preds)

print("\n=== Test results (best model) ===")
print(f"Accuracy:  {test_acc:.4f}")
print(f"ROC-AUC:   {test_auc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print("Confusion matrix (TEST):")
print(test_cm)


# -----------------------------
# 7) Calibration curve + Brier score (best model)
# -----------------------------
print("\n=== Calibration (best model) ===")

# Calibration curve: compares predicted probability vs actual frequency
# We compute it on the test set probabilities
prob_true, prob_pred = calibration_curve(y_test, best_test_probas, n_bins=10, strategy="uniform")

# Brier score: lower is better calibrated (for probabilistic predictions)
brier = brier_score_loss(y_test, best_test_probas)

print(f"Brier score (TEST): {brier:.6f}")

# Plot calibration curve
plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title(f"Calibration curve (best={best_model_name})\nBrier={brier:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration_curve.png"), dpi=200)
plt.close()

# Save a summary JSON (useful for README)
summary = {
    "best_model": best_model_name,
    "val_results": results_val.to_dict(orient="records"),
    "test_accuracy": float(test_acc),
    "test_roc_auc": float(test_auc),
    "test_precision": float(test_prec),
    "test_recall": float(test_rec),
    "test_confusion_matrix": test_cm.tolist(),
    "test_brier_score": float(brier),
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved outputs to:", OUTPUT_DIR)
print("Files include: target_balance.png, corr_with_target.csv, validation_results.csv, calibration_curve.png, summary.json")
