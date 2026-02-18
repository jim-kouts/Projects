# train.py
# ------------------------------------------------------------
# Train a baseline model (logreg OR histgradientboosting) with:
# - internal validation split
# - preprocessing pipeline
# - save model + metrics
#
# Outputs:
#   models/model_<model>.joblib
#   reports/train_metrics_<model>.json
#   reports/feature_info.json
# ------------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    parser = argparse.ArgumentParser(description="Train baseline tabular model (logreg or hgb).")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory.")
    parser.add_argument("--target_col", type=str, default="target", help="Target column name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--valid_size", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "hgb"], help="Which model to train.")

    # logreg params
    parser.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength.")
    parser.add_argument("--max_iter", type=int, default=2000, help="LogReg max iterations.")

    # hgb params (keep simple)
    parser.add_argument("--max_depth", type=int, default=6, help="HGB max depth.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="HGB learning rate.")
    parser.add_argument("--max_leaf_nodes", type=int, default=31, help="HGB max leaf nodes.")
    parser.add_argument("--max_bins", type=int, default=255, help="HGB max bins.")
    parser.add_argument("--l2_regularization", type=float, default=0.0, help="HGB L2 regularization.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train")

    reports_dir = Path("reports")
    models_dir = Path("models")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = Path(args.data_dir) / "processed" / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")

    df = pd.read_parquet(train_path)
    if args.target_col not in df.columns:
        raise RuntimeError(f"Target column '{args.target_col}' not found.")

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col].astype(int)

    # Internal validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.valid_size, random_state=args.seed, stratify=y
    )

    # Identify numeric vs categorical columns (robust)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # Save feature info once (overwrites is fine)
    feature_info = {
        "n_features": int(X.shape[1]),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    with open(reports_dir / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    # Preprocessing
    # For HGB: scaling not required, but it doesn't hurt numeric features too much.
    # However, trees don't need scaling. We'll keep the SAME preprocess for simplicity.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Choose model
    if args.model == "logreg":
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="lbfgs")
    else:
        # HistGradientBoosting is strong for tabular and doesn't need extra libraries
        model = HistGradientBoostingClassifier(
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            max_bins=args.max_bins,
            l2_regularization=args.l2_regularization,
            random_state=args.seed,
        )

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    logger.info("Training model=%s ...", args.model)
    clf.fit(X_train, y_train)
    logger.info("Training done.")

    # Quick validation metrics (prob-based)
    y_prob = clf.predict_proba(X_valid)[:, 1]
    roc_auc = float(roc_auc_score(y_valid, y_prob))
    pr_auc = float(average_precision_score(y_valid, y_prob))

    logger.info("VALID: ROC-AUC=%.4f | PR-AUC=%.4f", roc_auc, pr_auc)

    # Save model artifact
    model_path = models_dir / f"model_{args.model}.joblib"
    joblib.dump(clf, model_path)
    logger.info("Saved model: %s", model_path)

    # Save metrics
    metrics = {
        "model": args.model,
        "seed": args.seed,
        "valid_size": args.valid_size,
        "valid_roc_auc": roc_auc,
        "valid_pr_auc": pr_auc,
        "params": {
            "C": args.C,
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "max_bins": args.max_bins,
            "l2_regularization": args.l2_regularization,
        },
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
    }

    metrics_path = reports_dir / f"train_metrics_{args.model}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved metrics: %s", metrics_path)
    logger.info("Done ✅")


if __name__ == "__main__":
    main()
