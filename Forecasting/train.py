"""
Forecasting lite (Bike Sharing Demand):
- Naive baseline: predict next hour as last hour
- Lag-based Linear Regression: predict next hour using lag features
- Backtesting with TimeSeriesSplit (no shuffling) + MAE
- Save a backtesting plot and metrics JSON

Dataset:
- Bike_Sharing_Demand from OpenML via scikit-learn fetch_openml
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# -----------------------------
# Main script
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Forecasting lite: naive vs lag-linear-regression")

    # Data / output
    parser.add_argument("--output_dir", type=str, default="outputs_forecasting", help="Where to save outputs")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of time-series CV splits")
    parser.add_argument("--test_size", type=int, default=3000, help="Test fold size (number of rows)")
    parser.add_argument("--max_train_size", type=int, default=10000, help="Max train size per fold (keeps runtime reasonable)")
    parser.add_argument("--gap", type=int, default=48, help="Gap between train and test to reduce leakage (e.g., 48 hours)")
    parser.add_argument("--lags", type=str, default="1,2,24,168",
                        help="Comma-separated lags to use, e.g. '1,2,24,168'")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # 1) Load dataset
    # -----------------------------
    # The scikit-learn example uses: fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).
    bike = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas")
    df = bike.frame.copy()

    # The target we forecast is "count" (bike demand).
    # We'll ignore other columns.
    y = df["count"].astype(float)

    # -----------------------------
    # 2) Build lagged feature table
    # -----------------------------
    # Convert lags argument "1,2,24,168" -> [1,2,24,168]
    lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]

    # Create a DataFrame with the target and its lagged versions.
    # shift(k) means: value from k steps (hours) in the past.
    lagged_df = pd.DataFrame({"y": y})
    for k in lags:
        lagged_df[f"lag_{k}"] = y.shift(k)

    # Drop rows with NaNs created by shifting (early rows don't have enough history)
    lagged_df = lagged_df.dropna()

    # X is lag columns, y is current value
    X = lagged_df.drop(columns=["y"])
    y_supervised = lagged_df["y"]

    # We'll also keep the original y aligned with X for naive baseline comparisons
    # (y_supervised is already aligned to the same rows as X)
    print("=== Data prepared ===")
    print("Lag features:", list(X.columns))
    print("X shape:", X.shape, "| y shape:", y_supervised.shape)

    # -----------------------------
    # 3) Backtesting setup (time-respecting CV)
    # -----------------------------
    # TimeSeriesSplit ensures:
    # - train indices are always earlier than test indices
    # - no random shuffling (important for forecasting)
    tscv = TimeSeriesSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        gap=args.gap,
        max_train_size=args.max_train_size,
    )

    # We'll store fold MAEs for both methods
    maes_naive = []
    maes_lr = []

    # We'll also store predictions from the last fold for plotting
    last_fold = None

    # -----------------------------
    # 4) Backtesting loop
    # -----------------------------
    # Model: simple Linear Regression
    lr = LinearRegression()

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X, y_supervised), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_supervised.iloc[train_idx], y_supervised.iloc[test_idx]

        # ---- Naive baseline ----
        # Predict y_t as y_{t-1}.
        # Since our supervised y_test is aligned with lagged_df rows,
        # y_{t-1} is simply the "lag_1" feature.
        #
        # If the user didn't include lag_1 in --lags, we can still compute naive by shifting y_supervised.
        if "lag_1" in X_test.columns:
            y_pred_naive = X_test["lag_1"].values
        else:
            # fallback (still simple): use previous value in the supervised series
            y_pred_naive = y_test.shift(1).fillna(method="bfill").values

        mae_naive = mean_absolute_error(y_test, y_pred_naive)
        maes_naive.append(mae_naive)

        # ---- Lag-based Linear Regression ----
        # Fit on past data, predict on future data
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        maes_lr.append(mae_lr)

        print(f"Fold {fold_idx}/{args.n_splits} | MAE naive={mae_naive:.3f} | MAE linear_reg={mae_lr:.3f}")

        # Save last fold details for plotting after loop
        last_fold = {
            "fold_idx": fold_idx,
            "y_test": y_test.values,
            "y_pred_naive": y_pred_naive,
            "y_pred_lr": y_pred_lr,
        }

    # -----------------------------
    # 5) Summarize results
    # -----------------------------
    results = {
        "dataset": "Bike_Sharing_Demand (OpenML)",
        "lags": lags,
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "max_train_size": args.max_train_size,
        "gap": args.gap,
        "mae_naive_folds": [float(x) for x in maes_naive],
        "mae_lr_folds": [float(x) for x in maes_lr],
        "mae_naive_mean": float(np.mean(maes_naive)),
        "mae_lr_mean": float(np.mean(maes_lr)),
    }

    print("\n=== Summary ===")
    print(f"Mean MAE naive: {results['mae_naive_mean']:.3f}")
    print(f"Mean MAE linear regression: {results['mae_lr_mean']:.3f}")

    # Save metrics to JSON (useful for GitHub)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # -----------------------------
    # 6) Plot last fold: actual vs predictions
    # -----------------------------
    # This plot helps you visually compare the baseline vs model.
    # We keep it simple: just lines over the test window.
    y_true = last_fold["y_test"]
    y_naive = last_fold["y_pred_naive"]
    y_lr = last_fold["y_pred_lr"]

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_naive, label="Naive (lag-1)")
    plt.plot(y_lr, label="LinearReg (lags)")
    plt.xlabel("Time steps (test window)")
    plt.ylabel("Demand (count)")
    plt.title(f"Backtest predictions (fold {last_fold['fold_idx']})")
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "backtest_last_fold.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\nSaved:")
    print("-", metrics_path)
    print("-", plot_path)


if __name__ == "__main__":
    main()
