import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────

def evaluate_classification(model, X_test, y_test) -> dict:
    """
    Evaluate a classification model.
    Returns accuracy, weighted F1, and a full classification_report string.
    """
    logger.info("Evaluating classification model...")
    preds = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds, average="weighted"), 4),
        "report":   classification_report(y_test, preds, zero_division=0),
    }


def evaluate_regression(model, X_test, y_test) -> dict:
    """
    Evaluate a regression model.
    Returns RMSE, MAE, and R2.
    """
    logger.info("Evaluating regression model...")
    preds = model.predict(X_test)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "mae":  round(float(mean_absolute_error(y_test, preds)), 4),
        "r2":   round(float(r2_score(y_test, preds)), 4),
    }


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────

def shap_explain(model, X_sample):
    """
    Generate SHAP values using TreeExplainer.
    Unwraps XGBoostClassifierWrapper automatically if needed.
    """
    logger.info("Generating SHAP explanations...")
    actual_model = model.model if hasattr(model, "model") else model
    explainer    = shap.TreeExplainer(actual_model)
    shap_values  = explainer(X_sample)
    return shap_values


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> plt.Figure:
    """
    Plot a heatmap confusion matrix for a classification model.

    Rows = true severity class, Columns = predicted severity class.
    Diagonal = correct predictions, off-diagonal = misclassifications.
    """
    preds  = model.predict(X_test)
    labels = sorted(y_test.unique())
    cm     = confusion_matrix(y_test, preds, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Pred {l}" for l in labels],
        yticklabels=[f"True {l}" for l in labels],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Severity")
    ax.set_ylabel("True Severity")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Feature Importance Bar Chart
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str) -> plt.Figure:
    """
    Plot a horizontal bar chart of feature importances.

    Works for tree-based models (Random Forest, XGBoost) that expose
    feature_importances_, and for Neural Networks where we fall back to
    the absolute mean of the first-layer weights as a proxy.
    """
    actual_model = model.model if hasattr(model, "model") else model

    # Tree-based models: built-in impurity-based importance
    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_

    # Neural network: use absolute mean of input-layer weights as proxy
    elif isinstance(actual_model, (MLPClassifier, MLPRegressor)):
        importances = np.abs(actual_model.coefs_[0]).mean(axis=1)

    else:
        logger.warning(f"Feature importance not available for {model_name}")
        return None

    # Sort features by importance descending
    indices = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    # Only show top 20 features to keep chart readable
    top_n = min(20, len(sorted_names))
    sorted_names  = sorted_names[:top_n]
    sorted_values = sorted_values[:top_n]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    ax.barh(range(top_n), sorted_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontsize=13, pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Actual vs Predicted (regression)
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(model, X_test, y_test, model_name: str) -> plt.Figure:
    """
    Scatter plot of actual vs predicted values for regression models.
    Points close to the diagonal = accurate predictions.
    """
    preds = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, preds, alpha=0.3, s=10, color="steelblue", label="Predictions")

    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect fit")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=13, pad=12)
    ax.legend()
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Accident Heatmap (hour x day of week)
# ─────────────────────────────────────────────

def plot_accident_heatmap(df) -> plt.Figure:
    """
    Heatmap of accident counts by hour of day (y) and day of week (x).

    Requires 'hour' and 'day_of_week' columns — these are added by
    add_time_features() in feature_engineering.py.

    Hot cells = times with the most accidents.
    """
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        logger.warning("Heatmap requires 'hour' and 'day_of_week' columns.")
        return None

    # Pivot: rows=hour (0-23), columns=day_of_week (0=Mon ... 6=Sun)
    pivot = df.groupby(["hour", "day_of_week"]).size().unstack(fill_value=0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [day_labels[d] for d in pivot.columns if d < len(day_labels)]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Accident Count"},
    )
    ax.set_title("Accident Frequency — Hour of Day × Day of Week", fontsize=13, pad=12)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Correlation Heatmap
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df, features: list) -> plt.Figure:
    """
    Pearson correlation heatmap for the selected feature columns.

    Shows the full matrix with all values annotated.
    Values close to +1 = strong positive correlation (red).
    Values close to -1 = strong negative correlation (blue).
    Values near  0    = little linear relationship (light grey).
    """
    num_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    if len(num_features) < 2:
        logger.warning("Need at least 2 numeric features for correlation heatmap.")
        return None

    corr = df[num_features].corr()
    n    = len(num_features)

    # Dynamic figure size — grows with number of features
    cell_size = 1.1
    fig_w = max(8, n * cell_size + 3)
    fig_h = max(6, n * cell_size + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use a grey background for the axes so near-zero (white) cells are visible
    ax.set_facecolor("#f0f0f0")

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=1.0,          # thicker lines so cells are always separated
        linecolor="#cccccc",
        ax=ax,
        annot_kws={"size": max(7, 11 - n // 3), "color": "black"},
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=max(7, 10 - n // 4))
    ax.tick_params(axis="y", rotation=0,  labelsize=max(7, 10 - n // 4))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Severity Distribution
# ─────────────────────────────────────────────

def plot_severity_distribution(df, target: str) -> plt.Figure:
    """
    Bar chart of how many rows belong to each target class.
    Useful for spotting class imbalance before training.
    """
    if target not in df.columns:
        return None

    counts = df[target].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(counts))),
        edgecolor="white",
    )

    # Add count labels on top of each bar
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.values.max() * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {target}", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    return fig