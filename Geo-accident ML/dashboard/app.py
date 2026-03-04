import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.preprocessing import (
    load_data, basic_cleaning,
    drop_irrelevant_columns, encode_features,
    train_test_split_data
)
from src.feature_engineering import add_time_features, add_rush_hour_flag
from src.spatial import add_spatial_clusters
from src.models import get_models, train_models, save_model
from src.evaluation import (
    evaluate_classification, evaluate_regression,
    shap_explain,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_accident_heatmap,
    plot_correlation_heatmap,
    plot_severity_distribution,
)
from src.drift import detect_drift
from utils.config import DATA_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geo Accident ML Dashboard",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Geo Accident Machine Learning Dashboard")
st.markdown("Train, evaluate, and explain ML models on US road accident data.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    run_spatial  = st.checkbox("Add spatial clusters (DBSCAN)", value=True)
    run_feat_eng = st.checkbox("Add time features",             value=True)
    run_drift    = st.checkbox("Run drift detection",           value=False)
    st.markdown("---")
    st.caption("Adjust DATA_PATH in utils/config.py to point to your CSV.")

# ── Load, clean, drop, encode, engineer ───────────────────────────────────────
@st.cache_data(show_spinner="Loading & preparing data...")
def get_data(use_feat_eng, use_spatial):
    df = load_data(DATA_PATH)

    # 1. Drop irrelevant identifier / free-text columns
    df = drop_irrelevant_columns(df)

    # 2. Basic cleaning (drop missing targets, fill NaNs, remove duplicates)
    df = basic_cleaning(df)

    # 3. Encode all string/bool columns to numbers
    #    (True/False -> 1/0, Weather Condition -> integer codes, etc.)
    df = encode_features(df)

    # 4. Optional: add time-based features (hour, month, rush_hour, etc.)
    if use_feat_eng and "Start_Time" in df.columns:
        df = add_time_features(df)
        df = add_rush_hour_flag(df)

    # 5. Optional: add DBSCAN geo cluster label
    if use_spatial and "Start_Lat" in df.columns:
        df = add_spatial_clusters(df)

    return df

df = get_data(run_feat_eng, run_spatial)

# ── Data preview ──────────────────────────────────────────────────────────────
st.subheader("📊 Data Preview")
st.dataframe(df.head(100), use_container_width=True)
st.caption(f"{len(df):,} rows × {df.shape[1]} columns after cleaning & encoding")

# ── Exploratory plots (always visible, no training needed) ────────────────────
st.subheader("📊 Exploratory Analysis")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    # Severity distribution — useful to spot class imbalance before training
    if "Severity" in df.columns:
        fig_sev = plot_severity_distribution(df, "Severity")
        if fig_sev:
            st.pyplot(fig_sev, use_container_width=True)

with exp_col2:
    # Accident heatmap by hour x day of week — requires time features
    if "hour" in df.columns and "day_of_week" in df.columns:
        fig_heat = plot_accident_heatmap(df)
        if fig_heat:
            st.pyplot(fig_heat, use_container_width=True)
    else:
        st.info("Enable 'Add time features' in the sidebar to see the accident heatmap.")

# ── Feature & target selection ────────────────────────────────────────────────
st.subheader("🎯 Model Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    features = st.multiselect(
        "Select Features",
        options=[c for c in df.columns if c != "Severity"],
        default=[c for c in ["hour", "month", "rush_hour", "geo_cluster"] if c in df.columns],
    )
with col2:
    target = st.selectbox(
        "Select Target",
        df.columns,
        index=list(df.columns).index("Severity") if "Severity" in df.columns else 0
    )
with col3:
    problem_type = st.selectbox("Problem Type", ["classification", "regression"])

# Correlation heatmap — updates live whenever features are changed
if len(features) >= 2:
    st.subheader("🔗 Feature Correlation Heatmap")
    fig_corr = plot_correlation_heatmap(df, features)
    if fig_corr:
        st.pyplot(fig_corr, use_container_width=True)

# ── Initialize session state ──────────────────────────────────────────────────
for key in ["trained", "X_train", "X_test", "y_train", "y_test", "problem_type"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Train button ──────────────────────────────────────────────────────────────
if st.button("🚀 Train Models", type="primary", disabled=not features):
    with st.spinner("Splitting data..."):
        X_train, X_test, y_train, y_test = train_test_split_data(df, features, target)

    st.session_state.X_train      = X_train
    st.session_state.X_test       = X_test
    st.session_state.y_train      = y_train
    st.session_state.y_test       = y_test
    st.session_state.problem_type = problem_type

    with st.spinner("Training models — check terminal for live progress..."):
        models = get_models(problem_type)
        st.session_state.trained = train_models(models, X_train, y_train)

    st.success(f"✅ Training complete!  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Results (only shown after training) ───────────────────────────────────────
if st.session_state.trained:
    trained      = st.session_state.trained
    X_train      = st.session_state.X_train
    X_test       = st.session_state.X_test
    y_train      = st.session_state.y_train
    y_test       = st.session_state.y_test
    problem_type = st.session_state.problem_type

    # ── Evaluation ────────────────────────────────────────────────────────────
    st.subheader("📈 Evaluation Results")
    results = {}
    for name, model in trained.items():
        if problem_type == "classification":
            r = evaluate_classification(model, X_test, y_test)
        else:
            r = evaluate_regression(model, X_test, y_test)
        results[name] = r

    metrics_df = pd.DataFrame(
        {name: {k: v for k, v in r.items() if k != "report"} for name, r in results.items()}
    ).T
    st.dataframe(
        metrics_df.style
            .highlight_max(axis=0, color="#d4edda")
            .highlight_min(axis=0, color="#f8d7da"),
        use_container_width=True,
    )

    if problem_type == "classification":
        for name, r in results.items():
            with st.expander(f"Classification Report — {name}"):
                st.code(r["report"])

    # ── Feature Importance bar charts ─────────────────────────────────────────
    st.subheader("📊 Feature Importance")
    fi_cols = st.columns(len(trained))
    for col, (name, model) in zip(fi_cols, trained.items()):
        with col:
            fig_fi = plot_feature_importance(model, list(X_test.columns), name)
            if fig_fi:
                st.pyplot(fig_fi, use_container_width=True)
            else:
                st.info(f"Not available for {name}")

    # ── Confusion Matrices (classification only) ──────────────────────────────
    if problem_type == "classification":
        st.subheader("🔢 Confusion Matrices")
        cm_cols = st.columns(len(trained))
        for col, (name, model) in zip(cm_cols, trained.items()):
            with col:
                fig_cm = plot_confusion_matrix(model, X_test, y_test, name)
                st.pyplot(fig_cm, use_container_width=True)

    # ── Actual vs Predicted (regression only) ────────────────────────────────
    if problem_type == "regression":
        st.subheader("🎯 Actual vs Predicted")
        avp_cols = st.columns(len(trained))
        for col, (name, model) in zip(avp_cols, trained.items()):
            with col:
                fig_avp = plot_actual_vs_predicted(model, X_test, y_test, name)
                st.pyplot(fig_avp, use_container_width=True)

    # ── SHAP (best non-NN model) ──────────────────────────────────────────────
    st.subheader("🔍 SHAP Feature Importance (Best Model)")

    metric    = "accuracy" if problem_type == "classification" else "r2"
    best_name = metrics_df[metric].idxmax()
    best_model = trained[best_name]

    # TreeExplainer does not support Neural Networks — fall back to best tree model
    if "Neural Network" in best_name:
        non_nn = {k: v for k, v in trained.items() if "Neural Network" not in k}
        if non_nn:
            best_name  = metrics_df.loc[list(non_nn.keys()), metric].idxmax()
            best_model = trained[best_name]
            st.info(f"SHAP uses TreeExplainer — showing results for **{best_name}**.")
        else:
            st.warning("SHAP is not available for Neural Network models.")
            best_model = None

    if best_model is not None:
        sample    = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_vals = shap_explain(best_model, sample)

        if hasattr(shap_vals, "values") and shap_vals.values.ndim == 3:
            n_classes = shap_vals.values.shape[2]
            class_idx = st.slider("Severity class", 1, n_classes, 1) - 1
            shap_plot = shap_vals[..., class_idx]
            st.caption(f"SHAP beeswarm — {best_name}  |  Severity {class_idx + 1}")
        else:
            shap_plot = shap_vals
            st.caption(f"SHAP beeswarm — {best_name}")

        fig_shap, _ = plt.subplots()
        shap.plots.beeswarm(shap_plot, show=False)
        st.pyplot(fig_shap, use_container_width=True)

    # ── Save models ───────────────────────────────────────────────────────────
    if st.checkbox("💾 Save trained models to disk"):
        for name, model in trained.items():
            path = save_model(model, name)
            st.write(f"Saved: `{path}`")

    # ── Drift detection ───────────────────────────────────────────────────────
    if run_drift:
        st.subheader("📡 Drift Detection")
        drift_feature = st.selectbox("Feature to check for drift", features)
        drift_result  = detect_drift(X_train[drift_feature], X_test[drift_feature])
        colour = "red" if drift_result["drift_detected"] else "green"
        st.markdown(
            f"KS statistic: **{drift_result['statistic']}** | "
            f"p-value: **{drift_result['p_value']}** | "
            f"Drift: :{colour}[{'YES' if drift_result['drift_detected'] else 'NO'}]"
        )