import time
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from utils.config import MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# XGBoost wrapper (handles 0-based label shift)
# ─────────────────────────────────────────────

class XGBoostClassifierWrapper:
    """
    Thin wrapper around XGBClassifier that handles the 0-based label
    requirement internally, so the rest of the pipeline never sees it.
    """
    def __init__(self, **kwargs):
        self.model        = XGBClassifier(**kwargs)
        self.label_offset = 0

    def fit(self, X, y):
        # Shift labels to start from 0 if needed (e.g. 1,2,3,4 -> 0,1,2,3)
        self.label_offset = int(y.min()) if y.min() > 0 else 0
        self.model.fit(X, y - self.label_offset)
        return self

    def predict(self, X):
        # Shift predictions back to original range (e.g. 0,1,2,3 -> 1,2,3,4)
        return self.model.predict(X) + self.label_offset

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __getattr__(self, name):
        return getattr(self.model, name)


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────

def get_models(problem_type: str) -> dict:
    """
    Return a dict of {name: unfitted model} for the given problem type.
    Includes Random Forest, XGBoost, and Neural Network (MLP) for both types.
    """
    logger.info(f"Building model registry for: {problem_type}")

    if problem_type == "classification":
        return {
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42,
            ),
            "XGBoost Classifier": XGBoostClassifierWrapper(
                n_estimators=100,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            ),
            "Neural Network Classifier": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
                activation="relu",                  # ReLU activation
                solver="adam",                      # Adam optimizer
                max_iter=200,                       # max training epochs
                early_stopping=True,                # stop if val loss stops improving
                validation_fraction=0.1,            # 10% of train used for validation
                random_state=42,
                verbose=False,
            ),
        }

    if problem_type == "regression":
        return {
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100,
                n_jobs=-1,
                random_state=42,
            ),
            "XGBoost Regressor": XGBRegressor(
                n_estimators=100,
                verbosity=0,
                random_state=42,
            ),
            "Linear Regression": LinearRegression(
                n_jobs=-1,
            ),
            "Neural Network Regressor": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
                activation="relu",
                solver="adam",
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=False,
            ),
        }

    raise ValueError(
        f"Unknown problem_type '{problem_type}'. "
        "Expected 'classification' or 'regression'."
    )


# ─────────────────────────────────────────────
# Training loop with progress logging
# ─────────────────────────────────────────────

def train_models(models: dict, X_train, y_train) -> dict:
    """
    Fit every model in *models* and return a dict of trained models.
    Prints ASCII progress bar + elapsed time per model.
    """
    trained = {}
    total   = len(models)

    logger.info(f"Starting training loop: {total} model(s)")
    print(f"\n{'='*55}")
    print(f"  Training {total} model(s) on {len(X_train):,} samples")
    print(f"{'='*55}")

    for idx, (name, model) in enumerate(models.items(), start=1):
        pct_start = int((idx - 1) / total * 100)
        print(f"\n[{_bar(pct_start)}] {pct_start:3d}%  Starting -> {name}")
        logger.info(f"[{idx}/{total}] Training '{name}' ...")

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        trained[name] = model
        pct_done = int(idx / total * 100)
        print(f"[{_bar(pct_done)}] {pct_done:3d}%  Done    -> {name}  ({elapsed:.1f}s)")
        logger.info(f"  '{name}' trained in {elapsed:.2f}s")

    print(f"\n{'='*55}")
    print(f"  All {total} model(s) trained successfully!")
    print(f"{'='*55}\n")
    logger.info("All models trained.")
    return trained


# ─────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────

def save_model(model, name: str) -> str:
    """Persist a trained model to MODEL_DIR using joblib."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name.replace(" ", "_").lower() + ".pkl")
    joblib.dump(model, path)
    logger.info(f"Model saved -> {path}")
    return path


def load_model(name: str):
    """Load a previously saved model from MODEL_DIR."""
    path = os.path.join(MODEL_DIR, name.replace(" ", "_").lower() + ".pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at '{path}'")
    logger.info(f"Loading model from: {path}")
    return joblib.load(path)


def _bar(pct: int, width: int = 30) -> str:
    """Return a simple ASCII progress bar string for a given percentage."""
    filled = int(width * pct / 100)
    return "#" * filled + "-" * (width - filled)