import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from utils.config import Config
from utils.logger import get_logger


def train_model(model_type: str):

    logger = get_logger("train_model")

    logger.info(f"Training model: {model_type}")

    X_train = pd.read_csv(os.path.join(Config.DATA_PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(Config.DATA_PROCESSED_DIR, "y_train.csv"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=Config.RANDOM_SEED
        )
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100,
            random_state=Config.RANDOM_SEED,
            eval_metric="logloss"
        )
    else:
        raise ValueError("Model type must be 'random_forest' or 'xgboost'")

    model.fit(X_train_scaled, y_train.values.ravel())

    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(Config.MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(Config.MODEL_DIR, "scaler.pkl"))

    logger.info("Model training completed and saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ML model")

    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        help="Choose model: random_forest or xgboost"
    )

    args = parser.parse_args()

    train_model(args.model_type)