import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config import Config
from utils.logger import get_logger


def preprocess_and_split(input_path: str, test_size: float):

    logger = get_logger("data_loader")

    logger.info("Loading dataset...")
    df = pd.read_csv(input_path)
    logger.info(f"Dataset shape: {df.shape}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    logger.info("Performing stratified train-test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=Config.RANDOM_SEED,
        stratify=y
    )

    os.makedirs(Config.DATA_PROCESSED_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(Config.DATA_PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(Config.DATA_PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(Config.DATA_PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(Config.DATA_PROCESSED_DIR, "y_test.csv"), index=False)

    logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess dataset")

    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(Config.DATA_RAW_DIR, "creditcard.csv"),
        help="Path to raw dataset"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=Config.TEST_SIZE,
        help="Test split ratio"
    )

    args = parser.parse_args()

    preprocess_and_split(args.input_path, args.test_size)