import os
import time
import argparse
import pandas as pd
from scipy.stats import ks_2samp
from utils.config import Config
from utils.logger import get_logger

EXCLUDE_COLUMNS = {"Time"}


def detect_drift(train_data: pd.DataFrame, recent_data: pd.DataFrame, window_size: int, logger):

    feature_cols = [c for c in train_data.columns if c not in EXCLUDE_COLUMNS]

    drift_detected = False

    for column in feature_cols:

        if column not in recent_data.columns:
            continue

        stat, p_value = ks_2samp(
            train_data[column].values[:window_size],
            recent_data[column].values
        )

        if p_value < Config.DRIFT_SIGNIFICANCE_LEVEL:
            logger.warning(f"Drift detected in feature '{column}' (p={p_value:.4f}, stat={stat:.4f})")
            drift_detected = True

    if not drift_detected:
        logger.info("No drift detected in this batch.")


def main(args):

    logger = get_logger("drift_detection")

    train_path = "data/processed/X_train.csv"
    stream_path = os.path.join(Config.DATA_PROCESSED_DIR, "current_batch.csv")

    logger.info("Loading training reference data...")
    train_data = pd.read_csv(train_path)

    processed_batches = set()

    logger.info("Starting continuous drift detection loop...")

    while True:
        try:
            if os.path.exists(stream_path):

                batch_id = os.path.getmtime(stream_path)

                if batch_id not in processed_batches:

                    logger.info("New batch detected — running drift detection...")

                    recent_data = pd.read_csv(stream_path)

                    detect_drift(train_data, recent_data, args.window_size, logger)

                    processed_batches.add(batch_id)

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            logger.info("Stopping drift detection service.")
            break

        except Exception as e:
            logger.error(f"Error during drift detection: {e}")
            time.sleep(2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Drift Detection")

    parser.add_argument(
        "--window_size",
        type=int,
        default=Config.DRIFT_WINDOW_SIZE,
        help="Number of samples from training data to use as reference"
    )

    parser.add_argument(
        "--poll_interval",
        type=float,
        default=1.0,
        help="Seconds between checks for a new batch"
    )

    args = parser.parse_args()

    main(args)