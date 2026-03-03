import os
import time
import argparse
import pandas as pd
import joblib
from utils.config import Config
from utils.logger import get_logger


def main(args):
    logger = get_logger("online_inference")

    model_path  = os.path.join(Config.MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(Config.MODEL_DIR, "scaler.pkl")

    stream_path = os.path.join(Config.DATA_PROCESSED_DIR, "current_batch.csv")
    output_path = os.path.join(Config.DATA_PROCESSED_DIR, "predictions.csv")

    logger.info("Loading model and scaler...")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    processed_batches = set()

    logger.info("Starting continuous inference loop...")

    while True:
        try:
            if os.path.exists(stream_path):

                batch_id = os.path.getmtime(stream_path)

                if batch_id not in processed_batches:

                    logger.info("New batch detected.")

                    df = pd.read_csv(stream_path)

                    X = scaler.transform(df)
                    predictions = model.predict(X)

                    # Build a clean output with only the columns we need
                    out = pd.DataFrame()
                    if "Time"   in df.columns: out["Time"]   = df["Time"].values
                    if "Amount" in df.columns: out["Amount"] = df["Amount"].values
                    out["prediction"] = predictions

                    # Append results — write header only on first write
                    write_header = not os.path.exists(output_path)
                    out.to_csv(output_path, mode="a", header=write_header, index=False)

                    processed_batches.add(batch_id)
                    logger.info(f"Predictions saved. Fraud in batch: {int(predictions.sum())}")

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            logger.info("Stopping inference service.")
            break

        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.py", help="Path to configuration file")
    parser.add_argument("--poll_interval", type=float, default=1.0)

    args = parser.parse_args()

    main(args)