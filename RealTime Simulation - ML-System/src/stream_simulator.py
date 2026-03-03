import argparse
import time
import pandas as pd
from utils.config import Config
from utils.logger import get_logger


def simulate_stream(batch_size: int, delay: float):

    logger = get_logger("stream_simulator")

    df = pd.read_csv("data/raw/creditcard.csv")
    df = df.drop("Class", axis=1)

    logger.info("Starting stream simulation...")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch.to_csv("data/processed/current_batch.csv", index=False)

        logger.info(f"Streamed batch {i // batch_size}")
        time.sleep(delay)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulate streaming")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.STREAM_BATCH_SIZE
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=Config.STREAM_DELAY
    )

    args = parser.parse_args()

    simulate_stream(args.batch_size, args.delay)