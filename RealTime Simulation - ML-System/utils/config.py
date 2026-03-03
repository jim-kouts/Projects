import os

class Config:
    """
    Central configuration class for the Real-Time ML System.
    All global settings should be defined here.
    """

    # -----------------------------
    # Project Paths
    # -----------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # -----------------------------
    # Dataset Parameters
    # -----------------------------
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    # -----------------------------
    # Streaming Parameters
    # -----------------------------
    STREAM_DELAY = 0.5  # seconds between transactions
    STREAM_BATCH_SIZE = 1

    # -----------------------------
    # Drift Detection Parameters
    # -----------------------------
    DRIFT_WINDOW_SIZE = 500
    DRIFT_SIGNIFICANCE_LEVEL = 0.05  # p-value threshold