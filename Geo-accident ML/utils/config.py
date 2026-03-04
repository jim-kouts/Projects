################################################################################
# FILE: utils/config.py
################################################################################
import os

# Base directory of the project (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH  = os.path.join(BASE_DIR, "data", "accidents.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
LOG_FILE   = os.path.join(LOG_DIR, "app.log")

# Ensure directories exist on import
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# Train / test split
TEST_SIZE    = 0.2   # 20% of data reserved for evaluation
RANDOM_STATE = 42    # Fixed seed for reproducibility

# Drift detection (KS-test): p < threshold => drift flagged
DRIFT_PVALUE_THRESHOLD = 0.05

# DBSCAN spatial clustering
DBSCAN_EPS         = 0.05   # ~5.5 km radius in decimal degrees
DBSCAN_MIN_SAMPLES = 50     # min points to form a cluster core

# Rush-hour windows (inclusive hour values, 24h clock)
RUSH_HOUR_MORNING_START = 7
RUSH_HOUR_MORNING_END   = 9
RUSH_HOUR_EVENING_START = 16
RUSH_HOUR_EVENING_END   = 18