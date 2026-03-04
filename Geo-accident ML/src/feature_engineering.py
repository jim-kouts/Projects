################################################################################
# FILE: src/feature_engineering.py
################################################################################
import pandas as pd
# from utils.config import (RUSH_HOUR_MORNING_START, RUSH_HOUR_MORNING_END,
#                            RUSH_HOUR_EVENING_START, RUSH_HOUR_EVENING_END)
# from utils.logger import get_logger
# logger = get_logger(__name__)

RUSH_HOUR_MORNING_START, RUSH_HOUR_MORNING_END   = 7, 9
RUSH_HOUR_EVENING_START, RUSH_HOUR_EVENING_END   = 16, 18


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'Start_Time' and extract cyclic / calendar features:
      - hour          (0-23)
      - day_of_week   (0=Monday … 6=Sunday)
      - month         (1-12)
      - is_weekend    (1 if Saturday or Sunday)
      - hour_sin / hour_cos  — cyclic encoding so 23:00 and 00:00 are close
    """
    df = df.copy()                                    # never mutate the original
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

    df["hour"]        = df["Start_Time"].dt.hour
    df["day_of_week"] = df["Start_Time"].dt.dayofweek
    df["month"]       = df["Start_Time"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Cyclic encoding of hour: avoids the artificial jump from hour 23 to 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def add_rush_hour_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary 'rush_hour' column: 1 during morning (7-9) or evening (16-18),
    0 otherwise. Uses config constants so thresholds are easy to change.
    """
    df = df.copy()
    df["rush_hour"] = df["hour"].apply(
        lambda h: 1 if (
            RUSH_HOUR_MORNING_START <= h <= RUSH_HOUR_MORNING_END or
            RUSH_HOUR_EVENING_START <= h <= RUSH_HOUR_EVENING_END
        ) else 0
    )
    return df


import numpy as np   # needed for cyclic encoding above