import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.config import TEST_SIZE, RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)

# Columns that carry no predictive value and should always be dropped
COLUMNS_TO_DROP = [
    "ID", "Source", "Description", "Street",
    "Zipcode", "Country", "Timezone"
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load the accidents CSV and return a DataFrame.
    Raises FileNotFoundError with a helpful message if file is missing.
    """
    logger.info(f"Loading dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Check DATA_PATH in utils/config.py."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are identifiers or free-text fields with no
    predictive value (ID, Source, Description, Street, Zipcode, Country, Timezone).

    Only drops columns that actually exist in the DataFrame so this is
    safe to call even if the dataset is missing some of them.
    """
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
        logger.info(f"Dropped {len(cols_present)} irrelevant columns: {cols_present}")
    else:
        logger.info("No irrelevant columns found to drop.")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert non-numeric columns to numbers so every model can train on them.

    Rules applied in order:
      1. Boolean columns (True/False)  -> 1 / 0
      2. Binary string columns         -> 1 / 0  (e.g. 'Yes'/'No', 'T'/'F')
      3. All other object columns      -> integer codes via LabelEncoder
                                          (e.g. 'Fair', 'Cloudy', 'Rain' -> 0, 1, 2)

    A mapping dict is printed to the log so you can always trace back what
    number corresponds to which original label.
    """
    df = df.copy()
    encoding_map = {}   # stores label -> int mappings for reference

    for col in df.select_dtypes(include=["object", "bool"]).columns:

        # ── Rule 1: actual Python booleans ────────────────────────────────────
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            encoding_map[col] = {True: 1, False: 0}
            continue

        # Normalise to string and strip whitespace for consistent comparisons
        col_clean = df[col].astype(str).str.strip()

        unique_vals = set(col_clean.str.lower().unique()) - {"nan", "unknown", ""}

        # ── Rule 2: boolean-like strings ──────────────────────────────────────
        if unique_vals <= {"true", "false"}:
            df[col] = col_clean.str.lower().map({"true": 1, "false": 0})
            encoding_map[col] = {"true": 1, "false": 0}
            continue

        if unique_vals <= {"yes", "no"}:
            df[col] = col_clean.str.lower().map({"yes": 1, "no": 0})
            encoding_map[col] = {"yes": 1, "no": 0}
            continue

        if unique_vals <= {"t", "f"}:
            df[col] = col_clean.str.lower().map({"t": 1, "f": 0})
            encoding_map[col] = {"t": 1, "f": 0}
            continue

        # ── Rule 3: general categorical  (LabelEncoder) ───────────────────────
        le = LabelEncoder()
        # Fill NaN/Unknown with a placeholder before encoding
        col_filled = col_clean.fillna("Unknown")
        df[col] = le.fit_transform(col_filled)
        encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Log a summary so the user can see what was encoded
    if encoding_map:
        logger.info(f"Encoded {len(encoding_map)} column(s):")
        for col, mapping in encoding_map.items():
            # Only show first 8 entries to keep logs readable
            preview = dict(list(mapping.items())[:8])
            logger.info(f"  {col}: {preview}{'...' if len(mapping) > 8 else ''}")
    else:
        logger.info("No categorical columns found to encode.")

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning applied before feature engineering:
      1. Drop rows where 'Severity' is null (unusable for supervised learning).
      2. Fill remaining NaNs: numeric -> 0, categorical -> 'Unknown'.
      3. Drop fully-duplicate rows.
    """
    initial_rows = len(df)
    logger.info(f"Starting cleaning on {initial_rows:,} rows...")

    # Step 1: must have a target value
    df = df.dropna(subset=["Severity"])
    dropped_no_target = initial_rows - len(df)
    if dropped_no_target:
        logger.warning(f"  Dropped {dropped_no_target:,} rows missing 'Severity'")

    # Step 2: type-aware NaN fill
    numeric_cols = df.select_dtypes(include="number").columns
    object_cols  = df.select_dtypes(include="object").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[object_cols]  = df[object_cols].fillna("Unknown")
    logger.info("  Filled NaNs: numeric->0, categorical->'Unknown'")

    # Step 3: remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before - len(df)
    if dropped_dupes:
        logger.info(f"  Removed {dropped_dupes:,} duplicate rows")

    logger.info(f"Cleaning done. Remaining rows: {len(df):,}")
    return df.reset_index(drop=True)


def train_test_split_data(df, features, target, test_size=0.2, random_state=42):
    """
    Validate columns, then split df into X_train/X_test/y_train/y_test.
    Includes a final NaN safety-net fill so no model ever receives NaN input.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in DataFrame: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    X, y = df[features].copy(), df[target].copy()

    # Safety-net: fill any NaNs that slipped through (e.g. engineered features)
    numeric_cols = X.select_dtypes(include="number").columns
    object_cols  = X.select_dtypes(include="object").columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    X[object_cols]  = X[object_cols].fillna("Unknown")

    # Drop rows where target is still NaN
    mask = y.notna()
    X, y = X[mask], y[mask]

    stratify = y if y.nunique() <= 20 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    logger.info(f"Split complete: train={len(X_train):,}, test={len(X_test):,}")
    return X_train, X_test, y_train, y_test