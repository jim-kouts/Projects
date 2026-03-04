################################################################################
# FILE: src/drift.py
################################################################################
from scipy.stats import ks_2samp
# from utils.config import DRIFT_PVALUE_THRESHOLD
# from utils.logger import get_logger
# logger = get_logger(__name__)

DRIFT_PVALUE_THRESHOLD = 0.05


def detect_drift(train_feature, new_feature, threshold: float = DRIFT_PVALUE_THRESHOLD) -> dict:
    """
    Run the two-sample Kolmogorov-Smirnov test to detect distribution shift.

    The KS statistic measures the maximum difference between the two CDFs.
    A small p-value (< threshold) means the distributions are unlikely to
    be the same, indicating data drift.

    Parameters
    ----------
    train_feature : array-like — feature values from training data
    new_feature   : array-like — feature values from new/incoming data
    threshold     : p-value cutoff for flagging drift (default 0.05)

    Returns
    -------
    dict with keys: statistic, p_value, drift_detected
    """
    stat, p_value = ks_2samp(train_feature, new_feature)
    drift_detected = p_value < threshold

    if drift_detected:
        print(f"  [DRIFT ALERT] p={p_value:.4f} < {threshold}  => distribution shift detected!")
    else:
        print(f"  [OK] p={p_value:.4f} >= {threshold}  => no significant drift")

    return {
        "statistic":     round(float(stat), 6),
        "p_value":       round(float(p_value), 6),
        "drift_detected": drift_detected,
    }