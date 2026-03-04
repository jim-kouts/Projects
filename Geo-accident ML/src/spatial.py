################################################################################
# FILE: src/spatial.py
################################################################################
from sklearn.cluster import DBSCAN
# from utils.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES
# from utils.logger import get_logger
# logger = get_logger(__name__)

DBSCAN_EPS         = 0.05
DBSCAN_MIN_SAMPLES = 50


def add_spatial_clusters(df, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES):
    """
    Use DBSCAN to assign a geo_cluster label to every row in *df*.

    DBSCAN is ideal here because:
      - It discovers clusters of arbitrary shape (roads, highways).
      - Points too far from any cluster are labelled -1 (noise/outlier).
      - No need to specify the number of clusters in advance.

    Parameters
    ----------
    df          : DataFrame with 'Start_Lat' and 'Start_Lng' columns
    eps         : max neighbourhood radius (decimal degrees, ~5.5 km)
    min_samples : minimum points required to form a cluster core

    Returns
    -------
    df with a new 'geo_cluster' column (-1 = noise, 0+ = cluster id)
    """
    df = df.copy()
    coords = df[["Start_Lat", "Start_Lng"]].values

    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(coords)
    df["geo_cluster"] = clustering.labels_

    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise    = (clustering.labels_ == -1).sum()
    print(f"  Spatial clustering: {n_clusters} clusters, {n_noise:,} noise points")

    return df