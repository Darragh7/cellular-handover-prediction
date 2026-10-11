"""
EXCERPT - Cellular Handover Prediction (UCC Final Year Project, Darragh O'Keeffe)

Aligning LTE modem logs with an independently recorded GPS track.
Adapted and lightly cleaned up from the private data-merging notebook; it is not
the full pipeline and is not meant to run standalone. Full source on request.

Two logging systems ran in parallel on the train:
  * LTE modems  -> CSV, ~1 sample/s (RSRP, RSRQ, SNR, cell ID, TAC, ...)
  * Phone GPS   -> GPX track (timestamped lat/lon, speed, satellites)
Each modem sample is paired with the nearest GPS fix within 1 second.
"""
import pandas as pd
from geopy.distance import geodesic


def add_track_distance(gpx_df: pd.DataFrame) -> pd.DataFrame:
    """Geodesic distance between consecutive GPS fixes plus a running total.

    Cumulative distance becomes the shared spatial axis that lets repeated
    journeys be compared with each other, regardless of train speed or dwell time.
    """
    pts = list(zip(gpx_df["latitude_gpx"], gpx_df["longitude_gpx"]))
    deltas = [0.0] + [geodesic(a, b).meters for a, b in zip(pts[:-1], pts[1:])]
    gpx_df["delta_dist_m"] = deltas
    gpx_df["cum_dist_m"] = gpx_df["delta_dist_m"].cumsum()
    return gpx_df


def merge_modem_with_gps(modem_df: pd.DataFrame, gpx_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest GPS fix (within 1 s) to every modem sample."""
    # Corrupted rows from logging interruptions: drop empties and bad timestamps.
    modem_df = modem_df.dropna(how="all")
    modem_df["timestamp"] = pd.to_datetime(modem_df["timestamp"], utc=True, errors="coerce")
    modem_df = modem_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Only keep samples recorded while the GPS logger was running: anything
    # outside that window has no spatial context and can't be used for training.
    start, end = gpx_df["timestamp"].min(), gpx_df["timestamp"].max()
    modem_df = modem_df[modem_df["timestamp"].between(start, end)]

    merged = pd.merge_asof(
        modem_df,
        gpx_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",            # closest fix either side of the sample
        tolerance=pd.Timedelta("1s"),   # beyond 1 s the position is unreliable
    )

    # Samples that fell into short GPS outages are excluded later on.
    print(f"{len(merged)} rows merged, {merged['latitude_gpx'].isna().sum()} without a GPS match")
    return merged
