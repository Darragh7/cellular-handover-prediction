"""
EXCERPT - Cellular Handover Prediction (UCC Final Year Project, Darragh O'Keeffe)

Splitting a continuous day of logging into individual train journeys.
Adapted and lightly cleaned up from the private data-merging notebook; it is not
the full pipeline and is not meant to run standalone. Full source on request.

Early versions split trips on speed and dwell-time thresholds, which turned out
to be fragile with noisy, sparsely sampled GPS. The final approach uses
proximity to the two terminal stations, which are fixed and well known.
"""
import numpy as np
import pandas as pd

R_EARTH_M = 6_371_000
STATION_RADIUS_M = 100    # "at the station" = within 100 m (absorbs GPS drift)
MIN_DWELL_S = 5           # must stay inside the radius this long to count as arrival
MIN_TRIP_M = 13_400       # route is ~14 km; shorter segments are not real journeys
TRIP_MINUTES = (25, 42)   # plausible journey duration


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (vectorised over pandas/NumPy inputs)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH_M * np.arcsin(np.sqrt(a))


def segment_trips(df: pd.DataFrame, stations: dict) -> pd.DataFrame:
    """`stations` maps a terminal name to its public (lat, lon) from OpenStreetMap."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    near = np.zeros(len(df), dtype=bool)
    for lat, lon in stations.values():
        near |= haversine_m(df["latitude_gpx"], df["longitude_gpx"], lat, lon) < STATION_RADIUS_M

    # Accumulate dwell time inside each consecutive "near a station" run.
    run_id = (pd.Series(near) != pd.Series(near).shift()).cumsum()
    dt = df["timestamp"].diff().dt.total_seconds().fillna(0)
    dwell = dt.groupby(run_id).cumsum().where(near, 0)

    # A new trip starts the first moment the dwell threshold is crossed.
    arrival = near & (dwell >= MIN_DWELL_S) & (dwell.shift().fillna(0) < MIN_DWELL_S)
    arrival.iloc[0] = True
    df["trip_id"] = arrival.cumsum().map("trip_{:03d}".format)

    # Distance restarts at zero for every journey, so trips line up on one axis.
    df["cum_dist_m_per_trip"] = df.groupby("trip_id")["cum_dist_m"].transform(lambda d: d - d.min())

    # Drop "non-trips": walking between platforms, waiting with the modems on, etc.
    stats = df.groupby("trip_id").agg(
        length=("cum_dist_m_per_trip", "max"),
        minutes=("timestamp", lambda t: (t.max() - t.min()).total_seconds() / 60),
    )
    valid = stats[(stats.length >= MIN_TRIP_M) & stats.minutes.between(*TRIP_MINUTES)].index
    return df[df["trip_id"].isin(valid)]
