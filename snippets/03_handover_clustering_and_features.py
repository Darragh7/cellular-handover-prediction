"""
EXCERPT - Cellular Handover Prediction (UCC Final Year Project, Darragh O'Keeffe)

Labelling handovers, clustering where they happen along the line, and turning
that history into model features.
Adapted and lightly cleaned up from the private clustering and modelling notebooks;
it is not the full pipeline and is not meant to run standalone. Full source on request.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

R_EARTH_M = 6_371_000


def label_handovers(df: pd.DataFrame) -> pd.DataFrame:
    """A handover = the serving cell ID changes. Tracked per modem (per operator)."""
    df["prev_cell"] = df.groupby("modem_id")["cell_id"].shift(1)
    df["handover"] = df["cell_id"] != df["prev_cell"]
    return df


def cluster_handover_locations(df: pd.DataFrame, radius_m: float = 30) -> pd.DataFrame:
    """Group handover events from all trips into spatial hotspots.

    DBSCAN with the haversine metric works on lat/lon in radians, so the
    neighbourhood radius is expressed as an angle (metres / Earth radius).
    The resulting clusters were rendered in Google Earth and showed that handovers
    repeatedly occur in the same places, which motivated the spatial features below.
    """
    ho = df.loc[df["handover"], ["latitude_gpx", "longitude_gpx"]].copy()
    db = DBSCAN(eps=radius_m / R_EARTH_M, min_samples=1,
                algorithm="ball_tree", metric="haversine")
    ho["cluster"] = db.fit_predict(np.radians(ho.values))
    return ho.groupby("cluster").agg(count=("latitude_gpx", "size"))


def add_hist_handover_density(df: pd.DataFrame, bin_m: int = 200) -> pd.DataFrame:
    """Historical handover rate for every 200 m stretch of track.

    Uses distance along the route rather than raw coordinates, so the feature is
    comparable across journeys and both directions of travel.
    """
    df["distance_bin"] = (df["cum_dist_m_per_trip"] // bin_m) * bin_m
    density = (df.groupby("distance_bin")["handover"].mean()
                 .rename("hist_handover_density").reset_index())
    return df.merge(density, on="distance_bin", how="left").drop(columns="distance_bin")


def add_snr_variance(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Short rolling SNR variance: competing cells near a boundary make signal unstable.

    Grouped by modem and trip so the window never spans two journeys or operators.
    """
    df = df.sort_values(["modem_id", "trip_id", "timestamp"])
    df["snr_variance_window"] = (df.groupby(["modem_id", "trip_id"])["snr"]
                                   .transform(lambda s: s.rolling(window, min_periods=3).var()))
    return df
