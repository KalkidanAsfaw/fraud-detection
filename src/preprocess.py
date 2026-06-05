import socket
import struct

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def ip_to_int(ip_str: str) -> float:
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip_str)))[0]
    except Exception:
        return np.nan


def map_ip_to_country(df: pd.DataFrame, ip_ranges: pd.DataFrame) -> pd.DataFrame:
    """Merge fraud DataFrame with IP range lookup table to add a 'country' column."""
    df = df.copy()
    ip_ranges = ip_ranges.copy()

    df["ip_int"] = df["ip_address"].apply(ip_to_int)
    ip_ranges["lower_int"] = ip_ranges["lower_bound_ip_address"].apply(ip_to_int)
    ip_ranges["upper_int"] = ip_ranges["upper_bound_ip_address"].apply(ip_to_int)

    ip_ranges_sorted = ip_ranges.dropna().sort_values("lower_int").reset_index(drop=True)
    df_sorted = df.dropna(subset=["ip_int"]).sort_values("ip_int").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sorted,
        ip_ranges_sorted[["lower_int", "upper_int", "country"]],
        left_on="ip_int",
        right_on="lower_int",
        direction="backward",
    )
    merged = merged[merged["ip_int"] <= merged["upper_int"]].copy()
    merged["country"] = merged["country"].fillna("Unknown")
    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    user_tx_count = (
        df.groupby("user_id")["purchase_time"].count().rename("user_tx_count")
    )
    df = df.merge(user_tx_count, on="user_id", how="left")
    return df


def encode_and_scale(df: pd.DataFrame, num_cols: list, cat_cols: list, scaler=None):
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler
