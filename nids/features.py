"""Feature engineering shared by training and prediction."""

import numpy as np
import pandas as pd

from .data import PROTOCOLS, RAW_COLUMNS

BASE_FEATURES = [
    "duration", "dst_port", "src_bytes", "dst_bytes", "packets", "syn_ratio",
    "conn_count", "unique_ports", "failed_auth_ratio",
]
DERIVED_FEATURES = [
    "total_bytes", "bytes_per_sec", "packets_per_sec", "avg_packet_size", "bytes_ratio",
]
PROTOCOL_FEATURES = ["proto_" + p for p in PROTOCOLS]
FEATURE_NAMES = BASE_FEATURES + DERIVED_FEATURES + PROTOCOL_FEATURES


def build_features(df):
    """Validate raw flow columns and return the numeric feature matrix."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: %s" % ", ".join(missing))
    raw = df[RAW_COLUMNS]
    if raw.isna().any().any():
        raise ValueError("Input contains missing values in the required columns.")

    out = pd.DataFrame(index=df.index)
    for col in BASE_FEATURES:
        out[col] = pd.to_numeric(raw[col], errors="raise").astype(float)

    total = out["src_bytes"] + out["dst_bytes"]
    duration = out["duration"].clip(lower=1e-3)
    out["total_bytes"] = total
    out["bytes_per_sec"] = total / duration
    out["packets_per_sec"] = out["packets"] / duration
    out["avg_packet_size"] = total / out["packets"].clip(lower=1)
    out["bytes_ratio"] = out["src_bytes"] / (out["dst_bytes"] + 1.0)

    protocol = raw["protocol"].astype(str).str.strip().str.lower()
    for name in PROTOCOLS:
        out["proto_" + name] = (protocol == name).astype(float)
    return out[FEATURE_NAMES].replace([np.inf, -np.inf], 0.0)
