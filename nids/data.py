"""Synthetic network-flow generator.

Each row describes one network flow together with a few aggregate values for the
same source address over a short time window (connection count, distinct ports,
failed authentication ratio). The distributions are hand-designed to mimic
typical behaviour and are NOT real traffic. See the README for how to use a real
dataset instead.
"""

import numpy as np
import pandas as pd

CLASSES = ["normal", "port_scan", "syn_flood", "brute_force", "exfiltration"]
PROTOCOLS = ["tcp", "udp", "icmp"]
RAW_COLUMNS = [
    "duration", "protocol", "dst_port", "src_bytes", "dst_bytes", "packets",
    "syn_ratio", "conn_count", "unique_ports", "failed_auth_ratio",
]
CLASS_WEIGHTS = [0.70, 0.08, 0.08, 0.07, 0.07]
OVERLAP_PROBABILITY = 0.05  # share of rows that look like another class

_TCP_PORTS = [80, 443, 53, 22, 25, 8080, 3306, 445, 993, 21]
_TCP_PORT_P = [0.30, 0.34, 0.04, 0.08, 0.05, 0.05, 0.03, 0.03, 0.03, 0.05]


def _normal(rng):
    protocol = rng.choice(PROTOCOLS, p=[0.70, 0.25, 0.05])
    if protocol == "tcp":
        port = int(rng.choice(_TCP_PORTS, p=_TCP_PORT_P))
    elif protocol == "udp":
        port = int(rng.choice([53, 123, 161, 443]))
    else:
        port = 0
    src = int(rng.lognormal(6.5, 1.3))
    dst = int(rng.lognormal(8.0, 1.8))
    failed = float(rng.beta(1, 8)) if port in (21, 22) and rng.random() < 0.10 else 0.0
    return {
        "duration": float(rng.lognormal(-0.5, 1.2)),
        "protocol": str(protocol),
        "dst_port": port,
        "src_bytes": src,
        "dst_bytes": dst,
        "packets": max(1, int((src + dst) / rng.uniform(300, 900))),
        "syn_ratio": float(rng.beta(1, 12)) if protocol == "tcp" else 0.0,
        "conn_count": 1 + int(rng.poisson(3)),
        "unique_ports": 1 + int(rng.poisson(0.8)),
        "failed_auth_ratio": failed,
    }


def _port_scan(rng):
    return {
        "duration": float(rng.exponential(0.02)),
        "protocol": "tcp",
        "dst_port": int(rng.integers(1, 65536)),
        "src_bytes": int(rng.integers(40, 130)),
        "dst_bytes": int(rng.integers(0, 80)),
        "packets": int(rng.integers(1, 4)),
        "syn_ratio": float(rng.uniform(0.85, 1.0)),
        "conn_count": int(rng.integers(20, 300)),
        "unique_ports": int(rng.integers(15, 250)),
        "failed_auth_ratio": 0.0,
    }


def _syn_flood(rng):
    packets = int(rng.integers(1, 6))
    return {
        "duration": float(rng.exponential(0.01)),
        "protocol": "tcp",
        "dst_port": int(rng.choice([80, 443, 8080])),
        "src_bytes": packets * int(rng.integers(40, 70)),
        "dst_bytes": 0,
        "packets": packets,
        "syn_ratio": float(rng.uniform(0.9, 1.0)),
        "conn_count": int(rng.integers(100, 2000)),
        "unique_ports": 1 + int(rng.poisson(0.5)),
        "failed_auth_ratio": 0.0,
    }


def _brute_force(rng):
    return {
        "duration": float(rng.uniform(0.3, 2.5)),
        "protocol": "tcp",
        "dst_port": int(rng.choice([22, 21, 3389, 25])),
        "src_bytes": int(rng.integers(300, 1500)),
        "dst_bytes": int(rng.integers(300, 2500)),
        "packets": int(rng.integers(8, 25)),
        "syn_ratio": float(rng.beta(1, 10)),
        "conn_count": int(rng.integers(5, 60)),
        "unique_ports": 1,
        "failed_auth_ratio": float(rng.uniform(0.6, 1.0)),
    }


def _exfiltration(rng):
    src = int(rng.uniform(5e6, 4e8))
    return {
        "duration": float(rng.uniform(30, 900)),
        "protocol": "tcp",
        "dst_port": int(rng.choice([443, 21, 53, 8080])),
        "src_bytes": src,
        "dst_bytes": int(rng.integers(1_000, 60_000)),
        "packets": max(1, int(src / rng.uniform(900, 1400))),
        "syn_ratio": float(rng.beta(1, 80)),
        "conn_count": 1 + int(rng.poisson(1)),
        "unique_ports": 1,
        "failed_auth_ratio": 0.0,
    }


_GENERATORS = {
    "normal": _normal,
    "port_scan": _port_scan,
    "syn_flood": _syn_flood,
    "brute_force": _brute_force,
    "exfiltration": _exfiltration,
}


def generate(n_samples=20000, seed=42, labeled=True):
    """Generate a DataFrame of synthetic flows (with a 'label' column if labeled)."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    rng = np.random.default_rng(seed)
    labels = rng.choice(CLASSES, size=n_samples, p=CLASS_WEIGHTS)
    rows = []
    for label in labels:
        source = str(label)
        if rng.random() < OVERLAP_PROBABILITY:  # ambiguous flow: borrow another class's behaviour
            source = str(rng.choice([c for c in CLASSES if c != label]))
        row = _GENERATORS[source](rng)
        if labeled:
            row["label"] = str(label)
        rows.append(row)
    columns = RAW_COLUMNS + (["label"] if labeled else [])
    return pd.DataFrame(rows, columns=columns)
