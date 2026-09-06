# src/features.py
from __future__ import annotations

import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import ensure_dirs, to_parquet

US_SWING_TOURNEYS = {
    "Washington", "Cincinnati Masters", "Canada Masters",
    "Toronto Masters", "Montreal Masters", "U.S. Open", "US Open", "Cincinnati"
}

# ---------- helpers to make long (player-appearance) table ----------

def _make_long(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one for winner (win=1), one for loser (win=0)."""
    w = df.copy()
    w["player"] = w["winner_name"]
    w["opp"] = w["loser_name"]
    w["win"] = 1

    l = df.copy()
    l["player"] = l["loser_name"]
    l["opp"] = l["winner_name"]
    l["win"] = 0

    long = pd.concat([w, l], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["player", "date"], kind="mergesort").reset_index(drop=True)
    return long


def _add_player_features(long: pd.DataFrame) -> pd.DataFrame:
    """Add past-only rolling features per player and date."""
    long = long.copy()
    g = long.groupby("player", group_keys=False)

    # Recent form: last N matches (simple mean). shift(1) prevents peeking.
    def _recent(s, n, minp=3):
        return s.shift(1).rolling(window=n, min_periods=minp).mean()

    long["recent5"]  = g["win"].apply(lambda s: _recent(s, 5))
    # You can add recent10/20 if you want later.

    # Hard-court win rate over last 12 months (time-based rolling)
    is_hard = long["surface"].str.contains("Hard", case=False, na=False)
    long["_win_hard"] = np.where(is_hard, long["win"], np.nan)

    def _hard12(df):
        df = df.sort_values("date").copy()
        df = df.set_index("date")
        # 365D window; shift(1) to use only prior matches
        out = df["_win_hard"].rolling("365D").mean().shift(1)
        return out.reset_index(drop=True)

    long["hard_12m"] = g.apply(_hard12).reset_index(level=0, drop=True)

    # US swing same-season rate up to (but not including) this match
    long["year"] = long["date"].dt.year
    is_us = long["tourney_name"].isin(US_SWING_TOURNEYS)
    long["_us_win"] = np.where(is_us, long["win"], np.nan)

    def _us_rate(df):
        df = df.sort_values("date").copy()
        # expanding mean over US-swing rows only, ignoring NaN; shift to exclude current
        df["us_swing"] = df["_us_win"].expanding(min_periods=1).mean().shift(1)
        return df["us_swing"]

    long["us_swing"] = (
        long
        .sort_values(["player", "year", "date"])
        .groupby(["player", "year"], group_keys=False)
        .apply(_us_rate)
        .reset_index(drop=True)
    )

    # Match load last 14 / 30 days (time-based counts, past-only)
    def _mcount(df):
        df = df.sort_values("date").copy()
        df["one"] = 1
        df = df.set_index("date")
        m14 = df["one"].rolling("14D").sum().shift(1).fillna(0)
        m30 = df["one"].rolling("30D").sum().shift(1).fillna(0)
        out = pd.DataFrame({"m14": m14.astype(int), "m30": m30.astype(int)}).reset_index(drop=True)
        return out

    mloads = g.apply(_mcount)
    # After groupby-apply with DataFrame return, index is aligned; reattach:
    long["m14"] = mloads["m14"].values
    long["m30"] = mloads["m30"].values

    # Clean up temps
    long = long.drop(columns=["_win_hard", "_us_win"], errors="ignore")
    return long


# ---------- pair builder & final feature diffs ----------

def _make_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Create symmetric rows (A vs B with y=1; B vs A with y=0)."""
    a = df.copy()
    a["A"] = a["winner_name"]
    a["B"] = a["loser_name"]
    a["y"] = 1

    b = df.copy()
    b["A"] = b["loser_name"]
    b["B"] = b["winner_name"]
    b["y"] = 0

    out = pd.concat([a, b], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    # Keep only the columns we need from the base table for merging keys
    return out[["date", "tourney_name", "surface", "A", "B", "y"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort chronologically to keep all rolling logic consistent
    df = df.sort_values("date").reset_index(drop=True)

    # Long player table with past-only features
    long = _make_long(df)
    long = _add_player_features(long)

    # Pair rows
    pair = _make_pair_rows(df)

    # Merge player features for A and B at the same match date
    keep_cols = ["player", "date", "recent5", "hard_12m", "us_swing", "m14", "m30"]
    fa = long[keep_cols].rename(columns={c: f"A_{c}" for c in keep_cols if c != "date"})
    fb = long[keep_cols].rename(columns={c: f"B_{c}" for c in keep_cols if c != "date"})

    pair = (
        pair.merge(
            fa, left_on=["A", "date"], right_on=["A_player", "date"], how="left"
        )
        .drop(columns=["A_player"])
        .merge(
            fb, left_on=["B", "date"], right_on=["B_player", "date"], how="left"
        )
        .drop(columns=["B_player"])
    )

    # Differences: A - B
    for col in ["recent5", "hard_12m", "us_swing", "m14", "m30"]:
        pair[f"diff_{col}"] = pair[f"A_{col}"] - pair[f"B_{col}"]

    # Optional: if any diffs are still NaN (no history), treat as parity
    diff_cols = [c for c in pair.columns if c.startswith("diff_")]
    pair[diff_cols] = pair[diff_cols].fillna(0.0)

    # Final ordered columns
    keep = diff_cols + ["A", "B", "y", "date"]
    return pair[keep]


def build_and_save_features(start: dt.date, end: dt.date) -> None:
    ensure_dirs()
    df = pd.read_parquet(Path("data/interim/matches.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()

    feats = build_features(df)
    X = feats.drop(columns=["y"])
    y = feats["y"].astype(int)

    to_parquet(X, Path("data/processed/X_train.parquet"))
    to_parquet(pd.DataFrame({"y": y}), Path("data/processed/y_train.parquet"))
