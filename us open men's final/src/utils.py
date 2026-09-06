import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt


GLOBAL_RANDOM_SEED = 42


def seed_everything(seed: int = GLOBAL_RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    for p in [
        Path("data/raw"),
        Path("data/interim"),
        Path("data/processed"),
        Path("models"),
        Path("config"),
        Path("notebooks"),
    ]:
        p.mkdir(parents=True, exist_ok=True)


def default_start_end_from_years() -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    start = dt.date(today.year - 1, 1, 1)
    return start, today


def parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "tourney_date" in df.columns:
        # Jeff Sackmann format is YYYYMMDD integer
        df["date"] = pd.to_datetime(df["tourney_date"].astype(str), errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Missing date column (tourney_date or date)")
    return df


def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "winner_name": "winner_name",
        "loser_name": "loser_name",
        "surface": "surface",
        "tourney_name": "tourney_name",
        "round": "round",
        "best_of": "best_of",
    }
    df = df.rename(columns=rename)
    # Standardize name casing/spaces
    for c in ["winner_name", "loser_name", "tourney_name", "surface", "round"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def filter_date_range(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    return df.loc[mask].copy()


def to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


