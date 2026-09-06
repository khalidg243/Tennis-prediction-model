from __future__ import annotations

import io
import datetime as dt
from pathlib import Path
import pandas as pd
import requests

from .utils import ensure_dirs, parse_date_col, normalize_names, filter_date_range, to_parquet
from .schemas import MatchRecord


SACKMANN_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"


def _load_year(year: int) -> pd.DataFrame | None:
    local = Path(f"data/raw/atp_matches_{year}.csv")
    if local.exists():
        return pd.read_csv(local)
    # Try remote
    url = SACKMANN_BASE.format(year=year)
    try:
        r = requests.get(url, timeout=30)
        if r.ok:
            return pd.read_csv(io.StringIO(r.text))
    except Exception:
        pass
    return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_date_col(df)
    df = normalize_names(df)
    # minimal subset of columns; keep optional stats if present
    keep = [
        "date","tourney_name","surface","round","best_of","winner_name","loser_name",
        "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon",
        "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep]
    # Validate minimal schema
    for _, row in df.head(50).fillna(0).astype({"best_of":"int"}, errors="ignore").iterrows():
        _ = MatchRecord(
            date=str(pd.to_datetime(row["date"]).date()),
            tourney_name=row["tourney_name"],
            surface=row["surface"],
            round=row["round"],
            best_of=int(row["best_of"]) if pd.notna(row["best_of"]) else 3,
            winner_name=row["winner_name"],
            loser_name=row["loser_name"],
        )
    return df


def run_ingestion(start: dt.date, end: dt.date) -> None:
    ensure_dirs()
    years = sorted({start.year, end.year})
    dfs: list[pd.DataFrame] = []
    for y in years:
        df = _load_year(y)
        if df is None:
            continue
        df = _clean(df)
        df = filter_date_range(df, dt.date(y, 1, 1), dt.date(y, 12, 31))
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No data found. Place CSVs into data/raw/ (see README).")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = filter_date_range(all_df, start, end)
    # Focus on ATP main tour on hard; US Open final modeling emphasizes hard features later
    # Don't drop non-hard yet; features will filter appropriately when needed
    to_parquet(all_df, Path("data/interim/matches.parquet"))


