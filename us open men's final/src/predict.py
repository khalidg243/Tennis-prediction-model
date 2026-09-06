from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
import yaml
import shap

from .features import build_features
from .utils import ensure_dirs
from .schemas import FinalistsConfig


def _load_finalists(config_path: Optional[Path], a: Optional[str], b: Optional[str]) -> tuple[str, str]:
    if a and b:
        return a, b
    if config_path is None:
        config_path = Path("config/finalists.yml")
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text())
        cfg = FinalistsConfig(**data)
        return cfg.FINALIST_A, cfg.FINALIST_B
    raise ValueError("Finalists not provided. Use --a/--b or provide config/finalists.yml")


def _latest_row_for_pair(df: pd.DataFrame, A: str, B: str) -> pd.DataFrame:
    # Build features then take a single row with A/B and latest date
    feats = build_features(df)
    rows = feats[((feats["A"] == A) & (feats["B"] == B)) | ((feats["A"] == B) & (feats["B"] == A))]
    if rows.empty:
        # Construct a synthetic latest date row by taking last date in df
        last_date = df["date"].max()
        templ = {c: 0.0 for c in feats.columns if c.startswith("diff_")}
        templ.update({"A": A, "B": B, "date": last_date})
        return pd.DataFrame([templ])
    # Take latest
    return rows.sort_values("date").tail(1).drop(columns=["y"], errors="ignore")


def predict_final(a: Optional[str], b: Optional[str], config_path: Optional[Path], start: dt.date, end: dt.date) -> str:
    ensure_dirs()
    A, B = _load_finalists(config_path, a, b)
    model_obj = joblib.load("models/best.joblib")
    model = model_obj["model"]
    feat_cols = model_obj["features"]

    df = pd.read_parquet(Path("data/interim/matches.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()

    if not ((df["winner_name"].eq(A) | df["loser_name"].eq(A)).any() and (df["winner_name"].eq(B) | df["loser_name"].eq(B)).any()):
        raise ValueError(f"Finalists not found in data window: {A}, {B}")

    row = _latest_row_for_pair(df, A, B)
    X = row[feat_cols].fillna(0.0)
    proba_A = float(model.predict_proba(X)[:, 1][0])
    proba_B = 1.0 - proba_A

    pick = A if proba_A >= 0.5 else B

    # Explainability
    explain = ""
    try:
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model)
            sv = explainer(X)
            vals = pd.Series(abs(sv.values[0]), index=feat_cols).sort_values(ascending=False).head(10)
            explain = "\nTop features (approx.):\n" + "\n".join(f"- {k}: {v:.4f}" for k, v in vals.items())
    except Exception:
        if hasattr(model, "feature_importances_"):
            vals = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(10)
            explain = "\nTop features (importance):\n" + "\n".join(f"- {k}: {v:.4f}" for k, v in vals.items())

    out = (
        f"{A} win prob: {proba_A:.2f}\n"
        f"{B} win prob: {proba_B:.2f}\n"
        f"Pick: {pick}" + ("\n" + explain if explain else "")
    )
    return out


