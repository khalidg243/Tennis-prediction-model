from __future__ import annotations

import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import joblib

from .utils import ensure_dirs, seed_everything


def _time_groups(dates: pd.Series, n_groups: int = 5) -> np.ndarray:
    # Bucket dates by quantiles as proxy for blocked CV
    q = pd.qcut(dates.view("int64"), q=n_groups, labels=False, duplicates="drop")
    return q.to_numpy()


def train_and_save(start: dt.date, end: dt.date) -> None:
    ensure_dirs()
    seed_everything()
    X = pd.read_parquet(Path("data/processed/X_train.parquet"))
    y = pd.read_parquet(Path("data/processed/y_train.parquet"))["y"].to_numpy()

    # Numeric features only (diff_*)
    feat_cols = [c for c in X.columns if c.startswith("diff_")]
    Xnum = X[feat_cols].fillna(0.0)

    dates = pd.to_datetime(X["date"]) if "date" in X.columns else pd.Series(pd.Timestamp("2000-01-01"), index=X.index)
    groups = _time_groups(dates)
    cv = GroupKFold(n_splits=5)

    # Baseline Logistic Regression
    logit_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])

    logit = RandomizedSearchCV(
        estimator=logit_pipe,
        param_distributions={
            "clf__C": np.logspace(-2, 2, 20),
            "clf__penalty": ["l2"],
        },
        n_iter=20,
        cv=cv.split(Xnum, y, groups),
        scoring="neg_log_loss",
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    logit.fit(Xnum, y)

    # XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        random_state=42,
        tree_method="hist",
    )
    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions={
            "max_depth": [3, 4, 5, 6],
            "learning_rate": np.logspace(-3, -0.5, 10),
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 2, 3, 5],
        },
        n_iter=25,
        cv=cv.split(Xnum, y, groups),
        scoring="neg_log_loss",
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    xgb_search.fit(Xnum, y)

    # Evaluate both on overall data (proxy; CV used for selection)
    models = {
        "logit": logit.best_estimator_,
        "xgb": xgb_search.best_estimator_,
    }
    scores = {}
    for name, model in models.items():
        proba = model.predict_proba(Xnum)[:, 1]
        scores[name] = {
            "brier": brier_score_loss(y, proba),
            "logloss": log_loss(y, proba),
            "auc": roc_auc_score(y, proba),
            "acc": accuracy_score(y, (proba >= 0.5).astype(int)),
        }

    # Select best by logloss
    best_name = min(scores, key=lambda k: scores[k]["logloss"])
    best_model = models[best_name]
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "features": feat_cols, "type": best_name}, "models/best.joblib")

    print({"selected": best_name, "scores": scores})


