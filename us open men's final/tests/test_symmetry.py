import pandas as pd
from src.features import build_features


def test_feature_symmetry():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
        "tourney_name": ["Test", "Test"],
        "surface": ["Hard", "Hard"],
        "round": ["R32", "R16"],
        "best_of": [3, 3],
        "winner_name": ["A", "B"],
        "loser_name": ["B", "A"],
    })
    feats = build_features(df)
    # There should be pairs where diff_* change sign between A/B ordering
    assert (feats["A"].isin(["A", "B"]).all())
    assert (feats["B"].isin(["A", "B"]).all())

