import datetime as dt
import pandas as pd
from src.utils import filter_date_range


def test_date_filter():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01"])
    })
    out = filter_date_range(df, dt.date(2024, 2, 1), dt.date(2024, 12, 31))
    assert len(out) == 1

