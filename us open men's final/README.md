# US Open Men's Final Predictor

Predict the winner of the US Open Men’s Singles Final using only data from last year through today. Reproducible, typed, and with a clean CLI.

## Quickstart

### 1) Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows PowerShell
pip install -r requirements.txt
```

If you use bash/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) End-to-end run (ingest → features → train)

```bash
python cli.py ingest
python cli.py build-features
python cli.py train
```

Optional date window overrides (defaults: last year-01-01 to today):

```bash
python cli.py ingest --start 2024-01-01 --end 2025-09-07
python cli.py build-features --start 2024-01-01 --end 2025-09-07
python cli.py train --start 2024-01-01 --end 2025-09-07
```

### 3) Predict today’s final

Using config (default `config/finalists.yml`):

```bash
python cli.py predict --config config/finalists.yml
```

Override finalists via CLI:

```bash
python cli.py predict --a "Player A" --b "Player B"
```

Example:

```bash
python cli.py predict --a "Novak Djokovic" --b "Carlos Alcaraz"
```

## Repository Structure

```
usopen-final-predictor/
  README.md
  requirements.txt
  src/
    data_ingest.py
    features.py
    train.py
    predict.py
    utils.py
    schemas.py
  notebooks/
    01_eda.ipynb
  config/
    finalists.yml
  data/
    raw/
    interim/
    processed/
  cli.py
  tests/
    test_cli.py
    test_date_filter.py
    test_symmetry.py
```

## Data Sources and Fallbacks

This project uses public, unauthenticated sources when accessible:

- Jeff Sackmann’s ATP match CSVs (seasonal): `https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_YYYY.csv`.
- Optional ratings (e.g., Elo) if a public CSV is accessible for the same period (not required).

If downloads are blocked, place the CSVs here:

- `data/raw/atp_matches_2024.csv`
- `data/raw/atp_matches_2025.csv` (if current year is 2025)
- Optional: `data/raw/elo_2024_2025.csv` with columns including `player`, `date`, `elo`, and `surface` if applicable.

Expected minimal columns in match data (validated):

- `tourney_name`, `surface`, `round`, `best_of`, `match_num` (optional)
- `tourney_date` or `date` (YYYYMMDD or ISO date)
- `winner_name`, `loser_name`
- Optional stats: `w_ace`, `w_df`, `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `l_ace`, `l_df`, `l_svpt`, `l_1stIn`, `l_1stWon`, `l_2ndWon`

The ingestion step normalizes column names and dates, filters to ATP main tour matches, and saves cleaned data to `data/interim/`.

## Modeling

- Time-aware split within the 2-year window.
- Baseline: LogisticRegression (standardized).
- Stronger: XGBoost (binary:logistic). Blocked CV by date to avoid leakage.
- Metrics: Brier score, log loss, AUC, accuracy.

## CLI

```bash
python cli.py --help
```

Commands:

- `ingest` → pulls/validates data into `data/interim/`
- `build-features` → writes `data/processed/{X_train.parquet,y_train.parquet}`
- `train` → trains, tunes, saves `models/best.joblib`
- `predict` → prints probabilities and pick for finalists

All commands accept `--start` and `--end` to constrain dates.

## Notes

- Only data from Jan 1 of last year through today are used.
- Respect any public data ToS. Scraping is not included by default. You can provide CSVs instead as described above.
- Reproducibility: all randomness is seeded globally.

## Troubleshooting

- If finalists are not found in the data window, you’ll get a clear error. Ensure names match source CSVs.
- If downloads fail, place the files manually in `data/raw/` as described and rerun.


