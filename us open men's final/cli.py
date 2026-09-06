import datetime as dt
from pathlib import Path
import typer
from rich import print

from src.utils import default_start_end_from_years
from src import data_ingest, features, train as train_mod, predict as predict_mod

app = typer.Typer(add_completion=False, help="US Open Men's Final Predictor CLI")


def parse_dates(start: str | None, end: str | None) -> tuple[dt.date, dt.date]:
    if start and end:
        return (dt.date.fromisoformat(start), dt.date.fromisoformat(end))
    return default_start_end_from_years()


@app.command()
def ingest(
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD"),
):
    """Download or load raw data and write cleaned interim datasets."""
    s, e = parse_dates(start, end)
    data_ingest.run_ingestion(s, e)
    print("[green]Ingestion completed[/green]")


@app.command("build-features")
def build_features(
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD"),
):
    """Build symmetric training features and save to data/processed."""
    s, e = parse_dates(start, end)
    features.build_and_save_features(s, e)
    print("[green]Features built[/green]")


@app.command()
def train(
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD"),
):
    """Train models, tune, evaluate, and save best.joblib."""
    s, e = parse_dates(start, end)
    train_mod.train_and_save(s, e)
    print("[green]Training completed[/green]")


@app.command()
def predict(
    a: str | None = typer.Option(None, "--a", help="Finalist A name"),
    b: str | None = typer.Option(None, "--b", help="Finalist B name"),
    config: Path | None = typer.Option(Path("config/finalists.yml"), help="Config file with finalists"),
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD"),
):
    """Predict the final outcome and print probabilities."""
    s, e = parse_dates(start, end)
    result = predict_mod.predict_final(a, b, config, s, e)
    print(result)


if __name__ == "__main__":
    app()


