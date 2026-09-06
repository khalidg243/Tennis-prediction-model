from typer.testing import CliRunner
from cli import app


def test_cli_parse_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "US Open Men's Final Predictor CLI" in result.stdout

