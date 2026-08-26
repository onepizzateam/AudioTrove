from click.testing import CliRunner

from audiotrove.cli.main import cli


def test_pipeline_run_help_is_additive():
    result = CliRunner().invoke(cli, ["pipeline", "run", "--help"])
    assert result.exit_code == 0
    assert "skip-train" in result.output
    assert "skip-preview" in result.output
