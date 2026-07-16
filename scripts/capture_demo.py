from click.testing import CliRunner
from audiotrove.cli import main
import re
from pathlib import Path


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def normalize_slashes(s: str) -> str:
    return s.replace("\\", "/")


def run_curate():
    runner = CliRunner()
    # Run curate against tests fixtures with minimal filters
    result = runner.invoke(
        main.cli,
        ["curate", "tests/fixtures", "packaging/demo_out", "--workers", "1", "--snr-min", "15", "--vad-threshold", "0.3"],
    )
    out = strip_ansi(result.output)
    out = normalize_slashes(out)
    return out


def run_inspect():
    runner = CliRunner()
    result = runner.invoke(main.cli, ["inspect", "tests/fixtures", "--extensions", "wav"]) 
    out = strip_ansi(result.output)
    out = normalize_slashes(out)
    return out


def main_fn():
    Path("packaging").mkdir(exist_ok=True)
    curate_out = run_curate()
    inspect_out = run_inspect()
    with open("packaging/curate_demo_output_clean.txt", "w", encoding="utf8") as f:
        f.write(curate_out)
    with open("packaging/inspect_demo_output_clean.txt", "w", encoding="utf8") as f:
        f.write(inspect_out)


if __name__ == "__main__":
    main_fn()
