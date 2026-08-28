from click.testing import CliRunner
import json

from audiotrove.cli.main import cli


def test_pipeline_run_help_is_additive():
    result = CliRunner().invoke(cli, ["pipeline", "run", "--help"])
    assert result.exit_code == 0
    assert "skip-train" in result.output
    assert "skip-preview" in result.output


def test_pipeline_resume_after_training_failure(tmp_path, monkeypatch):
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fixture")
    output_path = tmp_path / "out"
    calls = {"curate": 0, "train": 0, "preview": 0}

    def fake_tts_pipeline(**kwargs):
        calls["curate"] += 1
        (tmp_path / "out" / "filelist.txt").write_text("audio.wav\t1\thello\n", encoding="utf-8")
        return {"kept": 1, "filtered": 0, "total_duration_seconds": 1.0,
                "output_files": [], "qc_report": "qc_report.json"}

    class FakeTrainer:
        def __init__(self, manifest, output_dir):
            self.output_dir = output_dir

        def train(self):
            calls["train"] += 1
            if calls["train"] == 1:
                raise RuntimeError("simulated interruption")
            path = tmp_path / "out" / "piper" / "last.ckpt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")

    def fake_synthesize(text, output_path, voice):
        calls["preview"] += 1
        Path(output_path).write_bytes(b"wav")

    from pathlib import Path
    monkeypatch.setattr("audiotrove.pipelines.tts.tts_pipeline", fake_tts_pipeline)
    monkeypatch.setattr("audiotrove.training.piper.PiperTrainer", FakeTrainer)
    monkeypatch.setattr("audiotrove.inference.preview.synthesize", fake_synthesize)

    first = CliRunner().invoke(cli, ["pipeline", "run", str(input_path), str(output_path)])
    assert first.exit_code != 0
    assert json.loads((output_path / "pipeline_state.json").read_text()) == {"stage": "curated"}, first.output

    second = CliRunner().invoke(cli, ["pipeline", "run", str(input_path), str(output_path)])
    assert second.exit_code == 0, second.output
    assert json.loads((output_path / "pipeline_state.json").read_text()) == {"stage": "previewed"}
    assert calls == {"curate": 1, "train": 2, "preview": 1}
