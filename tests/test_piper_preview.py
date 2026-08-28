from audiotrove.training.piper import PiperTrainer
import sys
import types


def test_piper_manifest_is_streamed(tmp_path):
    manifest = tmp_path / "filelist.txt"
    manifest.write_text("a.wav\t1.0\thello\nb.wav\t2.0\tworld\n", encoding="utf-8")
    trainer = PiperTrainer(str(manifest), str(tmp_path / "out"))
    assert list(trainer.iter_records()) == [["a.wav", "1.0", "hello"], ["b.wav", "2.0", "world"]]


def test_piper_training_command_is_cpu_ready(tmp_path, monkeypatch):
    manifest = tmp_path / "filelist.txt"
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    manifest.write_text(f"{source}\t5.0\thello\n", encoding="utf-8")
    captured = {}
    fake_piper = types.ModuleType("piper")
    fake_train = types.ModuleType("piper.train")
    fake_piper.train = fake_train
    monkeypatch.setitem(sys.modules, "piper", fake_piper)
    monkeypatch.setitem(sys.modules, "piper.train", fake_train)

    monkeypatch.setattr("subprocess.run", lambda command, **kwargs: captured.update(
        command=command, kwargs=kwargs
    ))
    result = PiperTrainer(str(manifest), str(tmp_path / "out")).train()

    command = captured["command"]
    assert result == tmp_path / "out" / "config.json"
    assert "--data.cache_dir" in command
    assert "--data.espeak_voice" in command
    assert "--trainer.accelerator" in command
    assert command[command.index("--trainer.accelerator") + 1] == "cpu"


def test_kokoro_augmentation_streams_transcripts(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from audiotrove.inference import preview

    class FakePipeline:
        def __init__(self, lang_code):
            self.lang_code = lang_code

        def __call__(self, text, voice):
            return iter([(None, None, np.zeros(32, dtype=np.float32))])

    fake_kokoro = types.ModuleType("kokoro")
    fake_kokoro.KPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a.wav\t1\thello\nempty.wav\t1\t\n", encoding="utf-8")
    outputs = preview.augment_manifest(str(manifest), str(tmp_path / "aug"))
    assert len(outputs) == 1
    assert sf.info(outputs[0]).samplerate == 24000
