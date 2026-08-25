from audiotrove.training.piper import PiperTrainer


def test_piper_manifest_is_streamed(tmp_path):
    manifest = tmp_path / "filelist.txt"
    manifest.write_text("a.wav\t1.0\thello\nb.wav\t2.0\tworld\n", encoding="utf-8")
    trainer = PiperTrainer(str(manifest), str(tmp_path / "out"))
    assert list(trainer.iter_records()) == [["a.wav", "1.0", "hello"], ["b.wav", "2.0", "world"]]
