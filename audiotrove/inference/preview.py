"""Optional Kokoro CPU preview and synthetic augmentation adapters."""

from pathlib import Path


def _pipeline():
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise ImportError(
            "Kokoro preview requires the optional extra: pip install audiotrove[infer]"
        ) from exc
    return KPipeline(lang_code="a")


def _write(pipeline, text: str, output_path: str, voice: str) -> str:
    import numpy as np
    import soundfile as sf

    audio_chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    if not audio_chunks:
        raise ValueError("Kokoro produced no audio")
    sf.write(output_path, np.concatenate(audio_chunks), 24000)
    return output_path


def synthesize(text: str, output_path: str, voice: str = "af_heart") -> str:
    return _write(_pipeline(), text, output_path, voice)


def augment_manifest(manifest: str, output_dir: str, voice: str = "af_heart",
                     limit: int | None = None) -> list[str]:
    """Synthesize one opt-in augmentation WAV per manifest transcript, streaming rows."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pipeline = _pipeline()
    outputs = []
    with Path(manifest).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                break
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) != 3 or not fields[2].strip():
                continue
            output = destination / f"kokoro_{index:06d}.wav"
            _write(pipeline, fields[2], str(output), voice)
            outputs.append(str(output))
    return outputs
