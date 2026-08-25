"""Optional Kokoro CPU preview adapter."""


def synthesize(text: str, output_path: str, voice: str = "af_heart") -> str:
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise ImportError(
            "Kokoro preview requires the optional extra: pip install audiotrove[infer]"
        ) from exc
    import soundfile as sf

    pipeline = KPipeline(lang_code="a")
    audio_chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    if not audio_chunks:
        raise ValueError("Kokoro produced no audio")
    import numpy as np
    sf.write(output_path, np.concatenate(audio_chunks), 24000)
    return output_path
