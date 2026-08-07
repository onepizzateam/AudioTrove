"""AudioTrove inference layer.

Thin wrappers around TTS, ASR and voice-conversion model families so users who
curate data with AudioTrove can run inference without switching tools. Heavy
model dependencies are imported lazily inside ``load()`` so importing this
package never pulls in torch/transformers/etc.
"""

from audiotrove.inference.base import InferenceResult, InferenceSession

__all__ = [
    "InferenceResult",
    "InferenceSession",
    "get_tts_session",
    "get_asr_session",
    "get_vc_session",
]


def get_tts_session(family: str, **kwargs):
    """Return a TTS inference session for ``family``. See inference.tts."""
    from audiotrove.inference.tts import get_tts_session as _factory

    return _factory(family, **kwargs)


def get_asr_session(family: str = "faster_whisper", **kwargs):
    """Return an ASR inference session for ``family``. See inference.asr."""
    from audiotrove.inference.asr import get_asr_session as _factory

    return _factory(family, **kwargs)


def get_vc_session(family: str, **kwargs):
    """Return a voice-conversion inference session for ``family``. See inference.vc."""
    from audiotrove.inference.vc import get_vc_session as _factory

    return _factory(family, **kwargs)
