"""AudioTrove training layer.

Thin wrappers around TTS training frameworks (F5-TTS, StyleTTS2, Piper,
Matcha-TTS). Heavy dependencies are imported lazily inside ``train()`` so
importing this package stays cheap and dependency-free.
"""

from audiotrove.training.base import BaseTrainer, TrainingConfig

__all__ = ["BaseTrainer", "TrainingConfig", "get_trainer"]


def get_trainer(framework: str, config: "TrainingConfig") -> "BaseTrainer":
    """Return a trainer instance for ``framework``.

    Args:
        framework: One of ``"f5tts"``, ``"styletts2"``, ``"piper"``, ``"matcha"``.
        config: A :class:`TrainingConfig`.

    Raises:
        ValueError: Unknown framework.
    """
    framework = framework.lower()
    if framework == "f5tts":
        from audiotrove.training.f5tts import F5TTSTrainer

        return F5TTSTrainer(config)
    if framework == "styletts2":
        from audiotrove.training.styletts2 import StyleTTS2Trainer

        return StyleTTS2Trainer(config)
    if framework == "piper":
        from audiotrove.training.piper import PiperTrainer

        return PiperTrainer(config)
    if framework == "matcha":
        from audiotrove.training.matcha import MatchaTrainer

        return MatchaTrainer(config)
    raise ValueError(
        f"Unknown training framework: {framework!r}. "
        "Choose from ['f5tts', 'styletts2', 'piper', 'matcha']"
    )
