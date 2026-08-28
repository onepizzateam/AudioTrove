"""Optional training integrations, loaded lazily to keep imports lightweight."""

from audiotrove.training.base import BaseTrainer, TrainingConfig

__all__ = ["BaseTrainer", "TrainingConfig", "get_trainer"]


def get_trainer(framework: str, config: TrainingConfig) -> BaseTrainer:
    """Return the requested trainer; heavy framework imports remain lazy."""
    framework = framework.lower()
    modules = {
        "f5tts": ("audiotrove.training.f5tts", "F5TTSTrainer"),
        "styletts2": ("audiotrove.training.styletts2", "StyleTTS2Trainer"),
        "piper": ("audiotrove.training.piper", "PiperTrainer"),
        "matcha": ("audiotrove.training.matcha", "MatchaTrainer"),
    }
    if framework not in modules:
        raise ValueError(
            f"Unknown training framework: {framework!r}. "
            "Choose from ['f5tts', 'styletts2', 'piper', 'matcha']"
        )
    import importlib
    module_name, class_name = modules[framework]
    return getattr(importlib.import_module(module_name), class_name)(config)
