"""Optional DeepFilterNet2 speech enhancement."""

import importlib.util

import numpy as np

from audiotrove.base import AudioTransformer
from audiotrove.document import AudioDocument


class DeepFilterEnhancer(AudioTransformer):
    """Enhance audio with DeepFilterNet2, loading the model lazily."""

    name = "deepfilternet2"
    _required_sample_rate = 48000

    def __init__(self):
        if importlib.util.find_spec("df") is None:
            raise ImportError(
                "Enhancement requires deepfilternet. Install it with: "
                "pip install audiotrove[enhance]"
            )

        try:
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise ImportError(
                "Enhancement requires deepfilternet. Install it with: "
                "pip install audiotrove[enhance]"
            ) from exc

        self._enhance_fn = enhance
        self._init_df = init_df
        self._model = None
        self._df_state = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        loaded = self._init_df()
        self._model, self._df_state = loaded[:2]

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return np.asarray(audio, dtype=np.float32)
        target_length = max(1, round(len(audio) * target_rate / source_rate))
        source_positions = np.arange(len(audio), dtype=np.float32)
        target_positions = np.linspace(0, len(audio) - 1, target_length, dtype=np.float32)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance a mono audio array and return float32 at its original rate."""
        import torch

        original = np.asarray(audio, dtype=np.float32)
        enhanced_input = self._resample(
            original, sample_rate, self._required_sample_rate
        )
        self._load_model()

        tensor = torch.from_numpy(enhanced_input)
        with torch.no_grad():
            enhanced = self._enhance_fn(self._model, self._df_state, tensor)
        if isinstance(enhanced, tuple):
            enhanced = enhanced[0]
        if hasattr(enhanced, "detach"):
            enhanced = enhanced.detach().cpu().numpy()

        enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)
        result = self._resample(
            enhanced, self._required_sample_rate, sample_rate
        )
        if len(result) != len(original):
            result = np.resize(result, len(original)).astype(np.float32)
        return result.astype(np.float32, copy=False)

    def transform(self, doc: AudioDocument) -> AudioDocument:
        doc.audio = self.enhance(doc.audio, doc.sample_rate)
        doc.duration_seconds = float(len(doc.audio)) / doc.sample_rate
        doc.metadata["enhanced"] = True
        doc.metadata["enhancement_model"] = "deepfilternet2"
        return doc