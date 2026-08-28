"""Tests for the GPU acceleration layer.

Covers:
- audiotrove/gpu/device.py: get_device / device_info / to_device
- audiotrove/base.py: GPUFilter / GPUTransformer abstract base classes
- audiotrove/document.py: gpu_tensor field pickle-clearing + __eq__ exclusion
"""

import pickle

import numpy as np
import pytest

from audiotrove.gpu import device as device_mod
from audiotrove.gpu.device import device_info, get_device, to_device
from audiotrove.base import GPUFilter, GPUTransformer
from audiotrove.document import AudioDocument

HAS_TORCH = device_mod.HAS_TORCH
if HAS_TORCH:
    import torch


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------
def test_get_device_cpu():
    dev = get_device("cpu")
    assert dev.type == "cpu"


def test_get_device_auto_returns_device():
    dev = get_device("auto")
    assert dev.type in {"cpu", "cuda", "mps"}


def test_get_device_default_is_auto():
    # Passing nothing should behave like "auto".
    assert get_device().type in {"cpu", "cuda", "mps"}


def test_get_device_empty_string_falls_back_to_auto():
    assert get_device("").type in {"cpu", "cuda", "mps"}


def test_get_device_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dev = get_device("cuda")
    assert dev.type == "cpu"


def test_get_device_cuda_selected_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    dev = get_device("cuda")
    assert dev.type == "cuda"


def test_get_device_mps_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(device_mod, "_mps_available", lambda: False)
    dev = get_device("mps")
    assert dev.type == "cpu"


def test_get_device_mps_selected_when_available(monkeypatch):
    monkeypatch.setattr(device_mod, "_mps_available", lambda: True)
    dev = get_device("mps")
    assert dev.type == "mps"


def test_get_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device("auto").type == "cuda"


def test_get_device_auto_prefers_mps_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_mod, "_mps_available", lambda: True)
    assert get_device("auto").type == "mps"


def test_get_device_auto_cpu_when_nothing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_mod, "_mps_available", lambda: False)
    assert get_device("auto").type == "cpu"


def test_get_device_case_insensitive(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device("CPU").type == "cpu"


def test_get_device_raises_without_torch(monkeypatch):
    monkeypatch.setattr(device_mod, "HAS_TORCH", False)
    with pytest.raises(RuntimeError):
        get_device("cpu")


# ---------------------------------------------------------------------------
# _mps_available
# ---------------------------------------------------------------------------
def test_mps_available_without_torch(monkeypatch):
    monkeypatch.setattr(device_mod, "HAS_TORCH", False)
    assert device_mod._mps_available() is False


def test_mps_available_no_backend(monkeypatch):
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)
    assert device_mod._mps_available() is False


def test_mps_available_backend_raises(monkeypatch):
    class _Boom:
        def is_available(self):
            raise RuntimeError("nope")

    monkeypatch.setattr(torch.backends, "mps", _Boom(), raising=False)
    assert device_mod._mps_available() is False


def test_mps_available_true(monkeypatch):
    class _Ok:
        def is_available(self):
            return True

    monkeypatch.setattr(torch.backends, "mps", _Ok(), raising=False)
    assert device_mod._mps_available() is True


# ---------------------------------------------------------------------------
# device_info
# ---------------------------------------------------------------------------
def test_device_info_keys():
    info = device_info()
    for key in (
        "torch_available",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_names",
        "mps_available",
        "selected",
    ):
        assert key in info


def test_device_info_without_torch(monkeypatch):
    monkeypatch.setattr(device_mod, "HAS_TORCH", False)
    info = device_info()
    assert info["torch_available"] is False
    assert info["selected"] == "cpu"
    assert info["cuda_available"] is False


def test_device_info_reports_torch_version():
    info = device_info()
    assert info["torch_available"] is True
    assert "torch_version" in info


def test_device_info_with_fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: f"FakeGPU{i}")
    info = device_info()
    assert info["cuda_available"] is True
    assert info["cuda_device_count"] == 2
    assert info["cuda_device_names"] == ["FakeGPU0", "FakeGPU1"]


# ---------------------------------------------------------------------------
# to_device
# ---------------------------------------------------------------------------
def test_to_device_cpu_keeps_fp32():
    t = torch.zeros(4, dtype=torch.float32)
    out = to_device(t, torch.device("cpu"))
    assert out.device.type == "cpu"
    assert out.dtype == torch.float32


def test_to_device_accepts_string_device():
    t = torch.zeros(4)
    out = to_device(t, "cpu")
    assert out.device.type == "cpu"


def test_to_device_int_tensor_unchanged_dtype():
    t = torch.zeros(4, dtype=torch.int64)
    out = to_device(t, torch.device("cpu"))
    assert out.dtype == torch.int64


def test_to_device_cuda_uses_fp16(monkeypatch):
    # Emulate a CUDA move without a real GPU by intercepting Tensor.to.
    captured = {}
    real_to = torch.Tensor.to

    def fake_to(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Stay on CPU to keep the test hardware-independent.
        if "dtype" in kwargs:
            return real_to(self, dtype=kwargs["dtype"])
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    t = torch.zeros(4, dtype=torch.float32)
    to_device(t, torch.device("cuda"))
    assert captured["kwargs"].get("dtype") == torch.float16


def test_to_device_raises_without_torch(monkeypatch):
    t = torch.zeros(2)
    monkeypatch.setattr(device_mod, "HAS_TORCH", False)
    with pytest.raises(RuntimeError):
        to_device(t, "cpu")


# ---------------------------------------------------------------------------
# GPUFilter / GPUTransformer base classes
# ---------------------------------------------------------------------------
class _DummyGPUFilter(GPUFilter):
    name = "dummy_gpu_filter"

    def __init__(self, device="cpu"):
        self._device = device

    @property
    def device(self):
        return self._device

    def filter(self, doc):
        return True


class _DummyGPUTransformer(GPUTransformer):
    name = "dummy_gpu_transformer"

    def __init__(self, device="cpu"):
        self._device = device

    @property
    def device(self):
        return self._device

    def transform(self, doc):
        return doc


def test_gpu_filter_default_to_returns_self():
    f = _DummyGPUFilter()
    assert f.to("cpu") is f


def test_gpu_transformer_default_to_returns_self():
    t = _DummyGPUTransformer()
    assert t.to("cpu") is t


def test_gpu_filter_device_property():
    f = _DummyGPUFilter(device="cpu")
    assert f.device == "cpu"


def test_gpu_transformer_device_property():
    t = _DummyGPUTransformer(device="cpu")
    assert t.device == "cpu"


def test_gpu_filter_cannot_instantiate_without_device():
    class Incomplete(GPUFilter):
        name = "incomplete"

        def filter(self, doc):
            return True

    with pytest.raises(TypeError):
        Incomplete()


def test_gpu_transformer_cannot_instantiate_without_device():
    class Incomplete(GPUTransformer):
        name = "incomplete"

        def transform(self, doc):
            return doc

    with pytest.raises(TypeError):
        Incomplete()


# ---------------------------------------------------------------------------
# AudioDocument.gpu_tensor
# ---------------------------------------------------------------------------
def _make_doc():
    audio = np.zeros(16, dtype=np.float32)
    return AudioDocument(
        audio=audio,
        sample_rate=16000,
        source_path="x.wav",
        duration_seconds=0.001,
        doc_id="doc1",
    )


def test_gpu_tensor_defaults_to_none():
    doc = _make_doc()
    assert doc.gpu_tensor is None


def test_gpu_tensor_cleared_on_pickle():
    doc = _make_doc()
    doc.gpu_tensor = torch.zeros(16)
    restored = pickle.loads(pickle.dumps(doc))
    assert restored.gpu_tensor is None
    # The CPU audio survives the round-trip.
    assert np.array_equal(restored.audio, doc.audio)


def test_gpu_tensor_excluded_from_equality():
    a = _make_doc()
    b = _make_doc()
    a.gpu_tensor = torch.ones(16)
    b.gpu_tensor = None
    # Different gpu_tensor must not change equality.
    assert a == b


def test_getstate_clears_gpu_tensor_directly():
    doc = _make_doc()
    doc.gpu_tensor = torch.zeros(4)
    state = doc.__getstate__()
    assert state["gpu_tensor"] is None


def test_setstate_restores_and_clears():
    doc = _make_doc()
    state = doc.__getstate__()
    new = AudioDocument.__new__(AudioDocument)
    new.__setstate__(state)
    assert new.gpu_tensor is None
    assert new.doc_id == "doc1"


def test_eq_non_document_returns_false():
    doc = _make_doc()
    assert (doc == "not a doc") is False
