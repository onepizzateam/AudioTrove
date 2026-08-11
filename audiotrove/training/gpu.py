"""Multi-GPU / DDP setup helpers used by trainers when ``num_gpus > 1``."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def launch_ddp(train_fn, num_gpus: int, **kwargs) -> None:
    """Launch ``train_fn`` across ``num_gpus`` using torch.multiprocessing.spawn.

    Args:
        train_fn: Callable with signature ``train_fn(rank, world_size, kwargs)``.
        num_gpus: Number of processes / GPUs to spawn.
        **kwargs: Extra arguments forwarded to ``train_fn``.
    """
    import torch.multiprocessing as mp

    mp.spawn(train_fn, args=(num_gpus, kwargs), nprocs=num_gpus, join=True)


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialise the default process group for DDP.

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
        backend: Distributed backend (``"nccl"`` for CUDA, ``"gloo"`` for CPU).
    """
    import torch.distributed as dist

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp() -> None:
    """Destroy the default process group."""
    import torch.distributed as dist

    dist.destroy_process_group()


def configure_threads(torch_threads=None, interop_threads=None) -> int:
    """Set intra-/inter-op thread counts for CPU training.

    Unlike the curation executor (which caps torch to half the cores so decode
    threads keep a share), training is compute-bound, so the default here is to
    let torch use *all* logical cores.

    Args:
        torch_threads: Intra-op thread count. ``None`` uses ``os.cpu_count()``.
        interop_threads: Optional inter-op thread count. Left untouched when
            ``None`` (setting it after the first parallel region raises).

    Returns:
        The intra-op thread count that was applied.
    """
    import os

    if torch_threads is None:
        torch_threads = os.cpu_count() or 4
    torch_threads = max(1, int(torch_threads))

    try:
        import torch

        torch.set_num_threads(torch_threads)
        if interop_threads is not None:
            torch.set_num_interop_threads(max(1, int(interop_threads)))
    except Exception:  # noqa: BLE001 - torch missing or threads already fixed
        logger.debug("configure_threads: unable to set torch thread counts", exc_info=True)
    return torch_threads


def apply_ipex(model, optimizer=None, dtype=None):
    """Optimise ``model`` (and ``optimizer``) with Intel Extension for PyTorch.

    A no-op that returns the inputs unchanged when IPEX is unavailable, so
    callers can wrap every model without branching on the backend.

    Args:
        model: The ``torch.nn.Module`` to optimise.
        optimizer: Optional optimizer to co-optimise (fused where supported).
        dtype: Optional target dtype (e.g. ``torch.bfloat16``).

    Returns:
        ``(model, optimizer)`` — optimised when IPEX applied, else unchanged.
    """
    from audiotrove.gpu.device import ipex_available

    if not ipex_available():
        return model, optimizer

    try:
        import intel_extension_for_pytorch as ipex

        if optimizer is not None:
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)
        else:
            model = ipex.optimize(model, dtype=dtype)
        logger.info("Applied IPEX optimisation (dtype=%s).", dtype)
    except Exception:  # noqa: BLE001 - never let an optional optimiser break training
        logger.warning("IPEX is importable but ipex.optimize failed; continuing without it.")
    return model, optimizer


