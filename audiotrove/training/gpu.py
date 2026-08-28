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
