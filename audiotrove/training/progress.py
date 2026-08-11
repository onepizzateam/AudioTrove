"""Training progress callbacks.

The base :class:`TrainingProgressCallback` defines the hook surface a trainer
invokes during a run. :class:`RichProgressCallback` renders a live progress bar
with epoch/step/loss using ``rich`` when it is installed, and degrades to plain
``print`` output otherwise so training never depends on ``rich``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TrainingProgressCallback:
    """No-op callback interface. Override the hooks you care about.

    Trainers call these in order:
        on_epoch_start(epoch, total_epochs)
        on_step(epoch, step, total_steps, loss)   # repeated per step
        on_epoch_end(epoch, avg_loss)
        on_train_end(metrics)
    """

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:  # noqa: D401
        """Called once at the start of each epoch."""

    def on_step(self, epoch: int, step: int, total_steps: int, loss: float) -> None:
        """Called after each optimisation step."""

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        """Called once at the end of each epoch."""

    def on_train_end(self, metrics: dict) -> None:
        """Called once when training finishes."""


class RichProgressCallback(TrainingProgressCallback):
    """Render training progress with ``rich`` when available.

    Falls back to periodic ``print`` statements when ``rich`` is not installed,
    so importing/using this class never fails on a base install.
    """

    def __init__(self, total_epochs: int, total_steps: int = 0):
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self._progress = None
        self._task = None
        self._epoch = 0

        try:
            from rich.progress import (
                BarColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} steps"),
                TextColumn("loss={task.fields[loss]:.4f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
        except Exception:  # noqa: BLE001 - rich missing or incompatible
            self._progress = None

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self._epoch = epoch
        self.total_epochs = total_epochs
        if self._progress is None:
            return
        if self._task is None:
            self._progress.start()
            self._task = self._progress.add_task(
                f"Epoch {epoch}/{total_epochs}",
                total=self.total_steps or None,
                loss=float("nan"),
            )
        else:
            self._progress.reset(
                self._task,
                total=self.total_steps or None,
                description=f"Epoch {epoch}/{total_epochs}",
                loss=float("nan"),
            )

    def on_step(self, epoch: int, step: int, total_steps: int, loss: float) -> None:
        if self._progress is not None and self._task is not None:
            self._progress.update(
                self._task,
                completed=step,
                total=total_steps or None,
                description=f"Epoch {epoch}/{self.total_epochs}",
                loss=loss,
            )
        elif total_steps and step % max(1, total_steps // 10) == 0:
            # Plain fallback: log ~10 updates per epoch.
            print(f"  epoch {epoch}/{self.total_epochs} step {step}/{total_steps} loss={loss:.4f}")

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        if self._progress is None:
            print(f"Epoch {epoch}/{self.total_epochs} complete — avg loss {avg_loss:.4f}")

    def on_train_end(self, metrics: dict) -> None:
        if self._progress is not None:
            try:
                self._progress.stop()
            except Exception:  # noqa: BLE001
                pass
        final = metrics.get("final_loss")
        if final is not None:
            print(f"Training complete — final loss {final:.4f}")
