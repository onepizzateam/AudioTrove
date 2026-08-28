"""Piper CLI entry point with a CPU-safe checkpoint policy."""

from lightning.pytorch.callbacks import ModelCheckpoint


def main() -> None:
    from piper.train.__main__ import VitsLightningCLI
    from piper.train.vits.dataset import VitsDataModule
    from piper.train.vits.lightning import VitsModel

    VitsLightningCLI(
        VitsModel,
        VitsDataModule,
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(
                    monitor="val_mel",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    filename="epoch={epoch}-val_mel={val_mel:.4f}",
                    auto_insert_metric_name=False,
                )
            ]
        },
    )


if __name__ == "__main__":
    main()
