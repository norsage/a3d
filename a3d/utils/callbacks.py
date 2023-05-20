from typing import cast

import numpy as np
import pytorch_lightning as pl
from aim import Distribution, Text
from aim.pytorch_lightning import AimLogger
from torch import Tensor, device
from torch.utils.data import DataLoader, Subset
from transformers import BatchEncoding

from a3d.model.t5 import TinyT5
from proteinlib.complex import ComplexSummary

from .metrics import AlignmentMetrics, get_metrics


class GenerateSequences(pl.Callback):
    def __init__(self, n_sequences: int = 4, n_samples: int = 100) -> None:
        self._n_sequences = n_sequences
        self.n_samples = n_samples

    def generate_dataloader(
        self, pl_module: TinyT5, loader: DataLoader
    ) -> tuple[list[str], list[ComplexSummary]]:
        generated_samples: list[str] = []
        items: list[ComplexSummary] = []
        for batch in loader:
            inputs, _, item_info = batch
            batch_generated = pl_module.generate_and_decode(
                inputs.input_ids.to(device=cast(device, pl_module.device))
            )
            generated_samples.extend(batch_generated)
            items.extend(item_info)
        return generated_samples, items

    @classmethod
    def calculate_and_log_metrics(
        cls,
        generated_samples: list[str],
        items: list[ComplexSummary],
        pl_module: pl.LightningModule,
    ) -> None:
        # collect metrics
        vh_scores: list[tuple[float, float, float]] = []
        vl_scores: list[tuple[float, float, float]] = []
        for generated, item in zip(generated_samples, items):
            if metrics := cls.item_metrics(generated, item):
                vh_metrics, vl_metrics = metrics
                vh_scores.append(
                    (
                        vh_metrics.fr_score,
                        vh_metrics.cdr_score,
                        vh_metrics.paratope_score,
                    )
                )
                vl_scores.append(
                    (
                        vl_metrics.fr_score,
                        vl_metrics.cdr_score,
                        vl_metrics.paratope_score,
                    )
                )

            else:
                vh_scores.append((0, 0, 0))
                vl_scores.append((0, 0, 0))

        vh_fr_score, vh_cdr_score, vh_paratope_score = zip(*vh_scores)
        vl_fr_score, vl_cdr_score, vl_paratope_score = zip(*vl_scores)

        # log distributions
        logger = cast(AimLogger, pl_module.logger)
        logger.experiment.track(
            Distribution(vh_fr_score), name="vh_fr_score", step=pl_module.global_step
        )
        logger.experiment.track(
            Distribution(vh_cdr_score), name="vh_cdr_score", step=pl_module.global_step
        )
        logger.experiment.track(
            Distribution(vh_paratope_score),
            name="vh_paratope_score",
            step=pl_module.global_step,
        )
        logger.experiment.track(
            Distribution(vl_fr_score), name="vl_fr_score", step=pl_module.global_step
        )
        logger.experiment.track(
            Distribution(vl_cdr_score), name="vl_cdr_score", step=pl_module.global_step
        )
        logger.experiment.track(
            Distribution(vl_paratope_score),
            name="vl_paratope_score",
            step=pl_module.global_step,
        )

        # load mean values
        pl_module.log("avg_vh_fr_score", np.mean(vh_fr_score), on_epoch=True, on_step=False)
        pl_module.log("avg_vh_cdr_score", np.mean(vh_cdr_score), on_epoch=True, on_step=False)
        pl_module.log(
            "avg_vh_paratope_score",
            np.mean(vh_paratope_score),
            on_epoch=True,
            on_step=False,
        )
        pl_module.log("avg_vl_fr_score", np.mean(vl_fr_score), on_epoch=True, on_step=False)
        pl_module.log("avg_vl_cdr_score", np.mean(vl_cdr_score), on_epoch=True, on_step=False)
        pl_module.log(
            "avg_vl_paratope_score",
            np.mean(vl_paratope_score),
            on_epoch=True,
            on_step=False,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the val epoch ends."""
        pl_module = cast(TinyT5, pl_module)

        datamodule: pl.LightningDataModule = trainer.datamodule  # type: ignore[attr-defined]
        loader = cast(DataLoader, datamodule.val_dataloader())

        # calculate_metrics
        generated_samples, items = self.generate_dataloader(
            pl_module, self.create_subset_loader(loader, n=self.n_samples)
        )
        self.calculate_and_log_metrics(generated_samples, items, pl_module)

        # sample sequences
        batch = next(iter(self.create_subset_loader(loader, n=self._n_sequences)))
        aim_text = Text("\n".join(self.process_batch(batch, pl_module)))

        logger = cast(AimLogger, trainer.logger)
        logger.experiment.track(aim_text, name="generated_samples", step=trainer.global_step)

    @staticmethod
    def create_subset_loader(loader: DataLoader, n: int) -> DataLoader:
        subset = Subset(
            loader.dataset,
            indices=np.random.randint(len(loader.dataset), size=n),  # type: ignore[arg-type]
        )
        return DataLoader(subset, batch_size=loader.batch_size, collate_fn=loader.collate_fn)

    @classmethod
    def process_batch(
        cls, batch: tuple[BatchEncoding, BatchEncoding, list[ComplexSummary]], pl_module: TinyT5
    ) -> list[str]:
        # antigen, _, item_info = batch
        inputs, _, item_info = batch
        generated_sequences = pl_module.generate_and_decode(
            inputs.input_ids.to(device=cast(device, pl_module.device))
        )
        return [
            cls.process_item(generated, item)
            for generated, item in zip(generated_sequences, item_info)
        ]

    @staticmethod
    def item_metrics(
        generated: str, item: ComplexSummary
    ) -> tuple[AlignmentMetrics, AlignmentMetrics] | None:
        if generated[0] == "=":
            generated = generated[1:]
        parts = generated.split("-")
        if not (len(parts) == 2 and min(map(len, parts)) > 0):
            return None

        generated_heavy, generated_light = parts
        heavy_metrics = get_metrics(
            generated_heavy,
            item.heavy_chain_sequence,
            item.heavy_chain_regions,
            item.heavy_chain_contacts,
        )
        light_metrics = get_metrics(
            generated_light,
            item.light_chain_sequence,
            item.light_chain_regions,
            item.light_chain_contacts,
        )

        return heavy_metrics, light_metrics

    @classmethod
    def process_item(cls, generated: str, item: ComplexSummary) -> str:
        metrics = cls.item_metrics(generated, item)
        if not metrics:
            return f"> {item.uid} {item.linear_epitope} [INVALID]\n" f"{generated}\n"

        heavy_metrics, light_metrics = metrics
        return f"> {item.uid} {item.linear_epitope} [VH+VL]\n" f"{heavy_metrics}\n{light_metrics}\n"
