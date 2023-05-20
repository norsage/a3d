from enum import IntEnum
from typing import Any, NamedTuple, cast

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn.functional import nll_loss
from transformers import BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5 import T5Config, T5ForConditionalGeneration
from typing_extensions import TypeAlias

from a3d.datamodule.tokenizer import AminoAcidTokenizer
from proteinlib.complex import ComplexSummary

_Batch: TypeAlias = tuple[BatchEncoding, BatchEncoding, list[ComplexSummary]]


class StepOutput(NamedTuple):
    loss: Tensor
    paratope_loss: Tensor
    logits: Tensor


class Fold(IntEnum):
    train = 0
    val = 1


class TinyT5(pl.LightningModule):
    def __init__(
        self,
        t5_config: T5Config,
        tokenizer: AminoAcidTokenizer,
        optimizer: Any,
        lr_scheduler: Any = None,
        paratope_coef: float = 2.0,
    ):
        super().__init__()
        self._t5 = T5ForConditionalGeneration(t5_config)
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._paratope_coef = paratope_coef

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        decoder_attention_mask: Tensor,
    ) -> Seq2SeqLMOutput:
        outputs = cast(
            Seq2SeqLMOutput,
            self._t5.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            ),
        )
        return outputs

    @staticmethod
    def mask_non_paratope(labels: torch.Tensor, paratope_ids: list[list[int]]) -> torch.Tensor:
        indices = torch.tensor([[i, k] for i, pid in enumerate(paratope_ids) for k in pid])
        # TODO: -100?
        masked = torch.zeros_like(labels)
        masked[indices[:, 0], indices[:, 1]] = labels[indices[:, 0], indices[:, 1]]
        return masked

    @staticmethod
    def nll_loss(preds: Tensor, targets: Tensor) -> torch.Tensor:
        _, N, C = preds.shape
        nll = nll_loss(
            torch.nn.functional.log_softmax(preds.view(-1, C), dim=-1),
            targets.view(-1),
            ignore_index=0,
        )
        return nll

    def _step(self, batch: _Batch) -> StepOutput:
        antigen_encoding, antibody_encoding, item_info = batch
        paratope_ids = [
            list(item.heavy_chain_contacts)
            + [x + len(item.heavy_chain_sequence) + 1 for x in item.light_chain_contacts]
            for item in item_info
        ]
        outputs = self.forward(
            input_ids=antigen_encoding.input_ids,
            attention_mask=antigen_encoding.attention_mask,
            labels=antibody_encoding.input_ids,
            decoder_attention_mask=antibody_encoding.attention_mask,
        )

        nll = self.nll_loss(outputs.logits, antibody_encoding.input_ids)
        nll_paratope = self.nll_loss(
            outputs.logits, self.mask_non_paratope(antibody_encoding.input_ids, paratope_ids)
        )

        return StepOutput(loss=nll, paratope_loss=nll_paratope, logits=outputs.logits)

    def training_step(self, batch: _Batch, batch_idx: int) -> Tensor:
        step_output = self._step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        fold = "train"
        self.log(f"{fold}_loss", step_output.loss, on_epoch=True, on_step=False)
        self.log(
            f"{fold}_paratope_loss",
            step_output.paratope_loss,
            on_epoch=True,
            on_step=False,
        )

        return step_output.loss + self._paratope_coef * step_output.paratope_loss

    def validation_step(self, batch: _Batch, batch_idx: int) -> Tensor:
        step_output = self._step(batch)
        fold = "val"
        self.log(f"{fold}_loss", step_output.loss, on_epoch=True, on_step=False)
        self.log(
            f"{fold}_paratope_loss",
            step_output.paratope_loss,
            on_epoch=True,
            on_step=False,
        )

        return step_output.loss

    def configure_optimizers(self) -> dict[str, Any]:
        # create optimizer
        optimizer = self._optimizer(self.parameters())
        if not self._lr_scheduler:
            return {"optimizer": optimizer}
        scheduler = self._lr_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def generate(self, inputs: Tensor) -> list[str]:
        self.eval()
        return self._t5.generate(inputs=inputs)

    def generate_and_decode(self, inputs: Tensor) -> list[str]:
        outputs = self.generate(inputs)
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
