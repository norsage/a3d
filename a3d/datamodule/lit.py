from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from .dataset import ComplexSummary, SAbDabDataset
from .tokenizer import AminoAcidTokenizer


class Seq2SeqDataModule(pl.LightningDataModule):
    datadir: Path
    train_csv: Path
    validation_csv: Path
    batch_size: int
    num_workers: int
    tokenizer: AminoAcidTokenizer

    train_dataset: SAbDabDataset
    val_dataset: SAbDabDataset

    def __init__(
        self,
        datadir: Path,
        train_csv: Path,
        validation_csv: Path,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.datadir = Path(datadir)
        self.train_csv = Path(train_csv)
        self.validation_csv = Path(validation_csv)
        assert self.datadir.is_dir(), f"{self.datadir} does not exist"
        assert self.train_csv.is_file(), f"{self.train_csv} does not exist"
        assert self.validation_csv.is_file(), f"{self.validation_csv} does not exist"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AminoAcidTokenizer()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = SAbDabDataset(self.datadir, self.train_csv, in_memory=True)
            self.val_dataset = SAbDabDataset(self.datadir, self.validation_csv, in_memory=True)
        elif stage in ("validate", "predict"):
            self.val_dataset = SAbDabDataset(self.datadir, self.validation_csv, in_memory=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(
        self, batch: list[ComplexSummary]
    ) -> tuple[BatchEncoding, BatchEncoding, list[ComplexSummary]]:
        antigen = ["+" + item.linear_epitope for item in batch]
        antibody = [item.heavy_chain_sequence + "-" + item.light_chain_sequence for item in batch]

        ag_encoding = self.tokenizer(antigen, return_tensors="pt", padding="longest")
        ab_encoding = self.tokenizer(antibody, return_tensors="pt", padding="longest")
        return ag_encoding, ab_encoding, batch
