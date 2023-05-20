import json
from pathlib import Path
from typing import NamedTuple

from torch.utils.data import Dataset
from tqdm import tqdm

from proteinlib.complex import ComplexSummary


class ItemUID(NamedTuple):
    pdb_uid: str
    heavy_chain: str
    light_chain: str
    antigen_chain: str

    def __str__(self) -> str:
        return f"{self.pdb_uid}_{self.heavy_chain}+{self.light_chain}-{self.antigen_chain}"


class SAbDabDataset(Dataset):
    datadir: Path
    subset_csv: Path
    in_memory: bool

    _complex_ids: list[ItemUID]
    _data: list[ComplexSummary]

    def __init__(
        self,
        datadir: Path,
        subset_csv: Path,
        in_memory: bool = True,
        sep: str = ",",
    ):
        self.datadir = datadir
        self.subset_csv = subset_csv
        self.in_memory = in_memory

        self._complex_ids = [
            item
            for item in map(lambda s: ItemUID(*s.split(sep)), self.subset_csv.read_text().split())
            if (self.datadir / f"{item}.json").is_file()
        ]
        self._data = []
        if in_memory:
            self._data = [
                self.item_from_json(complex_id)
                for complex_id in tqdm(self._complex_ids, total=len(self._complex_ids))
            ]

    def __len__(self) -> int:
        return len(self._complex_ids)

    def __getitem__(self, index: int) -> ComplexSummary:
        if self.in_memory:
            return self._data[index]
        return self.item_from_json(self._complex_ids[index])

    def item_from_json(self, item_uid: ItemUID) -> ComplexSummary:
        complex_id = str(item_uid)
        record = json.loads((self.datadir / f"{complex_id}.json").read_text())
        return ComplexSummary(**record)
