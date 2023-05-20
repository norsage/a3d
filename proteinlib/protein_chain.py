from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Iterator, cast

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue
from numpy.typing import NDArray


@dataclass
class ProteinChain:
    residues: Polypeptide

    def linear_segment_sequence(self, start: int, end: int) -> str:
        return self.residues.get_sequence()[start:end]

    def select_indices(self, indices: list[int]) -> ProteinChain:
        return ProteinChain([self.residues[i] for i in indices])

    @cached_property
    def atoms(self) -> tuple[list[int], list[Atom]]:
        chain_residue_ids: list[int] = []
        chain_atoms: list[Atom] = []
        for i, residue in enumerate(self.residues):
            residue_atoms: list[Atom] = list(cast(Residue, residue).get_atoms())
            chain_residue_ids += [i] * len(residue_atoms)
            chain_atoms += residue_atoms

        return chain_residue_ids, chain_atoms

    @staticmethod
    def atom_coordinates(atoms: Iterable[Atom]) -> NDArray:
        return np.array([atom.get_coord() for atom in atoms])

    def __iter__(self) -> Iterator[Residue]:
        return iter(self.residues)
