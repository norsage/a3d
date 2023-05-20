from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import NamedTuple

from .antibody_numbering import SCHEME_BORDERS_DICT, NumberingScheme
from .common_types import ContactResidueIds
from .protein_chain import Polypeptide, ProteinChain


class AntibodyRegions(NamedTuple):
    fr1: str = ""
    cdr1: str = ""
    fr2: str = ""
    cdr2: str = ""
    fr3: str = ""
    cdr3: str = ""
    fr4: str = ""


class AntibodyRegion(IntEnum):
    FR1 = 0
    CDR1 = 1
    FR2 = 2
    CDR2 = 3
    FR3 = 4
    CDR3 = 5
    FR4 = 6


@dataclass
class AntibodyChain(ProteinChain):
    is_heavy: bool
    scheme: NumberingScheme = NumberingScheme.CHOTHIA

    def __post_init__(self) -> None:
        # leave only Fv region
        self.residues = Polypeptide(
            self.residues[self.region_boundaries[0] : self.region_boundaries[-1]]
        )

    @property
    def numbering_borders(self) -> tuple[int, ...]:
        return (1,) + (
            SCHEME_BORDERS_DICT[self.scheme].heavy_chain
            if self.is_heavy
            else SCHEME_BORDERS_DICT[self.scheme].light_chain
        )

    @cached_property
    def region_boundaries(self) -> tuple[int, ...]:
        # TODO: warn about missing residues
        region_boundaries: list[int] = []
        k = 0
        for i, res in enumerate(self.residues):
            if res.id[1] >= self.numbering_borders[k]:
                region_boundaries.append(i)
                k += 1
                if k >= len(self.numbering_borders):
                    break

        if k < len(self.numbering_borders):
            region_boundaries.append(len(self.residues))

        return tuple(region_boundaries)

    @cached_property
    def regions(self) -> AntibodyRegions:
        return AntibodyRegions(
            *(
                self.linear_segment_sequence(
                    self.region_boundaries[i], self.region_boundaries[i + 1]
                )
                for i in range(len(self.region_boundaries) - 1)
            )
        )

    @property
    def cdr_lengths(self) -> tuple[int, int, int]:
        return len(self.regions.cdr1), len(self.regions.cdr2), len(self.regions.cdr3)

    def contacts_regions(self, contacts: ContactResidueIds) -> dict[str, list[int]]:
        annotation: dict[str, list[int]] = defaultdict(list)
        for contact in contacts:
            try:
                region_index = next(
                    (
                        i - 1
                        for i, region_end in enumerate(self.region_boundaries)
                        if region_end > contact
                    )
                )
                annotation[AntibodyRegion(region_index).name].append(contact)
            except StopIteration:
                continue

        return dict(annotation)
