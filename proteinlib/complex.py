from enum import Enum
from typing import NamedTuple, Sequence

import numpy as np

from .antibody_chain import AntibodyChain
from .contacts_search import ContactsSearchTree, Interface
from .protein_chain import ProteinChain


class Task(Enum):
    GENERATION_FULL = "generation"
    GENERATION_CDRH3 = "generation_cdrh3"
    GENERATION_CDR = "generation_cdr"
    INFILLING = "infilling"


class ComplexSummary(NamedTuple):
    uid: str
    # heavy chain
    heavy_chain_sequence: str
    heavy_chain_regions: Sequence[int]
    heavy_chain_contacts: Sequence[int]
    # light chain
    light_chain_sequence: str
    light_chain_regions: Sequence[int]
    light_chain_contacts: Sequence[int]
    # antigen
    antigen_sequence: str
    antigen_heavy_contacts: Sequence[int]
    antigen_light_contacts: Sequence[int]

    @property
    def paratope_ids(self) -> list[int]:
        return list(self.heavy_chain_contacts) + [
            x + len(self.heavy_chain_sequence) + 1 for x in self.light_chain_contacts
        ]

    @property
    def linear_epitope(self) -> str:
        if self.antigen_heavy_contacts or self.antigen_light_contacts:
            start, end = self.linearize_contacts(
                sorted(list(self.antigen_heavy_contacts) + list(self.antigen_light_contacts))
            )
            return self.antigen_sequence[start : end + 1]
        return ""

    @staticmethod
    def linearize_contacts(contacts: list[int], max_distance: int = 7) -> tuple[int, int]:
        current_cluster_start: int = contacts[0]
        current_cluster_end: int = current_cluster_start
        clusters: list[tuple[int, int]] = []
        for i, x in enumerate(contacts[1:], 1):
            if x - current_cluster_end > max_distance:
                # create new cluster
                clusters.append((current_cluster_start, current_cluster_end))
                current_cluster_start = x
                current_cluster_end = x
            else:
                current_cluster_end = x

        clusters.append((current_cluster_start, current_cluster_end))
        cluster_sizes = [end - start + 1 for start, end in clusters]
        return clusters[np.argmax(cluster_sizes)]


class AbAgComplex:
    uid: str
    heavy_chain: AntibodyChain
    light_chain: AntibodyChain
    antigen_chain: ProteinChain
    _search_tree: ContactsSearchTree

    def __init__(
        self,
        uid: str,
        heavy_chain: AntibodyChain,
        light_chain: AntibodyChain,
        antigen_chain: ProteinChain,
    ):
        self.uid = uid
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.antigen_chain = antigen_chain
        self._search_tree = ContactsSearchTree.from_protein_chain(self.antigen_chain)

    def light_interface(self, threshold: float = 5.0) -> Interface:
        return self._search_tree.get_interface(self.light_chain, threshold=threshold)

    def heavy_interface(self, threshold: float = 5.0) -> Interface:
        return self._search_tree.get_interface(self.heavy_chain, threshold=threshold)

    # stats
    def summary(self) -> ComplexSummary:
        # heavy chain + contacts
        heavy_chain_contacts, antigen_contacts_h = self.heavy_interface()
        # light chain + contacts
        light_chain_contacts, antigen_contacts_l = self.light_interface()

        # epitope
        return ComplexSummary(
            uid=self.uid,
            # heavy chain
            heavy_chain_sequence=str(self.heavy_chain.residues.get_sequence()),
            heavy_chain_regions=self.heavy_chain.region_boundaries,
            heavy_chain_contacts=heavy_chain_contacts,
            # light chain
            light_chain_sequence=str(self.light_chain.residues.get_sequence()),
            light_chain_regions=self.light_chain.region_boundaries,
            light_chain_contacts=light_chain_contacts,
            # antigen
            antigen_sequence=str(self.antigen_chain.residues.get_sequence()),
            antigen_heavy_contacts=antigen_contacts_h,
            antigen_light_contacts=antigen_contacts_l,
        )
