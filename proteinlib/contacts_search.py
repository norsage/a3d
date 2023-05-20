from __future__ import annotations

from itertools import chain
from typing import NamedTuple

from numpy.typing import NDArray
from sklearn.neighbors import BallTree
from typing_extensions import TypeAlias

from .common_types import ContactResidueIds
from .protein_chain import ProteinChain

Interface: TypeAlias = tuple[ContactResidueIds, ContactResidueIds]


class ContactsSearchTree(NamedTuple):
    tree: BallTree
    residue_indices: tuple[int, ...]

    @staticmethod
    def from_protein_chain(protein_chain: ProteinChain) -> ContactsSearchTree:
        residue_ids, atoms = protein_chain.atoms
        return ContactsSearchTree(
            tree=BallTree(protein_chain.atom_coordinates(atoms)),
            residue_indices=tuple(residue_ids),
        )

    def get_atomic_interactions(self, threshold: float) -> list[tuple[int, int]]:
        self_contacts = self.tree.query_radius(self.tree.data, r=threshold)
        edges: list[tuple[int, int]] = []
        for i, atom_cotacts in enumerate(self_contacts):
            edges.extend(((i, j) for j in atom_cotacts))

        return edges

    def get_interface(self, antibody_chain: ProteinChain, threshold: float) -> Interface:
        chain_residue_ids, chain_atoms = antibody_chain.atoms

        neighbors: list[NDArray] = self.tree.query_radius(
            antibody_chain.atom_coordinates(chain_atoms), r=threshold
        )
        chain_contact_residues: list[int] = []

        antigen_contact_atoms = set(chain(*neighbors))
        antigen_contact_residues = sorted(
            set([self.residue_indices[i] for i in antigen_contact_atoms])
        )

        for chain_atom_id, antigen_atom_ids in enumerate(neighbors):
            if len(antigen_atom_ids) > 0:
                chain_contact_residues.append(chain_residue_ids[chain_atom_id])

        return tuple(sorted(set(chain_contact_residues))), tuple(antigen_contact_residues)
