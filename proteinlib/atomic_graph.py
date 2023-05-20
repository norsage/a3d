from typing import NamedTuple

from numpy.typing import NDArray

from .contacts_search import ContactsSearchTree
from .protein_chain import ProteinChain

ELEMENT_TO_ID = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "P": 7,
    "I": 8,
    "F": 9,
    "others": 10,
}
# ATOM_NAME_TO_ID

# C
# CA
# CB
# CD
# CD1
# CD2
# CE
# CE1
# CE2
# CE3
# CG
# CG1
# CG2
# CH2
# CZ
# CZ2
# CZ3
# N
# ND1
# ND2
# NE
# NE1
# NE2
# NH1
# NH2
# NZ
# O
# OD1
# OD2
# OE1
# OE2
# OG
# OG1
# OH
# SD
# SG


class AtomicGraph(NamedTuple):
    atom_names: list[int]
    residue_ids: list[int]
    coordinates: NDArray
    interactions: list[tuple[int, int]]


def extract_atomic_graph(protein_chain: ProteinChain, threshold: float) -> AtomicGraph:
    residue_ids, atoms = protein_chain.atoms
    coordinates = protein_chain.atom_coordinates(atoms)

    search_tree = ContactsSearchTree.from_protein_chain(protein_chain)
    edges = search_tree.get_atomic_interactions(threshold)
    return AtomicGraph(
        atom_names=[ELEMENT_TO_ID.get(atom.element, 10) for atom in atoms],
        residue_ids=residue_ids,
        coordinates=coordinates,
        interactions=edges,
    )
