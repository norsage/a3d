from pathlib import Path

from Bio import PDB
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import Polypeptide, is_aa
from Bio.PDB.Structure import Structure

from .antibody_chain import AntibodyChain, NumberingScheme
from .complex import AbAgComplex
from .protein_chain import ProteinChain


class PDBParserException(Exception):
    ...


class PolypeptideExtractionException(Exception):
    ...


def chain_to_polypeptide(protein_chain: Chain, merge: bool = False) -> list[Polypeptide]:
    return [
        Polypeptide(filter(lambda res: is_aa(res, standard=True), protein_chain.get_residues()))
    ]


def parse_complex(
    pdb: Path,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_ids: tuple[str, ...],
    scheme: NumberingScheme = NumberingScheme.CHOTHIA,
) -> AbAgComplex:
    if len(antigen_chain_ids) > 1:
        raise NotImplementedError("Multimer antigens are not supported yet")

    antigen_chain_id = antigen_chain_ids[0]

    uid = f"{pdb.stem}_{heavy_chain_id}+{light_chain_id}-{antigen_chain_id}"

    parser = PDB.PDBParser(QUIET=True)
    try:
        model: Structure = parser.get_structure(uid, pdb)[0]

    except Exception as err:
        raise PDBParserException(str(err))

    heavy_chain_polypeptides = chain_to_polypeptide(model[heavy_chain_id], merge=True)
    light_chain_polypeptides = chain_to_polypeptide(model[light_chain_id], merge=True)
    antigen_chain_polypeptides = chain_to_polypeptide(model[antigen_chain_id], merge=True)

    if len(antigen_chain_polypeptides[0]) == 0:
        raise PolypeptideExtractionException(f"Empty antigen, check chain={antigen_chain_id}")

    antigen_chain = ProteinChain(antigen_chain_polypeptides[0])
    heavy_chain = AntibodyChain(heavy_chain_polypeptides[0], is_heavy=True, scheme=scheme)

    light_chain = AntibodyChain(light_chain_polypeptides[0], is_heavy=False, scheme=scheme)

    return AbAgComplex(
        uid=uid,
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        antigen_chain=antigen_chain,
    )
