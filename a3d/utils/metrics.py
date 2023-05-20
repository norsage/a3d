from itertools import chain
from typing import NamedTuple, Sequence

from Bio.Align import Alignment, PairwiseAligner, substitution_matrices


class AlignmentMetrics(NamedTuple):
    score: float
    fr_score: float
    cdr_score: float
    paratope_score: float
    alignment: str

    def __repr__(self) -> str:
        return (
            f"{self.alignment}\n"
            f"FR Score: {self.fr_score:.4f}, CDR Score: {self.cdr_score:.4f}, "
            f"Paratope Score: {self.paratope_score:.4f}"
        )


def get_metrics(
    generated: str, reference: str, regions: Sequence[int], contacts: Sequence[int]
) -> Alignment:
    alignment = align(generated, reference, use_blosum=False)
    return AlignmentMetrics(
        score=alignment.score,
        fr_score=fr_score(alignment, regions),
        cdr_score=cdr_score(alignment, regions),
        paratope_score=paratope_score(alignment, contacts),
        alignment=format_alignment_with_regions_and_contacts(alignment, regions, contacts),
    )


def align(generated: str, target: str, use_blosum: bool = False) -> Alignment:
    aligner = PairwiseAligner()
    if use_blosum:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    else:
        aligner.mismatch_score = 0.5  # prefer mismatch over gap
    return aligner.align(target, generated)[0]


def get_aligned_positions(
    reference_aligned_indices: Sequence[int], reference_positions: Sequence[int]
) -> list[int]:
    j = 0
    result: list[int] = []
    for k in reference_positions:
        # this line will raise StopIteration if k is not in reference_aligned_indices
        start = (
            0
            if k == 0
            else next(
                filter(
                    lambda i: reference_aligned_indices[i] == k,
                    range(j, len(reference_aligned_indices)),
                )
            )
        )
        try:
            end = next(
                filter(
                    lambda i: reference_aligned_indices[i] == k + 1,
                    range(start + 1, len(reference_aligned_indices)),
                )
            )
        except StopIteration:
            end = len(reference_aligned_indices)
        result.extend(range(start, end))
        j = end

    return result


def calculate_match_ratio(alignment: Alignment, target_positions: Sequence[int]) -> float:
    if not target_positions:
        return 0.0
    # assume that reference sequence goes first
    reference_aligned_indices = get_aligned_positions(alignment.indices[0], target_positions)
    matches = 0
    for i in reference_aligned_indices:
        matches += int(alignment[0][i] == alignment[1][i])

    return (matches) / len(reference_aligned_indices)


def get_fr_regions_positions(regions: Sequence[int]) -> list[int]:
    n = len(regions)
    return list(
        chain(
            range(regions[0], regions[1]) if n >= 2 else [],
            range(regions[2], regions[3]) if n >= 4 else [],
            range(regions[4], regions[5]) if n >= 6 else [],
            range(regions[6], regions[7]) if n == 8 else [],
        )
    )


def get_cdr_regions_positions(regions: Sequence[int]) -> list[int]:
    n = len(regions)
    return list(
        chain(
            range(regions[1], regions[2]) if n >= 3 else [],
            range(regions[3], regions[4]) if n >= 5 else [],
            range(regions[5], regions[6]) if n >= 7 else [],
        )
    )


def get_cdr3_region_positions(regions: Sequence[int]) -> list[int]:
    return list(range(regions[5], regions[6])) if len(regions) >= 7 else []


def paratope_score(alignment: Alignment, paratope_ids: Sequence[int]) -> float:
    if not paratope_ids:
        return 0.0
    return calculate_match_ratio(alignment, paratope_ids)


def fr_score(alignment: Alignment, regions: Sequence[int]) -> float:
    return calculate_match_ratio(alignment, get_fr_regions_positions(regions))


def cdr_score(alignment: Alignment, regions: Sequence[int]) -> float:
    return calculate_match_ratio(alignment, get_cdr_regions_positions(regions))


def cdr3_score(alignment: Alignment, regions: Sequence[int]) -> float:
    return calculate_match_ratio(alignment, get_cdr3_region_positions(regions))


def format_alignment(alignment: Alignment) -> str:
    alignment_string: list[str] = []
    for ref_aa, query_aa in zip(*alignment):
        if ref_aa == query_aa:
            alignment_string.append("|")
        elif ref_aa == "-" or query_aa == "-":
            alignment_string.append(" ")
        else:
            alignment_string.append(".")

    return "\n".join((alignment[0], "".join(alignment_string), alignment[1]))


def format_alignment_with_regions(alignment: Alignment, regions: Sequence[int]) -> str:
    frs = set(get_aligned_positions(alignment.indices[0], get_fr_regions_positions(regions)))
    alignment_string: list[str] = []
    region_string: list[str] = []
    for i, (ref_aa, query_aa) in enumerate(zip(*alignment)):
        if ref_aa == query_aa:
            alignment_string.append("|")
        elif ref_aa == "-" or query_aa == "-":
            alignment_string.append(" ")
        else:
            alignment_string.append(".")

        if i in frs:
            region_string.append(" ")
        else:
            region_string.append("^")

    return "\n".join(
        (alignment[0], "".join(alignment_string), alignment[1], "".join(region_string))
    )


def format_alignment_with_regions_and_contacts(
    alignment: Alignment, regions: Sequence[int], contacts: Sequence[int]
) -> str:
    cdrs = set(get_aligned_positions(alignment.indices[0], get_cdr_regions_positions(regions)))
    paratope = set(get_aligned_positions(alignment.indices[0], contacts))
    alignment_string: list[str] = []
    region_string: list[str] = []
    for i, (ref_aa, query_aa) in enumerate(zip(*alignment)):
        alignment_string.append(
            "|" if ref_aa == query_aa else " " if ref_aa == "-" or query_aa == "-" else "."
        )
        region_string.append(
            "X"
            if i in paratope and i in cdrs
            else "."
            if i in cdrs
            else "*"
            if i in paratope
            else " "
        )

    return "\n".join(
        ("".join(region_string), alignment[0], "".join(alignment_string), alignment[1])
    )
