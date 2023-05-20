from enum import IntEnum, auto
from typing import NamedTuple

from typing_extensions import TypeAlias

AntibodyChainRegionBorders: TypeAlias = tuple[int, int, int, int, int, int, int]


class SchemeRegionBorders(NamedTuple):
    heavy_chain: AntibodyChainRegionBorders
    light_chain: AntibodyChainRegionBorders


class NumberingScheme(IntEnum):
    CHOTHIA = auto()
    IMGT = auto()
    KABAT = auto()
    NORTH = auto()


# source: `abnumber` package
SCHEME_BORDERS_DICT = {
    # start pos. :  CDR1, FR2, CDR2, FR3, CDR3, FR4
    NumberingScheme.IMGT: SchemeRegionBorders(
        heavy_chain=(27, 39, 56, 66, 105, 118, 129),
        light_chain=(27, 39, 56, 66, 105, 118, 129),
    ),
    NumberingScheme.CHOTHIA: SchemeRegionBorders(
        heavy_chain=(26, 33, 52, 57, 95, 103, 114),
        light_chain=(24, 35, 50, 57, 89, 98, 108),
    ),
    NumberingScheme.KABAT: SchemeRegionBorders(
        heavy_chain=(31, 36, 50, 66, 95, 103, 114),
        light_chain=(24, 35, 50, 57, 89, 98, 108),
    ),
    NumberingScheme.NORTH: SchemeRegionBorders(
        heavy_chain=(23, 36, 50, 59, 93, 103, 114),
        light_chain=(24, 35, 49, 57, 89, 98, 108),
    ),
}
