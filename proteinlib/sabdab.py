import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Generator, Iterable, cast

import pandas as pd
from tqdm import tqdm

from .complex import AbAgComplex
from .pdb_parser import NumberingScheme, parse_complex


def parse_sabdab(  # noqa: C901
    chothia_subdir: Path, summary_csv: Path, logger: logging.Logger | None = None
) -> Generator[AbAgComplex, None, None]:
    if logger is None:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logger = logging.getLogger(__name__)
    # only protein and peptide antigens
    df = (
        pd.read_csv(
            summary_csv,
            sep="\t",
            usecols=["pdb", "Hchain", "Lchain", "antigen_chain", "antigen_type"],
        )
        .query("antigen_type in ('protein', 'peptide')")
        .dropna()
        .drop_duplicates()
    )
    print(f"Summary records: {df.shape[0]}")

    for _, row in cast(Iterable[tuple[int, pd.Series]], df.iterrows()):
        uid = f'{row["pdb"]}_{row["Hchain"]}+{row["Lchain"]}-{row["antigen_chain"]}'
        try:
            # row = cast(pd.Series, row)
            antigen_chains = tuple(map(lambda s: s.strip(), str(row["antigen_chain"]).split(" | ")))
            # skip complexes with multimers
            if len(antigen_chains) > 1:
                logging.info(f"Skipping {uid}: multimer antigen")
                continue

            complex = parse_complex(
                pdb=chothia_subdir / f"{row['pdb']}.pdb",
                heavy_chain_id=str(row["Hchain"]),
                light_chain_id=str(row["Lchain"]),
                antigen_chain_ids=antigen_chains,
                scheme=NumberingScheme.CHOTHIA,
            )
            yield complex
        # except (PDBParserException, PolypeptideExtractionException) as err:
        #     logging.warning(f"In complex {uid}: {traceback.format_exception(err)}")

        except Exception as err:
            logging.warning(f"In complex {uid}: {traceback.format_exception(err)}")
            continue
