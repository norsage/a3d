import json
import logging
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pyrootutils
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from proteinlib.sabdab import AbAgComplex, parse_sabdab


def save_complex(ab_complex: AbAgComplex, savedir: Path) -> None:
    summary = ab_complex.summary()
    with open(savedir / f"{summary.uid}.json", "w") as f:
        json.dump(summary._asdict(), f)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-i", "--chothia_subdir", type=str, default="data/chothia")
    argparser.add_argument("-s", "--summary_csv", type=str, default="data/summary.tsv")
    argparser.add_argument("-o", "--savedir", type=str, default="data/processed")
    argparser.add_argument("-l", "--logfile", type=str, default="process_log.txt")
    argparser.add_argument("-w", "--num_workers", type=int, default=4)
    args = argparser.parse_args()

    logging.basicConfig(
        filename=args.logfile,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    logger = logging.getLogger(__name__)

    chothia_subdir = Path(args.chothia_subdir)
    summary_csv = Path(args.summary_csv)
    savedir = Path(args.savedir)

    savedir.mkdir(parents=True, exist_ok=True)
    complex_gen = parse_sabdab(chothia_subdir, summary_csv, logger)

    with Pool(args.num_workers) as pool:
        list(
            pool.imap_unordered(
                partial(save_complex, savedir=savedir),
                tqdm(complex_gen),
            )
        )
