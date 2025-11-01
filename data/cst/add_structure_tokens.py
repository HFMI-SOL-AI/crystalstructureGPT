"""
Add tokenised CIF strings to a crystal structure embeddings parquet file.
Usage:
    python add_structure_tokens.py \
        --input crystal_structure_embeddings_20251031_230438.parquet \
        --output crystal_structure_embeddings_tokens.parquet \
        --workers 8 --chunksize 16
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

from structure_tokenizer.config import BinConfig
from structure_tokenizer.tokenizer import DEFAULT_CONFIG, TokenizerConfig, encode

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Issues encountered while parsing CIF",
)


_CUSTOM_BINS = BinConfig(
    L_bins=512,
    A_bins=512,
    F_bins=1024,
    a_lo=0.6,
    a_hi=31.4,
    ang_lo=55.0,
    ang_hi=125.0,
)

_TOKENIZER_CFG = TokenizerConfig(
    symprec=DEFAULT_CONFIG.symprec,
    angle_tolerance=DEFAULT_CONFIG.angle_tolerance,
    bins=_CUSTOM_BINS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach structure_token tokens to a parquet table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input parquet file with structure_cif column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination parquet file for the augmented table.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes to use for tokenisation.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Chunk size for distributing CIF strings across workers.",
    )
    return parser.parse_args()


def cif_to_tokens(cif_text: str) -> Sequence[str]:
    """Convert a CIF string into a sequence of tokenizer tokens."""
    structure = Structure.from_str(cif_text, fmt="cif")
    return encode(structure, _TOKENIZER_CFG)


def tokenize_cifs(
    cif_strings: Sequence[str], workers: int, chunksize: int
) -> list[Sequence[str]]:
    total = len(cif_strings)
    desc = "Tokenising CIFs"
    if workers <= 1:
        return [cif_to_tokens(cif) for cif in tqdm(cif_strings, desc=desc, unit="structure")]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(cif_to_tokens, cif_strings, chunksize=max(1, chunksize))
        return list(tqdm(iterator, total=total, desc=desc, unit="structure"))


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    if "structure_cif" not in df.columns:
        raise KeyError("Input parquet must contain a 'structure_cif' column.")

    df = df.copy()
    cif_strings = df["structure_cif"].tolist()
    tokens = tokenize_cifs(cif_strings, workers=args.workers, chunksize=args.chunksize)
    df["structure_tokens"] = tokens
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
