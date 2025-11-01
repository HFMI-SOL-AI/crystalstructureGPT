import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


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
    return parser.parse_args()


def main() -> None:
    # args = parse_args()
    input = Path("./crystal_structure_embeddings_tokens.parquet")
    tqdm.pandas()
    df = pd.read_parquet(input)
    print(df.columns)
    print(df.shape)
    print(df.head(1))
    # remove any row that contains a NaN value
    df = df.dropna()
    print(df.shape)

    print(df.head(1)["structure_id"].values[0])
    print(df.head(1)["structure_tokens"].values[0])


if __name__ == "__main__":
    main()
