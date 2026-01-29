"""Prepare crystal structure tokenizer datasets for nanoGPT training.
Example usage:
    python data/cst/prepare.py --input data/cst/crystal_structure_embeddings_tokens.parquet --embedding-key structure_embedding --structure-key structure_tokens --structure-format tokens --output-dir data/cst/out
"""

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from structure_tokenizer.config import DEFAULT_CONFIG, TokenizerConfig, BinConfig
from structure_tokenizer.quant import from_bin
from structure_tokenizer.tokenizer import encode_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert crystal structure data to nanoGPT-ready numpy arrays."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Pickle/JSON/CSV file containing embeddings and structure references.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="embeddings",
        help="Column/key in the input containing embedding vectors.",
    )
    parser.add_argument(
        "--structure-key",
        type=str,
        default="cif_path",
        help="Column/key describing each structure (CIF path, Structure object, or pre-tokenised sequence).",
    )
    parser.add_argument(
        "--structure-format",
        choices=["cif_path", "structure", "tokens"],
        default="cif_path",
        help="Interpretation of the structure column.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=DEFAULT_CONFIG.symprec,
        help="Symmetry tolerance forwarded to the tokenizer.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=DEFAULT_CONFIG.angle_tolerance,
        help="Angle tolerance forwarded to the tokenizer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write train/val splits (defaults to data/cst).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of samples placed into the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for shuffling before the train/val split.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Maximum allowed token sequence length; <= 0 keeps all samples.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif path.suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    elif path.suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    elif path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Loaded dataset is not a pandas DataFrame.")
    return df.reset_index(drop=True)


def load_structures(
    series: Sequence,
    structure_format: str,
) -> List[Structure]:
    if structure_format == "cif_path":
        return [Structure.from_file(str(Path(p))) for p in series]
    if structure_format == "structure":
        structures: List[Structure] = []
        for item in series:
            if isinstance(item, Structure):
                structures.append(item)
            elif hasattr(item, "as_dict"):
                structures.append(Structure.from_dict(item.as_dict()))
            elif isinstance(item, dict):
                structures.append(Structure.from_dict(item))
            else:
                raise TypeError(f"Unsupported structure object type: {type(item)!r}")
        return structures
    raise ValueError("structure_format='tokens' does not yield Structure objects.")


def load_token_sequences(
    data: Sequence,
    structure_format: str,
    cfg: TokenizerConfig,
) -> tuple[List[List[str]], dict, int]:
    """Return token sequences and tokenizer configuration metadata.

    When ``structure_format`` is not ``\"tokens\"``, structures are re-encoded
    with :func:`encode_to_dict` to guarantee canonical ordering and metadata.
    If the input already contains tokens, we attempt to keep any existing
    configuration payloads and fall back to ``cfg``.
    """
    if structure_format == "tokens":
        sequences: List[List[str]] = []
        tokenizer_config: dict | None = None
        tokenizer_version: int | None = None
        for item in data:
            if isinstance(item, Mapping):
                mapping = dict(item)
                toks = mapping.get("tokens")
                if toks is None:
                    raise KeyError("Token mapping must contain a 'tokens' field.")
                sequences.append(list(toks))
                if tokenizer_config is None:
                    tokenizer_config = mapping.get("config")
                if tokenizer_version is None and "version" in mapping:
                    tokenizer_version = int(mapping["version"])
            elif isinstance(item, (list, tuple, np.ndarray)):
                sequences.append([str(tok) for tok in item])
            elif isinstance(item, str):
                sequences.append(item.split())
            else:
                raise TypeError(
                    "structure_format='tokens' requires sequences, strings, or mappings."
                )
        if tokenizer_config is None:
            tokenizer_config = asdict(cfg)
        if tokenizer_version is None:
            tokenizer_version = 1
        return sequences, tokenizer_config, tokenizer_version

    structures = load_structures(data, structure_format)
    payloads = [encode_to_dict(struct, cfg) for struct in structures]
    sequences = [payload["tokens"] for payload in payloads]
    if payloads:
        tokenizer_config = payloads[0]["config"]
        tokenizer_version = int(payloads[0].get("version", 1))
    else:
        tokenizer_config = asdict(cfg)
        tokenizer_version = 1
    return sequences, tokenizer_config, tokenizer_version


OrdinalMetadata = Dict[str, Dict[str, object]]


def _parse_numeric_suffix(token: str, prefix: str) -> float | None:
    body = token[len(prefix) : -1]  # strip prefix and trailing ']'
    try:
        return float(body)
    except ValueError:
        try:
            return float(int(body, 10))
        except ValueError:
            return None


def _decode_bin_value(prefix: str, bin_idx: int, bins_cfg: Mapping[str, float]) -> float:
    """Return the physical value corresponding to a token bin index."""
    if prefix == "[a:":
        lo, hi, bins = bins_cfg["a_lo"], bins_cfg["a_hi"], bins_cfg["L_bins"]
    elif prefix in {"[b/a:", "[c/a:"}:
        lo, hi, bins = bins_cfg["r_lo"], bins_cfg["r_hi"], bins_cfg["L_bins"]
    elif prefix in {"[alpha:", "[beta:", "[gamma:"}:
        lo, hi, bins = bins_cfg["ang_lo"], bins_cfg["ang_hi"], bins_cfg["A_bins"]
    elif prefix in {"[fx:", "[fy:", "[fz:"}:
        lo, hi, bins = 0.0, 1.0, bins_cfg["F_bins"]
    else:
        return float(bin_idx)
    return from_bin(bin_idx, lo, hi, bins)


def build_vocab(
    sequences: Iterable[Sequence[str]],
    tokenizer_config: Mapping[str, object],
) -> tuple[List[str], dict[str, int], OrdinalMetadata]:
    ordinal_prefixes = [
        "[a:",
        "[b/a:",
        "[c/a:",
        "[alpha:",
        "[beta:",
        "[gamma:",
        "[fx:",
        "[fy:",
        "[fz:",
    ]
    ordinal_tokens: Dict[str, Dict[str, float]] = {
        prefix: {} for prefix in ordinal_prefixes
    }
    other_tokens: set[str] = set()

    if isinstance(tokenizer_config, Mapping):
        bins_cfg = tokenizer_config.get("bins")
    else:
        bins_cfg = getattr(tokenizer_config, "bins", None)
    if bins_cfg is None:
        bins_cfg = asdict(DEFAULT_CONFIG.bins)
    elif isinstance(bins_cfg, BinConfig):
        bins_cfg = asdict(bins_cfg)
    else:
        bins_cfg = dict(bins_cfg)

    for seq in sequences:
        for token in seq:
            matched = False
            for prefix in ordinal_prefixes:
                if token.startswith(prefix) and token.endswith("]"):
                    value = _parse_numeric_suffix(token, prefix)
                    if value is not None:
                        try:
                            bin_idx = int(round(value))
                        except (TypeError, ValueError):
                            bin_idx = None
                        ordinal_tokens[prefix][token] = (
                            _decode_bin_value(prefix, bin_idx, bins_cfg)
                            if bin_idx is not None
                            else value
                        )
                        matched = True
                    break
            if not matched:
                other_tokens.add(token)

    tokens: List[str] = ["<PAD>"]
    # Ensure <PAD> isn't duplicated
    other_tokens.discard("<PAD>")
    tokens.extend(sorted(other_tokens))

    ordinal_metadata: OrdinalMetadata = {}
    for prefix in ordinal_prefixes:
        entries = ordinal_tokens[prefix]
        if not entries:
            continue
        sorted_items = sorted(entries.items(), key=lambda kv: kv[1])
        start = len(tokens)
        tokens.extend(token for token, _ in sorted_items)
        size = len(sorted_items)
        name = prefix[1:-1]  # strip leading '[' and trailing ':'
        ordinal_metadata[name] = {
            "start": start,
            "size": size,
            "values": [value for _, value in sorted_items],
        }

    stoi = {tok: idx for idx, tok in enumerate(tokens)}
    return tokens, stoi, ordinal_metadata


def encode_sequences(
    sequences: Iterable[Sequence[str]], stoi: dict[str, int]
) -> List[List[int]]:
    encoded: List[List[int]] = []
    for seq in sequences:
        try:
            encoded.append([stoi[tok] for tok in seq])
        except KeyError as exc:
            raise KeyError(f"Token {exc.args[0]!r} missing from vocabulary.") from exc
    return encoded


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input)
    if args.embedding_key not in df or args.structure_key not in df:
        raise KeyError(
            f"Input data must contain '{args.embedding_key}' and '{args.structure_key}' columns."
        )

    custom_bins = BinConfig(
        L_bins=128,
        A_bins=128,
        F_bins=1024,
        a_lo=0.6,
        a_hi=31.4,
        ang_lo=55.0,
        ang_hi=125.0,
    )

    tokenizer_cfg = TokenizerConfig(
        symprec=DEFAULT_CONFIG.symprec,
        angle_tolerance=DEFAULT_CONFIG.angle_tolerance,
        bins=custom_bins,
    )

    # tokenizer_cfg = TokenizerConfig(
    #     symprec=args.symprec,
    #     angle_tolerance=args.angle_tolerance,
    #     bins=DEFAULT_CONFIG.bins,
    # )
    embeddings = np.stack(df[args.embedding_key].to_list()).astype(np.float32)
    sequences, tokenizer_config, tokenizer_version = load_token_sequences(
        df[args.structure_key], args.structure_format, tokenizer_cfg
    )

    max_length = args.max_length
    if max_length and max_length > 0:
        keep_mask = np.array([len(seq) <= max_length for seq in sequences], dtype=bool)
        dropped = int(len(sequences) - keep_mask.sum())
        if dropped > 0:
            print(f"Dropped {dropped} samples longer than {max_length} tokens.")
        sequences = [seq for seq, keep in zip(sequences, keep_mask) if keep]
        embeddings = embeddings[keep_mask]
        df = df.iloc[keep_mask].reset_index(drop=True)
        if not sequences:
            raise ValueError("All samples were dropped by the max-length filter.")

    vocab_list, stoi, ordinal_metadata = build_vocab(sequences, tokenizer_config)
    token_ids = encode_sequences(sequences, stoi)

    n = len(token_ids)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n)

    split = int(n * args.train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]

    def subset(idx: np.ndarray) -> dict[str, object]:
        return {
            "embeddings": embeddings[idx],
            "token_ids": [token_ids[i] for i in idx],
        }

    train_data = subset(train_idx)
    val_data = subset(val_idx)

    print(f"Training samples:   {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")

    meta = {
        "vocab_size": len(vocab_list),
        "itos": vocab_list,
        "stoi": stoi,
        "pad_token": "<PAD>",
        "pad_token_id": stoi["<PAD>"],
        "bos_token": "[BOS]" if "[BOS]" in stoi else None,
        "eos_token": "[EOS]" if "[EOS]" in stoi else None,
        "embedding_dim": embeddings.shape[1],
        "tokenizer_config": tokenizer_config,
        "tokenizer_version": tokenizer_version,
        "ordinal_metadata": ordinal_metadata,
    }

    np.save(output_dir / "train.npy", train_data)
    np.save(output_dir / "val.npy", val_data)
    with open(output_dir / "meta.pkl", "wb") as fh:
        pickle.dump(meta, fh)

    print(
        f"Wrote {len(train_idx)} training and {len(val_idx)} validation samples to {output_dir}"
    )
    print("Vocabulary size:", meta["vocab_size"])


if __name__ == "__main__":
    main()
