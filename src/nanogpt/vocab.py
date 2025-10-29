import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def load_meta(data_dir: str | Path) -> dict:
    meta_path = Path(data_dir) / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find meta file at {meta_path}")
    with meta_path.open("rb") as fh:
        return pickle.load(fh)


def get_stoi_itos(meta: dict) -> tuple[Dict[str, int], Dict[int, str]]:
    stoi = meta["stoi"]
    itos_raw = meta["itos"]
    if isinstance(itos_raw, dict):
        itos = itos_raw
    else:
        itos = {idx: tok for idx, tok in enumerate(itos_raw)}
    return stoi, itos


def decode_token_ids(
    token_ids: Sequence[int],
    itos: Dict[int, str],
    skip_tokens: Iterable[str] | None = None,
    join: bool = True,
) -> str | List[str]:
    skip = set(skip_tokens or [])
    tokens = [itos[idx] for idx in token_ids if idx in itos and itos[idx] not in skip]
    return " ".join(tokens) if join else tokens
