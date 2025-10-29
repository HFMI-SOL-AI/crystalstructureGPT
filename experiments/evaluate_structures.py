"""
Evaluate reconstruction quality of generated crystal structures.

Loads a trained checkpoint, generates token sequences conditioned on
validation embeddings, decodes them back to Pymatgen structures, and
compares against the ground-truth structures.
Usage example:
python -m src.nanogpt.evaluate_structures \
    --data-dir data/cst/test_out \
    --checkpoint out/ckpt.pt \
    --split val \
    --limit 100 \
    --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pymatgen.core import Structure

from nanogpt.model import GPT, GPTConfig
from nanogpt.vocab import get_stoi_itos, load_meta

try:
    from structure_tokenizer.tokenizer import decode
    from structure_tokenizer.matcher import tokens_match
except ImportError as exc:  # pragma: no cover - dependency missing
    raise RuntimeError(
        "structure_tokenizer package is required for evaluation."
    ) from exc


@dataclass
class EvalResult:
    embedding_index: int
    exact_match: bool
    spacegroup_match: bool
    wyckoff_match: bool
    lattice_rmse: float
    frac_rmse: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark conditioned sampling by structural reconstruction."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cst/test_out"),
        help="Directory containing train.npy/val.npy/meta.pkl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/ckpt.pt"),
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate (use <= 0 for all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k sampling restriction.",
    )
    return parser.parse_args()


def _rmse(a: Iterable[float], b: Iterable[float]) -> float:
    vec_a = np.array(list(a), dtype=np.float64)
    vec_b = np.array(list(b), dtype=np.float64)
    return float(np.sqrt(np.mean((vec_a - vec_b) ** 2)))


def _structure_from_tokens(tokens: list[str]) -> Structure:
    try:
        _, conventional = decode(tokens)
    except Exception:
        return None  # type: ignore[return-value]
    return conventional


def evaluate(args: argparse.Namespace) -> list[EvalResult]:
    data_dir = args.data_dir
    cache = np.load(data_dir / f"{args.split}.npy", allow_pickle=True).item()

    meta = load_meta(data_dir)
    stoi, itos = get_stoi_itos(meta)
    bos_token = meta.get("bos_token") or "[BOS]"
    eos_token_id = stoi.get(meta.get("eos_token") or "[EOS]")
    eos_token = meta.get("eos_token") or "[EOS]"

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(args.device)

    embeddings = cache["embeddings"]
    token_ids = cache["token_ids"]
    num_samples = len(token_ids) if args.limit <= 0 else min(args.limit, len(token_ids))

    results: list[EvalResult] = []

    for idx in range(num_samples):
        cond = torch.tensor(embeddings[idx]).float().unsqueeze(0).to(args.device)
        start = torch.tensor([[stoi[bos_token]]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            generated = model.generate(
                idx=start,
                embeddings=cond,
                max_new_tokens=len(token_ids[idx]),
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token=eos_token_id,
            )[0].tolist()

        gt_ids = token_ids[idx]

        def _ids_to_tokens(sequence: Iterable[int]) -> list[str]:
            tokens: list[str] = []
            for token_id in sequence:
                token = itos.get(token_id)
                if token is None:
                    continue
                tokens.append(token)
                if token == eos_token:
                    break
            return tokens

        generated_tokens = _ids_to_tokens(generated)
        reference_tokens = _ids_to_tokens(gt_ids)
        try:
            exact = tokens_match(generated_tokens, reference_tokens)
        except Exception:
            exact = False

        gen_struct = _structure_from_tokens(generated_tokens)
        gt_struct = _structure_from_tokens(reference_tokens)

        if gen_struct is not None and gt_struct is not None:
            sg_match = gen_struct.get_space_group_info()[0] == gt_struct.get_space_group_info()[0]
            wyckoff_match = [
                tok for tok in generated_tokens if tok.startswith("[WY:")
            ] == [
                tok for tok in reference_tokens if tok.startswith("[WY:")
            ]
            lattice_rmse = _rmse(gen_struct.lattice.parameters, gt_struct.lattice.parameters)

            try:
                frac_gen = np.array(gen_struct.frac_coords)
                frac_gt = np.array(gt_struct.frac_coords)
                if frac_gen.shape == frac_gt.shape:
                    frac_rmse = float(np.sqrt(np.mean((frac_gen - frac_gt) ** 2)))
                else:
                    frac_rmse = None
            except Exception:
                frac_rmse = None
        else:
            sg_match = False
            wyckoff_match = False
            lattice_rmse = float("nan")
            frac_rmse = None

        results.append(
            EvalResult(
                embedding_index=idx,
                exact_match=exact,
                spacegroup_match=sg_match,
                wyckoff_match=wyckoff_match,
                lattice_rmse=lattice_rmse,
                frac_rmse=frac_rmse,
            )
        )

    return results


def main() -> None:
    args = parse_args()
    results = evaluate(args)

    total = len(results)
    exact = sum(r.exact_match for r in results)
    sg = sum(r.spacegroup_match for r in results)
    wy = sum(r.wyckoff_match for r in results)
    lattice_errors = np.array([r.lattice_rmse for r in results], dtype=float)
    frac_errors = [r.frac_rmse for r in results if r.frac_rmse is not None]

    valid_mask = ~np.isnan(lattice_errors)
    valid_results = [r for r, ok in zip(results, valid_mask) if ok]
    valid_total = len(valid_results)
    valid_exact = sum(r.exact_match for r in valid_results)
    valid_sg = sum(r.spacegroup_match for r in valid_results)
    valid_wy = sum(r.wyckoff_match for r in valid_results)
    valid_lattice = lattice_errors[valid_mask]
    valid_frac = [r.frac_rmse for r in valid_results if r.frac_rmse is not None]

    print(f"Samples evaluated: {total}")
    print(f"Exact structural match: {exact / total:.2%}")
    print(f"Spacegroup match:       {sg / total:.2%}")
    print(f"Wyckoff sequence match: {wy / total:.2%}")
    print(f"Lattice RMSE (mean):    {np.nanmean(lattice_errors):.4f}")
    if frac_errors:
        print(f"Fractional RMSE (mean): {np.mean(frac_errors):.4f}")

    if valid_total:
        print("--- Valid decodes only ---")
        print(f"Valid samples:          {valid_total}")
        print(f"Exact structural match: {valid_exact / valid_total:.2%}")
        print(f"Spacegroup match:       {valid_sg / valid_total:.2%}")
        print(f"Wyckoff sequence match: {valid_wy / valid_total:.2%}")
        print(f"Lattice RMSE (mean):    {valid_lattice.mean():.4f}")
        if valid_frac:
            print(f"Fractional RMSE (mean): {np.mean(valid_frac):.4f}")


if __name__ == "__main__":
    main()
