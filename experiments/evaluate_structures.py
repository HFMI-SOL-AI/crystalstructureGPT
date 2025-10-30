"""
Evaluate reconstruction quality of generated crystal structures.

Loads a trained checkpoint, generates token sequences conditioned on
validation embeddings, decodes them back to Pymatgen structures, and
compares against the ground-truth structures.
Usage example:
python evaluate_structures.py \
    --data-dir data/cst/out \
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

import tqdm
import numpy as np
import torch
from pymatgen.core import Structure

from nanogpt.constraints import CrystalConstraintBuilder
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
    composition_match: bool


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
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="Disable constrained decoding and fall back to unconstrained sampling.",
    )
    parser.add_argument(
        "--force-sg",
        action="store_true",
        help="Seed generation with the ground-truth spacegroup token to assess upper-bound performance.",
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
    constraint_builder = None if args.no_constraints else CrystalConstraintBuilder.from_meta(meta)
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

    # for idx in range(num_samples):
    for idx in tqdm.tqdm(range(num_samples), desc="Evaluating samples"):
        cond = torch.tensor(embeddings[idx]).float().unsqueeze(0).to(args.device)

        if args.force_sg:
            prefix_ids: list[int] = []
            seen_sg = False
            for token_id in token_ids[idx]:
                token_int = int(token_id)
                token = itos.get(token_int)
                if token is None:
                    continue
                prefix_ids.append(token_int)
                if token.startswith("[SG:"):
                    seen_sg = True
                    break
            if not prefix_ids or not itos.get(prefix_ids[0], "").startswith("[BOS]"):
                prefix_ids.insert(0, stoi[bos_token])
            if not seen_sg:
                # fallback to BOS only if SG token missing
                prefix_ids = [stoi[bos_token]]
        else:
            prefix_ids = [stoi[bos_token]]

        start = torch.tensor([prefix_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            generated = model.generate(
                idx=start,
                embeddings=cond,
                max_new_tokens=len(token_ids[idx]),
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token=eos_token_id,
                constraints=[constraint_builder.new_state(start[0].tolist())] if constraint_builder else None,
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

        composition_match = False

        if gen_struct is not None and gt_struct is not None:
            try:
                sg_match = gen_struct.get_space_group_info()[0] == gt_struct.get_space_group_info()[0]
            except Exception:
                sg_match = False
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

            try:
                composition_match = (
                    gen_struct.composition.reduced_formula
                    == gt_struct.composition.reduced_formula
                )
            except Exception:
                composition_match = False
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
                composition_match=composition_match,
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
    comp = sum(r.composition_match for r in results)
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
    print(f"Composition match:      {comp / total:.2%}")

    if valid_total:
        print("--- Valid decodes only ---")
        print(f"Valid samples:          {valid_total}")
        print(f"Exact structural match: {valid_exact / valid_total:.2%}")
        print(f"Spacegroup match:       {valid_sg / valid_total:.2%}")
        print(f"Wyckoff sequence match: {valid_wy / valid_total:.2%}")
        print(f"Lattice RMSE (mean):    {valid_lattice.mean():.4f}")
        if valid_frac:
            print(f"Fractional RMSE (mean): {np.mean(valid_frac):.4f}")
        valid_comp = sum(r.composition_match for r in valid_results)
        print(f"Composition match:      {valid_comp / valid_total:.2%}")


if __name__ == "__main__":
    main()
