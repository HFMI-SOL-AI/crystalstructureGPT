"""
Sample crystal-structure tokenizer outputs from a trained nanoGPT model.
Usage example:
python sample.py \
    --data-dir data/cst/out \
    --out-dir out \
    --num-samples 10 \
    --temperature 0.7

"""

from __future__ import annotations

import os
from contextlib import nullcontext

import numpy as np
import torch

from nanogpt.constraints import CrystalConstraintBuilder
from nanogpt.model import GPTConfig, GPT
from nanogpt.vocab import decode_token_ids, get_stoi_itos, load_meta

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = os.path.join('..','out')
num_samples = 5
max_new_tokens = 100  # maximum generated length (tokens)
temperature = 0.2
top_k = 30
seed = 42
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
data_dir = os.path.join('..','data', 'cst', 'out')
# Set to False to disable constrained decoding during sampling.
use_constraints = True
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

meta = load_meta(data_dir)
stoi, itos = get_stoi_itos(meta)
pad_token = meta.get("pad_token")
bos_token = meta.get("bos_token") or "[BOS]"
eos_token = meta.get("eos_token") or "[EOS]"
constraint_builder = CrystalConstraintBuilder.from_meta(meta) if use_constraints else None

if bos_token not in stoi:
    raise KeyError(f"Start token {bos_token!r} is missing from the vocabulary.")

bos_token_id = stoi[bos_token]
eos_token_id = stoi.get(eos_token)
skip_tokens = {tok for tok in (pad_token, bos_token, eos_token) if tok}

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
else:
    raise ValueError("init_from must be 'resume' to use a trained checkpoint.")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)


def _decode(ids: list[int]) -> str:
    tokens = decode_token_ids(ids, itos, skip_tokens=skip_tokens, join=False)
    if isinstance(tokens, list):
        return " ".join(tokens)
    return tokens


def generate_from_embedding(
    embedding: np.ndarray,
    model: GPT,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int | None = None,
) -> str:
    embedding_tensor = torch.tensor(embedding).float().unsqueeze(0).to(device)
    start_tokens = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        with ctx:
            constraint_state = constraint_builder.new_state(start_tokens[0].tolist()) if constraint_builder else None
            constraint_list = [constraint_state] if constraint_state is not None else None
            generated_ids = model.generate(
                idx=start_tokens,
                embeddings=embedding_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token=eos_token_id,
                constraints=constraint_list,
            )

    return _decode(generated_ids[0].tolist())


print("Loading validation data...")
val_data = np.load(os.path.join(data_dir, 'val.npy'), allow_pickle=True).item()
test_embeddings = val_data['embeddings'][:num_samples]
test_tokens = val_data['token_ids'][:num_samples]

print("\nGenerating crystal-structure tokens from embeddings...")
print("=========================================================")

for i, embedding in enumerate(test_embeddings):
    print(f"\nSample {i + 1}:")
    print("------------------")
    generated = generate_from_embedding(
        embedding=embedding,
        model=model, # pyright: ignore
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    print(f"Generated: {generated}")

    if i < len(test_tokens):
        original = _decode(test_tokens[i])
        print(f"Original:  {original}")
    print("------------------")


def generate_from_specific_embedding(embedding: np.ndarray) -> str:
    """Helper for interactive use."""
    return generate_from_embedding(
        embedding=embedding,
        model=model, # pyright: ignore
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
