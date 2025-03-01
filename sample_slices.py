"""
Sample slices from a trained model using embeddings
"""
import os
from contextlib import nullcontext
import torch
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import GPTConfig, GPT
from vocab import decode_slice, stoi

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out'
num_samples = 10
max_new_tokens = 100  # Length of generated slice
temperature = 0.3     # Lower for more focused samples
top_k = 30           # Restrict to top K most likely tokens
seed = 42
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

def generate_from_embedding(embedding, model, max_tokens=100, temperature=0.7, top_k=40):
    embedding_tensor = torch.tensor(embedding).float().unsqueeze(0).to(device)
    start_token_id = stoi['<START>']
    start_tokens = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        with ctx:
            tokens = model.generate(
                idx=start_tokens,
                embeddings=embedding_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token=stoi['<EOS>']
            )

    generated_slice = decode_slice(tokens[0].tolist())
    return generated_slice

print("Loading validation data...")
val_data = np.load('data/slice/val.npy', allow_pickle=True).item()
test_embeddings = val_data['embeddings'][:num_samples]
test_slices = val_data['slice_ids'][:num_samples]

print("\nGenerating slices from embeddings...")
print("=====================================")

for i, embedding in enumerate(test_embeddings):
    print(f"\nSample {i+1}:")
    print("---------------")
    generated_slice = generate_from_embedding(
        embedding=embedding,
        model=model,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    print(f"Generated: {generated_slice}")

    if i < len(test_slices):
        true_slice = decode_slice(test_slices[i])
        print(f"Original: {true_slice}")
    print("---------------")

# print("\nDone! You can use generate_from_specific_embedding(embedding) for custom embeddings.")

def generate_from_specific_embedding(embedding):
    """Generate slice from a specific embedding"""
    return generate_from_embedding(
        embedding=embedding,
        model=model,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

# # If running as main script, show some examples of how to use this
# if __name__ == "__main__":
#     print("\nTo generate from a specific embedding, use:")
#     print("from sample_slices import generate_from_specific_embedding")
#     print("slice_text = generate_from_specific_embedding(your_embedding)")
