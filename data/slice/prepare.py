import os
import sys
import numpy as np
import pandas as pd
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from vocab import vocab, encode_slice, stoi, itos

# Load your data
data = pd.read_pickle("/p/project1/hai_solaihack/datasets/slices/embeddings_with_slices")

data = {
    'embeddings': data['embeddings'][:int(len(data['embeddings']))],
    'slices': data['slices'][:int(len(data['slices']))]
}

embedding_dim = len(data['embeddings'][0])
print(f"Embedding dimension: {embedding_dim}")

# Split into train/val sets
n = len(data['embeddings'])
train_emb = np.array(data['embeddings'][:int(n*0.9)].tolist())
val_emb = np.array(data['embeddings'][int(n*0.9):].tolist())

train_slices = data['slices'][:int(n*0.9)]
val_slices = data['slices'][int(n*0.9):]

# Encode slices using custom vocabulary
train_slice_ids = [encode_slice(slice_text) for slice_text in train_slices]
val_slice_ids = [encode_slice(slice_text) for slice_text in val_slices]

# Save embeddings and encoded slices
train_data = {
    'embeddings': train_emb,
    'slice_ids': train_slice_ids
}
val_data = {
    'embeddings': val_emb,
    'slice_ids': val_slice_ids
}

# Save vocabulary info
meta = {
    'vocab_size': len(vocab),
    'stoi': stoi,
    'itos': itos
}

# Save to files
np.save(os.path.join(os.path.dirname(__file__), 'train.npy'), train_data)
np.save(os.path.join(os.path.dirname(__file__), 'val.npy'), val_data)
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Saved to:")
print(f"{os.path.join(os.path.dirname(__file__), 'train.npy')}")
print(f"{os.path.join(os.path.dirname(__file__), 'val.npy')} and meta.pkl")
