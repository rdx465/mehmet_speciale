import torch
from neurovfm.pipelines import load_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kører på: {device}")

print("1) Loader NeuroVFM encoder...")
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")
encoder.model = encoder.model.to(device)

print("2) Bygger dummy data...")
num_patches = 10
patch_dim = 1024

x = torch.randn(num_patches, patch_dim, device=device)
coords = torch.zeros(num_patches, 3, dtype=torch.long, device=device)

dummy_batch = {
    "img": x,
    "coords": coords,
    "series_cu_seqlens": torch.tensor([0, num_patches], dtype=torch.int32, device=device),
    "series_max_len": num_patches,
    "mode": ["ct"],
    "path": ["BrainWindow"]
}

print("3) Kører forward pass...")
with torch.no_grad():
    out = encoder.embed(dummy_batch)

print("\n SUCCESS!")
# Hvis out er en pakke (tuple), tager vi det første element, ellers bare out
features = out[0] if isinstance(out, tuple) else out
print(f"Modellen returnerede: {type(out)} (Længde: {len(out) if isinstance(out, tuple) else 1})")
print(f"Shape af dine udtrukne features: {features.shape}")
