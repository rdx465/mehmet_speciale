import torch
from neurovfm.pipelines import load_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kører på: {device}")

print("1) Loader NeuroVFM encoder...")
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")
encoder.model = encoder.model.to(device)

# Vi peger på den fil, vi lige har kopieret og omdøbt
scan_path = "test_BrainWindow.nii"

print(f"2) Preprocesser ægte CT-scanning: {scan_path}...")
batch = preprocessor.load_study(scan_path, modality="ct")

# Flyt alt data fra scanningen over på dit grafikkort
for key in batch:
    if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].to(device)

print("3) Kører forward pass ")
with torch.no_grad():
    out = encoder.embed(batch)

features = out[0] if isinstance(out, tuple) else out

print("\nÆGTE CT-DATA SUCCESS!")
print(f"Shape af dine udtrukne features: {features.shape}")
print(f"Hjernen blev delt op i {features.shape[0]} brikker, med {features.shape[1]} features i hver!")
