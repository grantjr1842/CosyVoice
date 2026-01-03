import os

import torch
from safetensors.torch import save_file

model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
pt_path = os.path.join(model_dir, "hift.pt")
sf_path = os.path.join(model_dir, "hift_verify.safetensors")

print(f"Loading {pt_path}...")
state_dict = torch.load(pt_path, map_location="cpu")

# Check keys?
# print(state_dict.keys())

print(f"Saving to {sf_path}...")
save_file(state_dict, sf_path)
print("Done.")
