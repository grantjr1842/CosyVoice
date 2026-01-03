import os
from pathlib import Path

model_dir = Path('pretrained_models/Fun-CosyVoice3-0.5B')

# 1. Restore model.safetensors.bak if it exists
bak = model_dir / 'model.safetensors.bak'
if bak.exists():
    target = model_dir / 'model.safetensors'
    if target.exists():
        print(f"Removing existing {target}")
        target.unlink()
    bak.rename(target)
    print(f"Restored {target}")

# 2. Rename LLM and Flow weights to avoid conflict if HiFT is also named hift.safetensors
# We suspect Candle might be doing some global mmap or prefix search.
# But actually, the error says it's looking for hift.safetensors/model.safetensors.
# This happens if VarBuilder thinks hift.safetensors is a directory containing a model.safetensors file.

# Let's try to rename hift.safetensors to something unique that doesn't share a base name with others.
hift = model_dir / 'hift.safetensors'
if hift.exists():
    new_hift = model_dir / 'hift_gen.safetensors'
    if new_hift.exists():
        new_hift.unlink()
    hift.rename(new_hift)
    print(f"Renamed hift.safetensors to {new_hift}")

