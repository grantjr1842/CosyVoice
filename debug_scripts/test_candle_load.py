import torch
import os
import sys
from pathlib import Path

# Try to find where the .so is
sys.path.append(os.getcwd())

try:
    from cosyvoice_rust_backend import HiFTRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
    print(f"Attempting to load: {p}")
    # Instead of full HiFTRust which might have other issues, let's see if we can just test VarBuilder via HiFTRust::new
    h = HiFTRust(p)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
