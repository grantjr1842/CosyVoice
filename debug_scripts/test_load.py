import torch
import sys
from pathlib import Path
try:
    from cosyvoice_rust_backend import HiFTRust
    p = 'pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors'
    print(f"Attempting to load {p}")
    h = HiFTRust(p)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
