import os
import torch
try:
    from cosyvoice_rust_backend import HiFTRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
    print(f"Attempting to load HiFTRust from {p}")
    h = HiFTRust(p)
    print("Success loading HiFTRust")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
