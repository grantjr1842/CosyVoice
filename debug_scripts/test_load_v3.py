import os
import torch
from pathlib import Path

try:
    from cosyvoice_rust_backend import HiFTRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.pt')
    print(f"Attempting to load PTH: {p}")
    print(f"File exists: {os.path.exists(p)}")
    h = HiFTRust(p)
    print("Success loading PTH")
except Exception as e:
    print(f"Error loading PTH: {e}")
