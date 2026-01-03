import torch
import os
import sys
from pathlib import Path

# Add current directory to path to find .so
sys.path.append(os.getcwd())

try:
    from cosyvoice_rust_backend import HiFTRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.pt')
    print(f"Attempting to load PTH: {p}")
    h = HiFTRust(p)
    print("Success loading PTH")
except Exception as e:
    print(f"Error loading PTH: {e}")

try:
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
    print(f"\nAttempting to load Safetensors: {p}")
    h = HiFTRust(p)
    print("Success loading Safetensors")
except Exception as e:
    print(f"Error loading Safetensors: {e}")
