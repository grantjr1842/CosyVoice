import torch
from safetensors import safe_open
from pathlib import Path
import os

p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
print(f"Testing direct safetensors load from {p}")
try:
    with safe_open(p, framework="pt", device="cpu") as f:
        print("Successfully opened with safetensors")
        keys = list(f.keys())
        print(f"Number of keys: {len(keys)}")
        print(f"First 5 keys: {keys[:5]}")
except Exception as e:
    print(f"Failed to open with safetensors: {e}")

from cosyvoice_rust_backend import HiFTRust
print("\nTesting HiFTRust(p) again...")
try:
    h = HiFTRust(p)
    print("HiFTRust SUCCESS")
except Exception as e:
    print(f"HiFTRust FAIL: {e}")
