import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from cosyvoice_rust_backend import HiFTRust

def test_load(path_str):
    p = os.path.abspath(path_str)
    print(f"\n--- Testing load from: {p} ---")
    try:
        h = HiFTRust(p)
        print("SUCCESS")
    except Exception as e:
        print(f"FAIL: {e}")

# Try loading directly
test_load('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
