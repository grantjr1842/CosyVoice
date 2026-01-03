import sys
import os
from pathlib import Path

# Add current directory to path to find .so
sys.path.append(os.getcwd())

from cosyvoice_rust_backend import HiFTRust

def test_load(path_str):
    print(f"\n--- Testing load from: {path_str} ---")
    p = os.path.abspath(path_str)
    print(f"Abs path: {p}")
    print(f"Exists: {os.path.exists(p)}")
    print(f"Is file: {os.path.isfile(p)}")
    try:
        h = HiFTRust(p)
        print("SUCCESS")
    except Exception as e:
        print(f"FAIL: {e}")

# Test with original filename
test_load('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
test_load('pretrained_models/Fun-CosyVoice3-0.5B/hift.pt')
