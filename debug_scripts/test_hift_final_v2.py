import os
import sys
from pathlib import Path

# Add current directory to path to find .so
sys.path.append(os.getcwd())

try:
    from cosyvoice_rust_backend import HiFTRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
    print(f"Testing HiFTRust with: {p}")
    h = HiFTRust(p)
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {e}")
