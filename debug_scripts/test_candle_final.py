import sys
import os
from pathlib import Path

# Add current directory to path to find .so
sys.path.append(os.getcwd())

from cosyvoice_rust_backend import HiFTRust

p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
print(f"Testing load from: {p}")
try:
    h = HiFTRust(p)
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {e}")
    # Inspect the current working directory and environment
    print(f"CWD: {os.getcwd()}")
    print(f"PATH: {os.environ.get('PATH')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
