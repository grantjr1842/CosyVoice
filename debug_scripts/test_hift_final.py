import os
import torch
import sys
from pathlib import Path

# Add current directory to path to find .so
sys.path.append(os.getcwd())

try:
    print("Python version:", sys.version)
    print("Python path:", sys.path)
    print("Checking torch...")
    import torch
    print("Torch version:", torch.__version__)
    
    print("Checking for .so...")
    so_path = Path('cosyvoice_rust_backend.so')
    print(f"SO exists: {so_path.exists()}")

    from cosyvoice_rust_backend import HiFTRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
    print(f"Attempting to load HiFTRust from {p}")
    h = HiFTRust(p)
    print("Success loading HiFTRust")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
