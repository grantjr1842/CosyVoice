import os
from pathlib import Path

# Try to find where the .so is
for root, dirs, files in os.walk('.'):
    for f in files:
        if f == 'cosyvoice_rust_backend.so':
             print(f"Found .so at: {os.path.join(root, f)}")

try:
    from cosyvoice_rust_backend import HiFTRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
    print(f"Attempting to load {p}")
    print(f"File exists: {os.path.exists(p)}")
    print(f"Is file: {os.path.isfile(p)}")
    h = HiFTRust(p)
    print("Success")
except Exception as e:
    print(f"Error type: {type(e)}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
