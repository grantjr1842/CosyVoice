import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from cosyvoice_rust_backend import HiFTRust

p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.pt')
print(f"Testing HiFTRust with PTH: {p}")
try:
    h = HiFTRust(p)
    print("HiFTRust SUCCESS")
except Exception as e:
    print(f"HiFTRust FAIL: {e}")
