import os
import sys
sys.path.append(os.getcwd())
from cosyvoice_rust_backend import HiFTRust

p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.pt')
print(f"Testing load from: {p}")
try:
    h = HiFTRust(p)
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {e}")
