import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from cosyvoice_rust_backend import HiFTRust

p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors')
print(f"Testing HiFTRust with {p}")
try:
    # We suspect the issue is inside VarBuilder::from_mmaped_safetensors which is called by HiFTRust::new
    h = HiFTRust(p)
    print("HiFTRust: SUCCESS")
except Exception as e:
    print(f"HiFTRust: FAIL: {e}")
    import traceback
    traceback.print_exc()
