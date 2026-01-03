import sys
import os
from pathlib import Path

# Add current directory to path to find .so
sys.path.append(os.getcwd())

from cosyvoice_rust_backend import Qwen2Rust, FlowRust, HiFTRust

model_dir = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B')

def test_component(name, func, *args):
    print(f"\n--- Testing {name} ---")
    try:
        func(*args)
        print(f"{name}: SUCCESS")
    except Exception as e:
        print(f"{name}: FAIL: {e}")

test_component("Qwen2Rust", Qwen2Rust, os.path.join(model_dir, 'CosyVoice-BlankEN'))
test_component("FlowRust", FlowRust, model_dir)
test_component("HiFTRust (direct)", HiFTRust, os.path.join(model_dir, 'hift.safetensors'))
