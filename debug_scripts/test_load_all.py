import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from cosyvoice_rust_backend import Qwen2Rust, FlowRust, HiFTRust

base_dir = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B')

print("--- Testing Qwen2Rust ---")
try:
    q = Qwen2Rust(os.path.join(base_dir, 'CosyVoice-BlankEN'))
    print("Qwen2Rust: SUCCESS")
except Exception as e:
    print(f"Qwen2Rust: FAIL: {e}")

print("\n--- Testing FlowRust ---")
try:
    f = FlowRust(base_dir)
    print("FlowRust: SUCCESS")
except Exception as e:
    print(f"FlowRust: FAIL: {e}")

print("\n--- Testing HiFTRust ---")
try:
    h = HiFTRust(os.path.join(base_dir, 'hift.safetensors'))
    print("HiFTRust: SUCCESS")
except Exception as e:
    print(f"HiFTRust: FAIL: {e}")
