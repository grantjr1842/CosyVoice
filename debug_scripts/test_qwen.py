import os
try:
    from cosyvoice_rust_backend import Qwen2Rust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN')
    print(f"Attempting to load Qwen2Rust from {p}")
    q = Qwen2Rust(p)
    print("Success loading Qwen2Rust")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
