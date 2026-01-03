import os
try:
    from cosyvoice_rust_backend import FlowRust
    p = os.path.abspath('pretrained_models/Fun-CosyVoice3-0.5B')
    print(f"Attempting to load FlowRust from {p}")
    f = FlowRust(p)
    print("Success loading FlowRust")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
