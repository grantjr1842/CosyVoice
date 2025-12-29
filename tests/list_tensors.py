from safetensors import safe_open

path = "pretrained_models/Fun-CosyVoice3-0.5B/model.safetensors"
try:
    with safe_open(path, framework="pt") as f:
        print(f"Tensors in {path}:")
        for key in sorted(f.keys()):
            print(f"  {key}: {f.get_tensor(key).shape}")
except Exception as e:
    print(f"Error: {e}")
