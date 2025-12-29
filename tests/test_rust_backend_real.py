import os
import sys
import torch
import numpy as np

# Add paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'rust/server'))

try:
    import cosyvoice_rust_backend
    print("Successfully imported cosyvoice_rust_backend")
except ImportError as e:
    print(f"Failed to import cosyvoice_rust_backend: {e}")
    sys.exit(1)

model_dir = os.path.abspath("pretrained_models/Fun-CosyVoice3-0.5B")
print(f"Using model_dir: {model_dir}")

def test_qwen():
    print("\nTesting Qwen2Rust initialization...")
    try:
        llm = cosyvoice_rust_backend.Qwen2Rust(os.path.join(model_dir, "CosyVoice-BlankEN"))
        print("Qwen2Rust initialized successfully")

        # Minimal forward test
        input_ids = [1, 2, 3, 4]
        shape = [1, 4]
        out = llm.forward(input_ids, shape)
        print(f"Qwen2Rust forward success, output size: {len(out)}")
    except Exception as e:
        print(f"Qwen2Rust failed: {e}")

def test_flow():
    print("\nTesting FlowRust initialization...")
    try:
        flow = cosyvoice_rust_backend.FlowRust(model_dir)
        print("FlowRust initialized successfully")

        # Minimal inference test
        B, N, mel_dim = 1, 10, 80
        mu = np.random.randn(B, mel_dim, N).astype(np.float32)
        mask = np.ones((B, 1, N)).astype(np.float32)
        spks = np.random.randn(B, mel_dim).astype(np.float32)
        cond = np.random.randn(B, mel_dim, N).astype(np.float32)

        out = flow.inference(mu, mask, 1, 1.0, spks, cond)
        print(f"FlowRust inference success, output shape: {out.shape}")
    except Exception as e:
        print(f"FlowRust failed: {e}")

def test_hift():
    print("\nTesting HiFTRust initialization...")
    try:
        # Check both .safetensors and .pt
        hift_path = os.path.join(model_dir, "hift.safetensors")
        if not os.path.exists(hift_path):
            hift_path = os.path.join(model_dir, "hift.pt")

        print(f"Loading HiFT from: {hift_path}")
        hift = cosyvoice_rust_backend.HiFTRust(hift_path)
        print("HiFTRust initialized successfully")

        # Minimal inference test
        B, N, mel_dim = 1, 10, 80
        mel_flat = np.random.randn(B * mel_dim * N).astype(np.float32).tolist()
        out = hift.inference(mel_flat, B, mel_dim, N)
        print(f"HiFTRust inference success, output size: {len(out)}")
    except Exception as e:
        print(f"HiFTRust failed: {e}")

if __name__ == "__main__":
    test_qwen()
    test_flow()
    test_hift()
