import argparse
import sys
import torch
import torchaudio
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from safetensors.torch import save_file, load_file

def parse_args():
    parser = argparse.ArgumentParser(description="Generate verification artifacts for parity testing.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Load Model Configuration
    model_dir = repo_root / "pretrained_models" / "Fun-CosyVoice3-0.5B"
    with open(model_dir / "cosyvoice3.yaml") as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})

    # Initialize Flow
    flow = configs["flow"]
    flow_weights = load_file(model_dir / "flow.safetensors")
    flow.load_state_dict(flow_weights, strict=False)
    flow.eval()
    flow.to(device)
    print("Flow model loaded.")

    # Initialize HiFT
    # We need to reload yaml to get hift config properly if needed, or just use what we loaded if hift wasn't overridden to None?
    # Actually we overrode hift: None in the first load. Let's load it again for HiFT.
    with open(model_dir / "cosyvoice3.yaml") as f:
        configs_hift = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})

    hift = configs_hift["hift"]
    hift_weights = load_file(model_dir / "hift.safetensors")
    hift.load_state_dict(hift_weights)
    hift.eval()
    hift.to(device)
    print("HiFT model loaded.")

    # Load Inputs (using test_artifacts.safetensors or creating new ones)
    # Ideally we use standard inputs. Let's look for existing inputs to reuse or define standard ones.
    # We'll use the inputs from `test_artifacts.safetensors` if it exists, otherwise we'll fail for now as we want specific inputs.
    # Actually, let's use the ones captured in `rust/server/rust_flow_debug.safetensors` if available as they are "real" inputs.
    # Or better, let's use `tests/test_artifacts.safetensors` if available to be consistent.

    test_artifacts_path = repo_root / "tests" / "test_artifacts.safetensors"
    if not test_artifacts_path.exists():
        print(f"Warning: {test_artifacts_path} not found. Trying rust_flow_debug.safetensors")
        input_source = repo_root / "rust" / "server" / "rust_flow_debug.safetensors"
    else:
        input_source = test_artifacts_path

    print(f"Loading inputs from {input_source}")
    inputs = load_file(input_source)

    # Extract Flow Inputs
    # We expect: output (speech tokens), prompt_speech_token, prompt_speech_feat, embedding

    # Handle different naming conventions depending on source
    def get_tensor(keys, required=True):
        for k in keys:
            if k in inputs:
                return inputs[k].to(device)
        if required:
            raise ValueError(f"Could not find any of {keys} in inputs: {inputs.keys()}")
        return None

    token = get_tensor(["token", "speech_token", "output"]) # Target speech tokens
    prompt_token = get_tensor(["prompt_token", "prompt_speech_token"])
    prompt_feat = get_tensor(["prompt_feat", "prompt_speech_feat"]) # Prompt Mel
    embedding = get_tensor(["embedding", "spk_embed"])

    print(f"  token: {token.shape}")
    print(f"  prompt_token: {prompt_token.shape}")
    print(f"  prompt_feat: {prompt_feat.shape}")
    print(f"  embedding: {embedding.shape}")

    tensors_to_save = {}

    # 1. Save Inputs
    tensors_to_save["token"] = token.cpu()
    tensors_to_save["prompt_token"] = prompt_token.cpu()
    tensors_to_save["prompt_feat"] = prompt_feat.cpu()
    tensors_to_save["embedding"] = embedding.cpu()

    # 2. Run Flow Inference
    # We want to trace intermediate steps if possible, but mainly the final mel
    print("Running Flow Inference...")
    with torch.no_grad():
        # flow.inference logic:
        # token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding
        token_len = torch.tensor([token.shape[1]], device=device)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], device=device)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], device=device)

        # Check Flow signature
        # inference(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding)
        try:
             # Some versions might return tuple
             flow_out = flow.inference(token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding)
             if isinstance(flow_out, tuple):
                 flow_out = flow_out[0] # mel
        except Exception as e:
            print(f"Error running flow inference: {e}")
            # Try calling forward/decoder directly if needed, but inference is best.
            # Assuming standard CosyVoice3 flow.
            raise e

    print(f"  flow_output (mel): {flow_out.shape}")
    tensors_to_save["python_flow_output"] = flow_out.cpu()

    # 3. Run HiFT Inference
    print("Running HiFT Inference...")
    with torch.no_grad():
        try:
            # hift.inference(mel)
            hift_out = hift.inference(flow_out)
             # returns dict or tensor?
            if isinstance(hift_out, dict):
                audio = hift_out['wav']
            elif isinstance(hift_out, tuple):
                audio = hift_out[0]
            else:
                audio = hift_out
        except Exception as e:
            print(f"Error running hift inference: {e}")
            raise e

    print(f"  hift_output (audio): {audio.shape}")
    tensors_to_save["python_audio_output"] = audio.cpu()

    # 4. Save to debug_artifacts.safetensors
    output_path = repo_root / "debug_artifacts.safetensors"
    save_file(tensors_to_save, output_path)
    print(f"Saved artifacts to {output_path}")

    # Save Wav
    wav_path = repo_root / "debug_artifacts.wav"
    torchaudio.save(wav_path, audio.cpu(), 24000)
    print(f"Saved audio to {wav_path}")

if __name__ == "__main__":
    main()
