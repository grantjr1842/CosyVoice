#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from hyperpyyaml import load_hyperpyyaml
from tokenizers import Tokenizer


def greedy_sampling(weighted_scores, decoded_tokens, sampling):
    return int(torch.argmax(weighted_scores).item())


def main():
    payload = json.load(sys.stdin)
    text = payload["text"]
    model_dir = Path(payload["model_dir"])
    tokenizer_path = Path(payload["tokenizer_path"])
    sampling_k = int(payload.get("sampling_k", 1))
    min_ratio = float(payload.get("min_ratio", 2.0))
    max_ratio = float(payload.get("max_ratio", 20.0))
    use_rl = bool(payload.get("use_rl", True))
    greedy = bool(payload.get("greedy", True))

    if payload.get("device"):
        device = torch.device(payload["device"])
    else:
        use_cuda = os.getenv("COSYVOICE_USE_CUDA", "1") != "0"
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    yaml_path = model_dir / "cosyvoice3.yaml"
    with yaml_path.open("r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={"qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")},
        )

    llm = configs["llm"]
    if greedy:
        llm.sampling = greedy_sampling

    rl_st = model_dir / "llm.rl.safetensors"
    base_st = model_dir / "llm.safetensors"
    rl_pt = model_dir / "llm.rl.pt"
    base_pt = model_dir / "llm.pt"

    if use_rl and rl_st.exists():
        state = load_file(str(rl_st))
    elif base_st.exists():
        state = load_file(str(base_st))
    else:
        llm_path = rl_pt if use_rl and rl_pt.exists() else base_pt
        state = torch.load(llm_path, map_location=device, weights_only=True)

    llm.load_state_dict(state, strict=True)
    llm.to(device).eval()

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text_ids = tokenizer.encode(text, add_special_tokens=True).ids
    text_tensor = torch.tensor([text_ids], dtype=torch.int64, device=device)
    text_len = torch.tensor([len(text_ids)], dtype=torch.int32, device=device)

    prompt_text = torch.zeros((1, 0), dtype=torch.int64, device=device)
    prompt_text_len = torch.tensor([0], dtype=torch.int32, device=device)
    prompt_speech_token = torch.zeros((1, 0), dtype=torch.int32, device=device)
    prompt_speech_token_len = torch.tensor([0], dtype=torch.int32, device=device)
    embedding = torch.zeros((0, 0), dtype=torch.float32, device=device)

    with torch.inference_mode():
        tokens = list(
            llm.inference(
                text=text_tensor,
                text_len=text_len,
                prompt_text=prompt_text,
                prompt_text_len=prompt_text_len,
                prompt_speech_token=prompt_speech_token,
                prompt_speech_token_len=prompt_speech_token_len,
                embedding=embedding,
                sampling=sampling_k,
                min_token_text_ratio=min_ratio,
                max_token_text_ratio=max_ratio,
                uuid="",
            )
        )

    result = {
        "text_len": len(text_ids),
        "min_len": int(len(text_ids) * min_ratio),
        "max_len": int(len(text_ids) * max_ratio),
        "token_len": len(tokens),
    }
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
