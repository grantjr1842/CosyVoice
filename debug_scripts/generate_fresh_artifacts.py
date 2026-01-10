#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from safetensors.torch import save_file

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from cosyvoice.cli.cosyvoice import AutoModel

def main():
    parser = argparse.ArgumentParser(description="Generate Python artifacts for parity testing.")
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="Path to CosyVoice3 model directory.",
    )
    parser.add_argument(
        "--prompt-wav",
        default="./asset/interstellar-tars-01-resemble-denoised.wav",
        help="Path to prompt WAV file.",
    )
    parser.add_argument(
        "--prompt-text",
        default=(
            "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. "
            "Nothing's changed on that."
        ),
        help="Prompt text (transcription for the prompt audio).",
    )
    parser.add_argument(
        "--tts-text",
        default="Hello! This is a test for parity verification.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write debug_artifacts.safetensors and debug_artifacts.wav.",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=0,
        help="Which normalized TTS segment to use (default: 0).",
    )
    args = parser.parse_args()

    # Configuration
    model_dir = args.model_dir
    prompt_wav = args.prompt_wav
    prompt_text = args.prompt_text
    tts_text = args.tts_text

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    cosyvoice = AutoModel(
        model_dir=model_dir,
        load_trt=False,
        fp16=False,
    )
    print("Model loaded.")

    # 1. Frontend
    print("Running Frontend...")
    # We need to look at inference_zero_shot logic to properly call frontend
    # prompt_text normalization
    prompt_text_norm = cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
    # tts_text normalization
    tts_text_norm_list = cosyvoice.frontend.text_normalize(
        tts_text, split=True, text_frontend=True
    )
    if not tts_text_norm_list:
        print("Error: text_normalize returned no segments.")
        sys.exit(1)
    if args.segment_index >= len(tts_text_norm_list) or args.segment_index < 0:
        print(
            f"Error: segment_index {args.segment_index} out of range "
            f"(0..{len(tts_text_norm_list) - 1})."
        )
        sys.exit(1)
    tts_text_norm = tts_text_norm_list[args.segment_index]

    model_input = cosyvoice.frontend.frontend_zero_shot(
        tts_text_norm, prompt_text_norm, prompt_wav, cosyvoice.sample_rate, ""
    )

    # model_input contains:
    # text, flow_embedding, llm_embedding, prompt_text, llm_prompt_speech_token,
    # flow_prompt_speech_token, prompt_speech_feat, source_speech_token

    # Move inputs to device
    def to_device(d):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                new_d[k] = v.to(device)
            else:
                new_d[k] = v
        return new_d

    model_input = to_device(model_input)

    # 2. LLM Inference
    print("Running LLM...")
    llm = cosyvoice.model.llm

    # Extract args for LLM
    text = model_input["text"]
    text_len = torch.tensor([text.shape[1]], dtype=torch.int32, device=device)
    prompt_text_t = model_input["prompt_text"]
    prompt_text_len = torch.tensor([prompt_text_t.shape[1]], dtype=torch.int32, device=device)
    llm_prompt_speech_token = model_input["llm_prompt_speech_token"]
    llm_prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32, device=device)
    llm_embedding = model_input["llm_embedding"]

    # Run LLM (non-streaming)
    # inference yield tokens, we collect them
    tts_speech_tokens = []
    print(f"Prompt text tokens: {prompt_text_t}")
    print(f"Target text tokens: {text}")
    print(f"Combined text tokens: {torch.cat([prompt_text_t, text], dim=1)}")

    with torch.inference_mode():
        for i in llm.inference(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text_t,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=llm_prompt_speech_token,
            prompt_speech_token_len=llm_prompt_speech_token_len,
            embedding=llm_embedding,
            uuid="test_uuid",
            sampling=25
        ):
            tts_speech_tokens.append(i)

    # Concatenate tokens
    # tts_speech_tokens list of tensors? No, inference yields tensor or list?
    # Checking cosyvoice/cli/model.py llm_job logic:
    # self.tts_speech_token_dict[uuid].append(i) -> it assumes i is appendable/mergeable
    # Usually LLM inference yields one token or chunk of tokens
    # let's assume it yields tensors and we cat them

    # Actually, let's verify what llm.inference yields.
    # It likely yields single token or small chunks.
    # model.py: self.tts_speech_token_dict[uuid].append(i)
    # Then: this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(0)
    # This implies `i` is an integer or a list of integers, NOT a tensor.

    # If `i` is tensor, torch.tensor([tensor]) would be weird.
    # Let's check `cosyvoice/model/llm.py` if needed, but model.py suggests it's a listable item.

    # For now let's assume flattened list of ints.
    full_speech_token_list = []
    for x in tts_speech_tokens:
        if isinstance(x, torch.Tensor):
            full_speech_token_list.extend(x.flatten().tolist())
        elif isinstance(x, list):
            full_speech_token_list.extend(x)
        else:
            full_speech_token_list.append(x)

    token = torch.tensor(full_speech_token_list, dtype=torch.int32, device=device).unsqueeze(0)
    print(f"Generated {token.shape[1]} speech tokens.")
    print(f"First 20 speech tokens: {full_speech_token_list[:20]}")

    # 3. Flow Inference
    print("Running Flow...")
    flow = cosyvoice.model.flow

    flow_prompt_speech_token = model_input["flow_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    flow_embedding = model_input["flow_embedding"]

    token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=device)
    prompt_token_len = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32, device=device)
    prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=device)

    with torch.inference_mode():
        # inference(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, ...)
        flow_out, _ = flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=flow_prompt_speech_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=flow_embedding,
            streaming=False,
            finalize=True
        )

    # flow_out is [Batch, 80, Time]
    print(f"Flow output (mel) shape: {flow_out.shape}")

    # 4. HiFT Inference
    print("Running HiFT...")
    hift = cosyvoice.model.hift

    with torch.inference_mode():
        audio, source = hift.inference(speech_feat=flow_out, finalize=True)

    print(f"HiFT output (audio) shape: {audio.shape}")

    # 5. Save Artifacts
    print("Saving artifacts...")
    artifacts = {
        "text": text.contiguous().cpu(), # [1, T_text]
        "prompt_text": prompt_text_t.contiguous().cpu(), # [1, T_prompt]
        "llm_prompt_speech_token": llm_prompt_speech_token.contiguous().cpu(),
        "llm_embedding": llm_embedding.contiguous().cpu(),
        "speech_tokens": token.contiguous().cpu(),
        # Actually for flow test we need: token, prompt_token, prompt_feat, embedding
        # Let's verify naming in rust/test_flow.rs later

        # Flow Inputs
        "token": token.contiguous().cpu(), # The generated tokens
        "prompt_token": flow_prompt_speech_token.contiguous().cpu(),
        "prompt_feat": prompt_feat.contiguous().cpu(),
        "embedding": flow_embedding.contiguous().cpu(),

        # Flow Output
        "python_flow_output": flow_out.contiguous().cpu(),

        # Audio Output
        "python_audio_output": audio.contiguous().cpu(),
        "python_hift_source": source.contiguous().cpu(),

        # Internal Noise Buffer (for parity)
        "rand_noise": flow.decoder.rand_noise.contiguous().cpu()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "debug_artifacts.safetensors"
    save_file(artifacts, output_path)
    print(f"Saved artifacts to {output_path}")

    wav_path = output_dir / "debug_artifacts.wav"
    torchaudio.save(str(wav_path), audio.contiguous().cpu(), 24000)
    print(f"Saved audio to {wav_path}")

if __name__ == "__main__":
    main()
