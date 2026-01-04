#!/usr/bin/env python3
"""Export LLM inference inputs as safetensors for Rust parity testing."""

from pathlib import Path

import torch
from safetensors.torch import save_file

# Setup paths
OUTPUT_DIR = Path("output")
DEFAULT_MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
DEFAULT_PROMPT_WAV = "./asset/interstellar-tars-01-resemble-denoised.wav"
DEFAULT_PROMPT_TEXT = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."


def main():
    from cosyvoice.cli.cosyvoice import AutoModel

    print("Loading model...")
    cosyvoice = AutoModel(model_dir=DEFAULT_MODEL_DIR)
    print(f"Model loaded. Sample rate: {cosyvoice.sample_rate} Hz")

    # Test text
    tts_text = "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?"
    prompt_prefix = "Please speak in English.<|endofprompt|>"
    full_prompt_text = prompt_prefix + DEFAULT_PROMPT_TEXT

    print("\n🔍 Exporting LLM inputs for analysis...")

    # Get model input from frontend
    model_input = cosyvoice.frontend.frontend_zero_shot(
        tts_text=tts_text,
        prompt_text=full_prompt_text,
        prompt_wav=DEFAULT_PROMPT_WAV,
        resample_rate=cosyvoice.sample_rate,
        zero_shot_spk_id="",
    )

    # Extract tensors
    export_tensors = {}

    # Text tokens
    export_tensors["text_token"] = model_input["text"].cpu()
    export_tensors["text_token_len"] = model_input["text_len"].cpu()
    export_tensors["prompt_text_token"] = model_input["prompt_text"].cpu()
    export_tensors["prompt_text_token_len"] = model_input["prompt_text_len"].cpu()

    # Speech tokens
    export_tensors["speech_token"] = model_input["llm_prompt_speech_token"].cpu()
    export_tensors["speech_token_len"] = model_input[
        "llm_prompt_speech_token_len"
    ].cpu()

    # Embeddings
    export_tensors["speaker_embedding"] = model_input["llm_embedding"].cpu()

    # Prompt mel
    export_tensors["prompt_speech_feat"] = model_input["prompt_speech_feat"].cpu()
    export_tensors["prompt_speech_feat_len"] = model_input[
        "prompt_speech_feat_len"
    ].cpu()

    # Get the LLM module to access embeddings
    llm = cosyvoice.model.llm

    # Get SOS and task_id embeddings
    print("\nLLM config:")
    print(f"  speech_token_size: {llm.speech_token_size}")
    print(f"  sos: {llm.sos}")
    print(f"  task_id: {llm.task_id}")
    print(f"  eos_token: {llm.eos_token}")
    print(f"  stop_token_ids: {llm.stop_token_ids[:5]}...")  # First 5

    # Get actual embeddings
    sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_id_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)

    export_tensors["sos_embedding"] = sos_emb.detach().cpu()
    export_tensors["task_id_embedding"] = task_id_emb.detach().cpu()

    # Get text token embeddings (using the Qwen2 model's embeddings)
    device = next(llm.parameters()).device
    combined_text = torch.concat(
        [model_input["prompt_text"], model_input["text"]], dim=1
    ).to(device)

    text_embeds = llm.llm.model.model.embed_tokens(combined_text)
    export_tensors["text_embeddings"] = text_embeds.detach().cpu()

    # Get speech token embeddings for prompt
    speech_token_emb = llm.speech_embedding(
        model_input["llm_prompt_speech_token"].to(device)
    )
    export_tensors["prompt_speech_embeddings"] = speech_token_emb.detach().cpu()

    # Build the full LLM input sequence as Python does
    lm_input = torch.concat(
        [
            sos_emb,
            text_embeds,
            task_id_emb,
            speech_token_emb,
        ],
        dim=1,
    )
    export_tensors["full_lm_input"] = lm_input.detach().cpu()

    # Print shapes
    print("\n📊 Tensor shapes:")
    for k, v in export_tensors.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}, dtype={v.dtype}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "llm_inputs.safetensors"

    # Convert all to float32 for consistency
    save_dict = {}
    for k, v in export_tensors.items():
        if v.dtype in [torch.int32, torch.int64]:
            save_dict[k] = v.contiguous().to(torch.int32)
        else:
            save_dict[k] = v.contiguous().to(torch.float32)

    save_file(save_dict, output_path)
    print(f"\n✅ Saved to {output_path}")

    # Also print first few values for quick check
    print("\n🔎 Quick peek at key tensors:")
    print(f"  SOS embedding first 5: {sos_emb.flatten()[:5].tolist()}")
    print(f"  task_id embedding first 5: {task_id_emb.flatten()[:5].tolist()}")
    print(f"  text_embeddings[0, 0, :5]: {text_embeds[0, 0, :5].tolist()}")
    print(f"  full_lm_input shape: {lm_input.shape}")
    print(f"  full_lm_input[0, 0, :5]: {lm_input[0, 0, :5].tolist()}")
    print(f"  full_lm_input[0, -1, :5]: {lm_input[0, -1, :5].tolist()}")


if __name__ == "__main__":
    main()
