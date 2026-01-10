
import os
import sys
import torch
from safetensors.torch import save_file
import logging

# Add project root to sys.path
sys.path.append(os.getcwd())

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

def main():
    logging.basicConfig(level=logging.INFO)
    set_all_random_seed(1234)

    print("Loading model...")
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    try:
        cosyvoice = AutoModel(model_dir=model_dir)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Prepare inputs
    # Text: "Hello world"
    text = "Hello world"

    print("Preparing inputs...")
    model = cosyvoice.model.llm
    device = next(model.parameters()).device

    # Access frontend
    frontend = cosyvoice.frontend

    prompt_text = "This is a prompt."
    prompt_speech_tokens = torch.randint(0, 100, (1, 50), device=device)

    # Text tokens
    model_input = frontend.frontend_zero_shot(text, prompt_text, "./asset/interstellar-tars-01-resemble-denoised.wav", 16000, "") # Use real wav
    # model_input is a dict.
    text_token = model_input["text"].to(device) # [1, L]
    text_token_len = model_input["text_len"].to(device)
    prompt_text_token = model_input["prompt_text"].to(device)
    prompt_text_token_len = model_input["prompt_text_len"].to(device)

    # LLM Inference input construction (from llm.py inference method)
    full_text_token = torch.concat([prompt_text_token, text_token], dim=1)

    # 1. Encode text (We want to pass embeddings to Rust, so we do it here)
    with torch.inference_mode():
        # Access the underlying Qwen2Encoder
        qwen_encoder = model.llm # Qwen2Encoder
        # Qwen2Encoder has self.model = Qwen2ForCausalLM

        # Debug types
        print(f"DEBUG: qwen_encoder type: {type(qwen_encoder)}")

        # Use get_input_embeddings which is safer
        # qwen_encoder.model is Qwen2ForCausalLM
        if hasattr(qwen_encoder.model, "get_input_embeddings"):
             text_embeds = qwen_encoder.model.get_input_embeddings()(full_text_token)
        else:
             # Fallback
             text_embeds = qwen_encoder.model.model.embed_tokens(full_text_token)
        print(f"Text Embeds Shape: {text_embeds.shape}")

        # 2. Prepare inputs for Rust debug_forward_one

        # Construct full input manually to be sure
        # CosyVoice3LM uses speech_embedding for SOS and Task ID
        sos_emb = model.speech_embedding.weight[model.sos].reshape(1, 1, -1)
        task_id_emb = model.speech_embedding.weight[model.task_id].reshape(1, 1, -1)

        # Embed prompt speech tokens
        prompt_speech_token_emb = model.speech_embedding(prompt_speech_tokens)

        lm_input = torch.concat(
            [sos_emb, text_embeds, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # 3. Run Qwen Forward
        lm_input_len = torch.tensor([lm_input.size(1)], device=device)

        # Qwen2Encoder.forward returns (hidden_states[-1], mask)
        hidden_states, _ = qwen_encoder.forward(lm_input, lm_input_len)

        # 4. Decode to logits
        # Rust `debug_forward_one` computes logits for the *last position*
        last_hidden_state = hidden_states[:, -1:, :] # [1, 1, H]
        logits = model.llm_decoder(last_hidden_state) # [1, 1, V]

        print(f"Logits Shape: {logits.shape}")

    # Save artifacts
    tensors = {
        "text_embeds": text_embeds.cpu().contiguous(),
        "prompt_speech_tokens": prompt_speech_tokens.cpu().contiguous(),
        "expected_logits": logits.cpu().contiguous(),
        # Save verify info
        "sos_emb": sos_emb.cpu().contiguous(),
        "task_id_emb": task_id_emb.cpu().contiguous(),
        "prompt_speech_token_emb": prompt_speech_token_emb.cpu().contiguous(),
        "lm_input": lm_input.cpu().contiguous(),
        "last_hidden_state": last_hidden_state.cpu().contiguous()
    }

    save_file(tensors, "debug_llm_data.safetensors")
    print("Saved debug_llm_data.safetensors")

if __name__ == "__main__":
    main()
