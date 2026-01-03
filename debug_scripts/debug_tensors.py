#!/usr/bin/env python3
import types

from safetensors.torch import save_file

from cosyvoice.cli.cosyvoice import AutoModel


def generate_debug_artifacts():
    print("Loading model...")
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    cosyvoice = AutoModel(model_dir=model_dir)

    # Monkey patch flow.inference
    original_inference = cosyvoice.model.flow.inference

    # Container for captured data
    captured_data = {}

    def hooked_inference(self, *args, **kwargs):
        print(
            f"Hooked flow.inference called! Args len: {len(args)}, Kwargs: {kwargs.keys()}"
        )
        try:
            ret = original_inference(*args, **kwargs)
            # Handle return: usually (feat, something)
            if isinstance(ret, tuple):
                feat = ret[0]
            else:
                feat = ret

            print("Capturing Flow artifacts...")

            # Extract inputs safely
            # Arg mapping for Causal/Non-Causal usually:
            # 0: token, 1: token_len, 2: prompt_token, 3: prompt_token_len, 4: prompt_feat, 5: prompt_feat_len, 6: embedding

            prompt_feat = kwargs.get("prompt_feat")
            if prompt_feat is None and len(args) > 4:
                prompt_feat = args[4]

            embedding = kwargs.get("embedding")
            if embedding is None and len(args) > 6:
                embedding = args[6]

            prompt_token = kwargs.get("prompt_token")
            if prompt_token is None and len(args) > 2:
                prompt_token = args[2]

            if prompt_feat is not None:
                # prompt_feat is [1, T, 80]. Transpose to [1, 80, T] for Rust
                captured_data["python_mel_24k"] = (
                    prompt_feat.detach().cpu().transpose(1, 2).contiguous()
                )
            else:
                print("WARNING: Could not find prompt_feat input")

            # feat is [1, 80, T]. Do NOT transpose.
            captured_data["python_flow_output"] = feat.detach().cpu().contiguous()

            if embedding is not None:
                captured_data["python_spk_emb"] = embedding.detach().cpu().contiguous()

            if prompt_token is not None:
                captured_data["python_speech_tokens"] = (
                    prompt_token.detach().cpu().contiguous()
                )

            print(f"Captured Flow Output Shape: {feat.shape}")

            return ret
        except Exception as e:
            print(f"Error in hooked_inference: {e}")
            import traceback

            traceback.print_exc()
            raise e

    # Bind hooked method
    cosyvoice.model.flow.inference = types.MethodType(
        hooked_inference, cosyvoice.model.flow
    )

    # Hook HiFT inference
    if hasattr(cosyvoice.model, "hift"):
        print("Hooking HiFT...")
        original_hift_inference = cosyvoice.model.hift.inference

        def hooked_hift_inference(self, speech_feat, finalize=True):
            print(f"Hooked HiFT inference. Input shape: {speech_feat.shape}")
            # speech_feat is [1, 80, T]. No transpose needed.
            captured_data["python_hift_mel_input"] = (
                speech_feat.detach().cpu().contiguous()
            )

            out, source = original_hift_inference(speech_feat, finalize=finalize)

            captured_data["python_hift_source"] = source.detach().cpu().contiguous()
            captured_data["python_hift_audio"] = out.detach().cpu().contiguous()
            print("Captured HiFT artifacts.")
            return out, source

        cosyvoice.model.hift.inference = types.MethodType(
            hooked_hift_inference, cosyvoice.model.hift
        )

        # Hook F0 Predictor
        if hasattr(cosyvoice.model.hift, "f0_predictor"):
            print("Hooking F0Predictor...")
            original_f0_forward = cosyvoice.model.hift.f0_predictor.forward

            def hooked_f0_forward(self, *args, **kwargs):
                # args[0] is x
                x = args[0]
                print(f"Hooked F0 forward. Input: {x.shape}")
                f0 = original_f0_forward(*args, **kwargs)
                captured_data["python_f0"] = f0.detach().cpu().contiguous()
                print(f"Captured F0. Shape: {f0.shape}")
                return f0

            # Bind
            cosyvoice.model.hift.f0_predictor.forward = types.MethodType(
                hooked_f0_forward, cosyvoice.model.hift.f0_predictor
            )

    prompt_wav = "./asset/interstellar-tars-01-resemble-denoised.wav"
    prompt_text = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."
    tts_text = "Hello world."

    print("Running inference to capture artifacts...")
    # Clean output dir to avoid confusion? No need.

    # Run simple inference
    # Note: we don't care about audio output, just capturing flow output
    try:
        for _ in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav):
            pass
    except Exception as e:
        print(f"Inference interrupted (expected if saving and exiting): {e}")

    if captured_data:
        print(
            f"Saving artifacts to debug_artifacts.safetensors with keys: {list(captured_data.keys())}"
        )
        save_file(captured_data, "debug_artifacts.safetensors")
        print("Saved debug_artifacts.safetensors")
    else:
        print("Error: No artifacts captured!")


if __name__ == "__main__":
    generate_debug_artifacts()
