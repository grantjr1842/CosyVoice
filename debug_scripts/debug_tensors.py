#!/usr/bin/env python3
import types

import torch
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

            # Hook conv_pre
            original_conv_pre_forward = cosyvoice.model.hift.conv_pre.forward

            def hooked_conv_pre_forward(self, *args, **kwargs):
                out = original_conv_pre_forward(*args, **kwargs)
                captured_data["python_conv_pre"] = out.detach().cpu().contiguous()
                print(f"Captured conv_pre. Shape: {out.shape}")
                return out

            cosyvoice.model.hift.conv_pre.forward = types.MethodType(
                hooked_conv_pre_forward, cosyvoice.model.hift.conv_pre
            )

            # Hook ResBlock 0 output (Loop 0 Fusion)
            # Cannot easily hook local variable inside 'decode'.
            # But we can hook 'ups[1]' INPUT. 'ups[1]' is called with Loop 0 Output.
            # We already hooked 'ups[1]'.
            # We can capture 'args[0]' in hooked_ups_1_forward!
            original_ups_1_forward = cosyvoice.model.hift.ups[1].forward

            def hooked_ups_1_forward_v2(self, *args, **kwargs):
                x_in = args[0]
                captured_data["python_loop_0_output"] = x_in.detach().cpu().contiguous()
                print(f"Captured Loop 0 Output (ups[1] input). Shape: {x_in.shape}")
                out = original_ups_1_forward(*args, **kwargs)
                captured_data["python_ups_1"] = out.detach().cpu().contiguous()
                print(f"Captured ups[1]. Shape: {out.shape}")
                return out

            cosyvoice.model.hift.ups[1].forward = types.MethodType(
                hooked_ups_1_forward_v2, cosyvoice.model.hift.ups[1]
            )

            # Hook ups[0]
            original_ups_0_forward = cosyvoice.model.hift.ups[0].forward

            def hooked_ups_0_forward(self, *args, **kwargs):
                out = original_ups_0_forward(*args, **kwargs)
                captured_data["python_ups_0"] = out.detach().cpu().contiguous()
                print(f"Captured ups[0]. Shape: {out.shape}")
                return out

            cosyvoice.model.hift.ups[0].forward = types.MethodType(
                hooked_ups_0_forward, cosyvoice.model.hift.ups[0]
            )

            # Hook Snake
            # Access Snake class via instance to avoid import issues
            # hift structure: hift -> resblocks -> acti1 -> Snake
            # We assume at least one resblock exists.
            try:
                # Try to find a Snake instance
                r0 = cosyvoice.model.hift.resblocks[0]
                print(f"ResBlock 0 attributes: {dir(r0)}")

                # Check known variants
                if hasattr(r0, "acti1"):
                    snake_instance = r0.acti1[0]
                elif hasattr(r0, "activations1"):
                    snake_instance = r0.activations1[0]
                elif hasattr(r0, "act1"):
                    snake_instance = r0.act1[0]
                else:
                    raise AttributeError("Could not find activations in ResBlock")

                Snake = snake_instance.__class__
                import inspect

                print(f"Snake Source Code:\n{inspect.getsource(Snake)}")
                print(f"Found Snake class: {Snake}")

                original_snake_forward = Snake.forward

                def hooked_snake_forward(self, x):
                    # Log alpha stats
                    if hasattr(self, "alpha"):
                        a = self.alpha.detach().cpu().numpy()
                        # Check if log scale logic is used?
                        # The Snake class usually has alpha_logscale.
                        # Check definition?
                        # But we just print raw alpha.
                        # print(f"Captured Snake Alpha. Mean: {a.mean():.4e}, Min: {a.min():.4e}, Max: {a.max():.4e}. LogScale: {getattr(self, 'l', 'Unknown')}")
                        # Reduce log spam, print only first call or summary?
                        # We print every call. It will be spammy but useful.
                        print(
                            f"Captured Snake Alpha. Mean: {a.mean():.4e}, Min: {a.min():.4e}, Max: {a.max():.4e}. LogScale: {getattr(self, 'alpha_logscale', 'Unknown')}"
                        )

                        # If logscale, calculate effective alpha
                        if getattr(self, "alpha_logscale", False):
                            eff = torch.exp(torch.tensor(a)).numpy()
                            print(f"  Effective Alpha (exp): Mean: {eff.mean():.4e}")
                        else:
                            print(f"  Effective Alpha (linear): Mean: {a.mean():.4e}")

                    return original_snake_forward(self, x)

                Snake.forward = hooked_snake_forward

                        elif hasattr(c1, "weight"):
                            w = c1.weight.detach().cpu()
                            print(
                                f"ResBlock Conv1 Weight Stats: Mean={w.mean().item():.4e}, Min={w.min().item():.4e}, Max={w.max().item():.4e}"
                            )

                        if hasattr(c1, "bias") and c1.bias is not None:
                            b = c1.bias.detach().cpu()
                            print(
                                f"ResBlock Conv1 Bias Stats: Mean={b.mean().item():.4e}"
                            )
                except Exception as e:
                    print(f"Failed to dump Conv Weights: {e}")

            except Exception as e:
                print(f"Failed to hook Snake: {e}")

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
