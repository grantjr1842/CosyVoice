import argparse
import gc
import os
from contextlib import nullcontext

from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture Flow-related tensors and mel output for debugging."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run the capture on (auto selects CUDA when available).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    import torch

    from cosyvoice.cli.cosyvoice import AutoModel

    def determine_device() -> torch.device:
        if args.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        desired = torch.device(args.device)
        if desired.type == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return desired

    def is_cuda_device(device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.is_available()

    def reset_cuda_state(device: torch.device):
        if not is_cuda_device(device):
            return
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except AttributeError:
            pass

    def log_cuda_memory(device: torch.device):
        if not is_cuda_device(device):
            return
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        print(
            f"[CUDA] memory allocated: {allocated / 1024**3:.3f} GB, reserved: {reserved / 1024**3:.3f} GB"
        )
        print(
            f"[CUDA] peak allocated: {max_allocated / 1024**3:.3f} GB, peak reserved: {max_reserved / 1024**3:.3f} GB"
        )

    def handle_cuda_oom(exc: RuntimeError, device: torch.device) -> bool:
        if is_cuda_device(device) and "out of memory" in str(exc).lower():
            print("CUDA OOM detected, clearing caches and retrying on CPU.")
            reset_cuda_state(device)
            gc.collect()
            return True
        return False

    def move_model_to_device(cosyvoice, device: torch.device):
        cosyvoice.model.llm.to(device)
        cosyvoice.model.flow.to(device)
        cosyvoice.model.hift.to(device)
        cosyvoice.model.device = device
        cosyvoice.model.llm_context = (
            torch.cuda.stream(torch.cuda.Stream(device))
            if device.type == "cuda"
            else nullcontext()
        )

    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    target_device = determine_device()

    def run_capture(device: torch.device):
        torch.manual_seed(1986)
        cosyvoice = AutoModel(model_dir=model_dir)
        move_model_to_device(cosyvoice, device)

        saved_data = {}

        original_decoder = cosyvoice.model.flow.decoder.forward

        def hooked_decoder(*args, **kwargs):
            mu = kwargs.get("mu", args[0] if len(args) > 0 else None)
            if mu is not None:
                saved_data["mu"] = mu.cpu().contiguous()
                z = cosyvoice.model.flow.decoder.rand_noise[:, :, : mu.size(2)]
                saved_data["noise"] = z.cpu().contiguous()
            return original_decoder(*args, **kwargs)

        cosyvoice.model.flow.decoder.forward = hooked_decoder

        original_inference = cosyvoice.model.flow.inference

        def hooked_inference(
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            streaming,
            finalize,
        ):
            saved_data["token"] = token.cpu().contiguous()
            saved_data["prompt_token"] = prompt_token.cpu().contiguous()
            saved_data["prompt_feat"] = prompt_feat.cpu().contiguous()
            saved_data["embedding"] = embedding.cpu().contiguous()

            feat, flow_cache = original_inference(
                token,
                token_len,
                prompt_token,
                prompt_token_len,
                prompt_feat,
                prompt_feat_len,
                embedding,
                streaming,
                finalize,
            )

            saved_data["flow_output"] = feat.cpu().contiguous()
            return feat, flow_cache

        cosyvoice.model.flow.inference = hooked_inference

        try:
            with torch.no_grad():
                for _ in cosyvoice.inference_zero_shot(
                    "Hello! I am an AI voice assistant powered by Fun-CosyVoice3-0.5B. "
                    "How may I help you today?",
                    "Please speak in English.<|endofprompt|>"
                    "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. "
                    "Nothing's changed on that.",
                    "./asset/interstellar-tars-01-resemble-denoised.wav",
                    stream=False,
                ):
                    pass
        finally:
            reset_cuda_state(device)

        if "flow_output" in saved_data:
            save_path = "debug_mel.safetensors"
            save_file(saved_data, save_path)
            print(f"\nSaved Flow debug data to {save_path}")
            print(f"  token shape: {saved_data['token'].shape}")
            print(f"  prompt_feat shape: {saved_data['prompt_feat'].shape}")
        else:
            print("\nFailed to capture Flow data!")

        log_cuda_memory(device)

    try:
        run_capture(target_device)
    except RuntimeError as err:
        if handle_cuda_oom(err, target_device):
            run_capture(torch.device("cpu"))
        else:
            raise


if __name__ == "__main__":
    main()
