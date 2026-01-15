
import torch
import sys
import os
from safetensors.torch import load_file, save_file
from hyperpyyaml import load_hyperpyyaml

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(repo_root)

    from cosyvoice.cli.cosyvoice import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load inputs
    debug_path = os.path.join(repo_root, "rust_flow_debug.safetensors")
    if not os.path.exists(debug_path):
        print(f"Error: {debug_path} not found.")
        return

    inputs = load_file(debug_path)
    # Shapes: [1, 80, 782] usually.
    mu = inputs["mu"].to(device)
    mask = inputs["mask"].to(device)
    spks = inputs["spks"].to(device)
    cond = inputs["cond"].to(device)
    x_init = inputs["x_init"].to(device) # This is x at t=0 (pre-scaled by temp?)
    # Capture script uses x_init = noise * temp.
    # We will run DiT one step at t=0.

    # Load model
    model_dir = os.path.join(repo_root, "pretrained_models/Fun-CosyVoice3-0.5B")
    with open(os.path.join(model_dir, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})

    decoder = configs["flow"].decoder
    decoder.to(device)
    decoder.eval()

    flow_weights = load_file(os.path.join(model_dir, "flow.safetensors"))
    configs["flow"].load_state_dict(flow_weights, strict=False)
    print("Model loaded.")

    dit = decoder.estimator

    # Prepare inputs for DiT.forward
    # x needs transpose [B, 80, T] -> [B, T, 80] ?
    # DiT.forward takes [B, 80, T] and transposes internally?
    # Let's check DiT.forward signature/implementation:
    # def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
    #     x = x.transpose(1, 2)
    #     mu = mu.transpose(1, 2)
    #     cond = cond.transpose(1, 2)

    # So we pass [B, 80, T].
    x = x_init.clone()
    # t=0
    t = torch.tensor([0.0], device=device).float()

    print("--- Module IDs ---")
    for i, block in enumerate(dit.transformer_blocks):
        print(f"  Block {i} attn_norm.norm id: {id(block.attn_norm.norm)}")
        print(f"  Block {i} ff_norm.norm id: {id(block.ff_norm)}")
    print(f"  Norm Out norm id: {id(dit.norm_out.norm)}")
    print("------------------")

    def tensor_stats(t, name, module=None, input_shape=None):
        mid = f" [id={id(module)}, type={type(module).__name__}]" if module else ""
        ishape = f", in_shape={list(input_shape)}" if input_shape else ""
        print(f"    [{name}]{mid}{ishape} stats: min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}, shape={list(t.shape)}")
        if t.ndim == 3:
            # Stats for first frame [0, 0, :]
            frame = t[0, 0, :]
            print(f"        Frame 0: mean={frame.mean():.6f}, std={frame.std(unbiased=False):.6f}, first 5={frame[:5].tolist()}")

    # Hooks
    layer_outputs = {}

    def hook_input_embed(module, args, output):
        t = output.detach().cpu().contiguous()
        layer_outputs["input_embed_out"] = t
        tensor_stats(t, "Input Embed", module, args[0].shape)

    def hook_time_embed(module, args, output):
        t = output.detach().cpu().contiguous()
        layer_outputs["time_embed_out"] = t
        tensor_stats(t, "Time Embed", module, args[0].shape)

    def hook_block0(module, args, output):
        t = output.detach().cpu().contiguous()
        layer_outputs["block0_out"] = t
        tensor_stats(t, "Block 0", module, args[0].shape)

    def hook_block7(module, args, output):
        t = output.detach().cpu().contiguous()
        layer_outputs["block7_out"] = t
        tensor_stats(t, "Block 7", module, args[0].shape)

    def hook_norm_out_pre(module, args, output):
        x = args[0]
        # output is norm(x)
        print(f"    [HOOK DEBUG] Norm Out Norm Input Shape: {x.shape}")
        print(f"    [HOOK DEBUG] Norm Out Norm Input Frame 0: mean={x[0,0,:].mean():.6f}, std={x[0,0,:].std(unbiased=False):.6f}")
        print(f"    [HOOK DEBUG] Norm Out Norm Output Shape: {output.shape}")
        print(f"    [HOOK DEBUG] Norm Out Norm Output Frame 0: mean={output[0,0,:].mean():.6f}, std={output[0,0,:].std(unbiased=False):.6f}")

        # Manually compute
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        eps = 1e-6
        manual = (x - mean) / torch.sqrt(var + eps)

        diff = (output - manual).abs().max()
        print(f"    [Python LN Debug] Manual vs Module Max Diff: {diff:.8e}")

        t = output.detach().cpu().contiguous()
        layer_outputs["norm_out_norm_out"] = t
        layer_outputs["norm_out_mean"] = x.mean(-1).detach().cpu().contiguous()
        layer_outputs["norm_out_var"] = x.var(-1, unbiased=False).detach().cpu().contiguous()
        tensor_stats(t, "Norm Out (Standardized)", module, x.shape)

    def hook_norm_out(module, args, output):
        t = output.detach().cpu().contiguous()
        layer_outputs["norm_out_out"] = t
        tensor_stats(t, "Norm Out (Final)", module, args[0].shape)

    dit.input_embed.register_forward_hook(hook_input_embed)
    dit.time_embed.register_forward_hook(hook_time_embed)
    dit.transformer_blocks[0].register_forward_hook(hook_block0)
    dit.transformer_blocks[-1].register_forward_hook(hook_block7)
    dit.norm_out.norm.register_forward_hook(hook_norm_out_pre)
    dit.norm_out.register_forward_hook(hook_norm_out)

    print("Running DiT forward...")
    with torch.inference_mode():
        _ = dit(x, mask, mu, t, spks, cond)

    save_path = os.path.join(repo_root, "debug_flow_layers.safetensors")
    save_file(layer_outputs, save_path)
    print(f"Saved layer outputs to {save_path}")
    for k, v in layer_outputs.items():
        print(f"  {k}: {v.shape}, mean={v.mean():.6f}")

if __name__ == "__main__":
    main()
