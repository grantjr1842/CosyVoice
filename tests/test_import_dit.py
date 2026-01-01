import os
import sys

from safetensors.torch import load_file

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party/Matcha-TTS"))

from cosyvoice.flow.transformer import DiT


def main():
    print("Imported DiT.")

    # Try to instantiate
    print("Instantiating DiT...")
    model = DiT(
        dim=512,  # guess from config or use defaults
        depth=6,
        heads=8,
        dim_head=64,
        ff_mult=4,
        mel_dim=80,
    )
    print("Instantiated DiT.")

    # Try to load weights from safe tensor if possible, or just print structure
    print(model.proj_out)
    print("proj_out bias:", model.proj_out.bias)

    # Check if we can load full flow model
    print("Loading flow.safetensors weights...")
    try:
        sd = load_file("pretrained_models/Fun-CosyVoice3-0.5B/flow.safetensors")
        print("Loaded weights.")
        keys = list(sd.keys())
        print("First 5 keys:", keys[:5])

        # Check proj_out.bias in weights
        proj_bias_key = "decoder.estimator.proj_out.bias"
        if proj_bias_key in sd:
            print(f"{proj_bias_key}:", sd[proj_bias_key])
            print(
                "Stats:",
                sd[proj_bias_key].min(),
                sd[proj_bias_key].max(),
                sd[proj_bias_key].mean(),
            )
        else:
            print(f"{proj_bias_key} not found in weights.")

    except Exception as e:
        print("Failed to load weights:", e)


if __name__ == "__main__":
    main()
