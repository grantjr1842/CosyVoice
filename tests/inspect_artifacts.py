import os
import sys

from safetensors.torch import load_file


def main():
    path = "tests/test_artifacts.safetensors"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    if not os.path.exists(path):
        print(f"Artifacts not found at {path}")
        return

    data = load_file(path)
    for k, v in data.items():
        print(f"{k}: {v.shape}, {v.dtype}")
        if "flow_output" in k or "x_init" in k or "flow_feat" in k:
            print(
                f"  mean={v.float().mean()}, std={v.float().std()}, min={v.float().min()}, max={v.float().max()}"
            )


if __name__ == "__main__":
    main()
