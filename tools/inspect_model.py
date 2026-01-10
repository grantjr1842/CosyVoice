import sys
from safetensors.torch import load_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path>")
        return

    path = sys.argv[1]
    print(f"Loading {path}...")
    try:
        tensors = load_file(path)
        print(f"Loaded {len(tensors)} tensors.")
        for k in tensors.keys():
            print(k)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
