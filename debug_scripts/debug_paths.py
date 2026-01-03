import os
from pathlib import Path

paths_to_check = [
    'pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors',
    'pretrained_models/Fun-CosyVoice3-0.5B/hift.pt',
    'pretrained_models/Fun-CosyVoice3-0.5B/flow.safetensors',
    'pretrained_models/Fun-CosyVoice3-0.5B/llm.safetensors'
]

for p in paths_to_check:
    abs_p = os.path.abspath(p)
    exists = os.path.exists(abs_p)
    is_file = os.path.isfile(abs_p)
    is_dir = os.path.isdir(abs_p)
    print(f"Path: {p}")
    print(f"  Abs: {abs_p}")
    print(f"  Exists: {exists}, IsFile: {is_file}, IsDir: {is_dir}")
    if exists and is_dir:
        print(f"  Contents: {os.listdir(abs_p)}")
