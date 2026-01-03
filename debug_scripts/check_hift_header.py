import struct
import os

p = 'pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors'
with open(p, 'rb') as f:
    header_size_bytes = f.read(8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]
    print(f"Header size: {header_size}")
    header_json = f.read(header_size).decode('utf-8')
    print("Header JSON (first 500 chars):", header_json[:500])
