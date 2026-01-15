#!/usr/bin/env python3
"""Export Python's seeded rand_noise tensor for Rust parity."""

import random
import numpy as np
import torch
from safetensors.torch import save_file

# Inline set_all_random_seed(0) to avoid cosyvoice import
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
rand_noise = torch.randn([1, 80, 50 * 300])  # [1, 80, 15000]

print(f"rand_noise shape: {rand_noise.shape}")
print(f"rand_noise dtype: {rand_noise.dtype}")
print(f"rand_noise stats: min={rand_noise.min().item():.6f}, max={rand_noise.max().item():.6f}, mean={rand_noise.mean().item():.6f}")

# Save to safetensors for Rust to load
save_file({"rand_noise": rand_noise}, "pretrained_models/Fun-CosyVoice3-0.5B/rand_noise.safetensors")
print("Saved to pretrained_models/Fun-CosyVoice3-0.5B/rand_noise.safetensors")
