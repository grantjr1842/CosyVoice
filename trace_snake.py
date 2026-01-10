import torch
import numpy as np

def snake_py(x, alpha):
    return x + 1 / alpha * torch.sin(alpha * x)**2

x = torch.tensor([1.3323], dtype=torch.float32)
alpha = torch.tensor([-0.6865], dtype=torch.float32)

y = snake_py(x, alpha)
print(f"Snake(1.3323, -0.6865) = {y.item():.8f}")
