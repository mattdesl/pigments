import Network
import sys
import json
import math
import os
import numpy as np
import scipy

import FastColorMath
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dtype = np.float32
device = torch.device("cpu")

def generate_grid(K, S, N=32, color_type = 'oklab'):
    # 1) Generate grid of sRGB values in [0..1].
    #    We want N discrete values along each axis, from 0..1 inclusive.
    #    (If you specifically want 0..255, just multiply afterwards.)
    rgb_vals = np.linspace(0.0, 255.0, N, endpoint=True)  # shape (N,)
    
    # 2) Create the full 3D grid, shape (N, N, N, 3)
    r, g, b = np.meshgrid(rgb_vals, rgb_vals, rgb_vals, indexing='ij')
    rgb_grid = np.stack([r, g, b], axis=-1)  # shape = (N, N, N, 3)
    
    # Flatten to (N^3, 3) for convenience in passing to srgb_to_oklab
    rgb_flat = rgb_grid.reshape(-1, 3)

    # 3) Convert to OKLab in a single vectorized pass
    # oklab_flat = FastColorMath.srgb_to_oklab(rgb_flat / 255.0)  # shape (N^3, 3)
    srgb = rgb_flat / 255.0
    
    if color_type == 'oklab':
      return FastColorMath.srgb_to_oklab(srgb)
    elif color_type == 'srgb-linear':
      return FastColorMath.srgb_to_linear_srgb(srgb)
    elif color_type == 'srgb':
      return srgb
    else:
      raise ValueError('Unknown color type')

N = 32

# Build a single PyTorch tensor for K of shape (num_pigments, n)
K, S = FastColorMath.load_standard_K_S()
color_type = 'oklab'
color_grid = generate_grid(K, S, N, color_type)

if color_type == 'srgb':
  oklab_grid = FastColorMath.srgb_to_oklab(color_grid)
elif color_type == 'srgb-linear':
  oklab_grid = FastColorMath.linear_srgb_to_oklab(color_grid)
else:
  oklab_grid = color_grid

color_tensor = torch.tensor(color_grid, dtype=torch.float32).to(device)

print("Gathering training data...")
X_torch = color_tensor
print('Pixel Count:', len(X_torch))

num_pixels = len(X_torch)
num_pigments = len(K)
Y_list = np.zeros((num_pixels, num_pigments), dtype=np.float32)
for i in range(num_pixels):
  oklab = oklab_grid[i]
  y = FastColorMath.optimize_weights(oklab, K, S)
  Y_list[i] = y
  if i % 1000 == 0:
    print(f"Progress: {i+1}/{num_pixels}")

Y_torch = torch.tensor(Y_list, dtype=torch.float32).to(device)

torch.save(X_torch, f'data/training-X-{N}-{color_type}.pt')
torch.save(Y_torch, f'data/training-Y-{N}-{color_type}.pt')