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

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
N = 32
color_type = 'oklab'
is_lms = False

# 3D color inputs
X_torch = torch.load(f'data/training-X-{N}-{color_type}.pt', weights_only=True).to(device)
# CMYK outputs
Y_torch = torch.load(f'data/training-Y-{N}-{color_type}.pt', weights_only=True).to(device)

X_input = X_torch
if is_lms:
  X_input = Network.oklab_to_lms_linear_torch(X_input)

# Build a single PyTorch tensor for K of shape (num_pigments, n)
K, S = FastColorMath.load_standard_K_S()

# Stack them => shape (num_pigments, n)
K_np = np.stack(K, axis=0)  # shape (4, n) for 4 pigments
S_np = np.stack(S, axis=0)  # shape (4, n)

# Convert to float32 PyTorch Tensors
K_torch = torch.from_numpy(K_np).float().to(device)  # shape (4, n)
S_torch = torch.from_numpy(S_np).float().to(device)  # shape (4, n)

# 1) Build your model
model = Network.PaintModel().to(device)

epochs = 4000
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# weight_decay = 0.01
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 4) Train for some epochs
print('Training...')

batch_size = len(X_input) // 2
print('Batch Size:', batch_size)
Network.train_cmyk_network(model, optimizer, X_input, Y_torch, K_torch, S_torch,
                    epochs=epochs, batch_size=batch_size, log_interval=10, save_checkpoints=True)

model.eval()
state = model.state_dict()
torch.save(state, 'data/model.pt')

# Convert tensors to lists for JSON serialization.
weights = {k: v.cpu().numpy().tolist() for k, v in state.items()}
with open("data/model_weights.json", "w") as f:
  json.dump(weights, f)

K, S = FastColorMath.load_standard_K_S()
np.random.seed(seed)
torch.manual_seed(seed)
print('Testing...')
with torch.no_grad():
  count = 2500
  error = 0.0
  cmyk_error = 0.0
  refl_error = 0.0
  oklab_error = 0.0
  for i in range(count):
    # or to test whole positions
    # test_rgb = np.random.randint(0, 256, size=3, dtype=np.uint8)
    # test_srgb = test_rgb / 255.0
    
    test_srgb = np.random.rand(3)
    test_oklab = FastColorMath.srgb_to_oklab(test_srgb)
    test_input = test_oklab
    if is_lms:
      test_input = FastColorMath.oklab_to_lms_linear(test_oklab)
    pred_weights = Network.predict_weights(model, test_input)
    target_weights = FastColorMath.optimize_weights(test_oklab, K, S)
    
    pred_refl = FastColorMath.mix_pigments(pred_weights, K, S)
    pred_xyz = FastColorMath.reflectance_to_xyz(pred_refl)
    pred_oklab = FastColorMath.xyz_to_oklab(pred_xyz)
    
    opt_refl = FastColorMath.mix_pigments(target_weights, K, S)
    opt_xyz = FastColorMath.reflectance_to_xyz(opt_refl)
    opt_oklab = FastColorMath.xyz_to_oklab(opt_xyz)
    
    cmyk_weight = 1.0
    refl_weight = 1.0
    oklab_weight = 1.0
    
    cur_cmyk_error = np.mean(np.square(pred_weights - target_weights))
    cur_refl_error = np.mean(np.square(pred_refl - opt_refl))
    cur_oklab_error = np.mean(np.square(pred_oklab - opt_oklab))
    
    cmyk_error += cur_cmyk_error
    refl_error += cur_refl_error
    oklab_error += cur_oklab_error
    
    # loss between weights
    error += cmyk_weight * cur_cmyk_error
    
    # loss between reflectance curves
    error += refl_weight * cur_refl_error

    # loss between colors
    error += oklab_weight * cur_oklab_error

  cmyk_error /= count
  refl_error /= count
  oklab_error /= count
  avg = error / count
  print(f"Average Total Error: {avg}")
  print(f"Average CMYK Error: {cmyk_error}")
  print(f"Average Reflectance Error: {refl_error}")
  print(f"Average OKLab Error: {oklab_error}")

