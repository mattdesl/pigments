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
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# wavelengths, pigments, paints = load_data()

CMF_X_spectrum_torch = torch.tensor(FastColorMath.CMF_X, dtype=torch.float32).to(device)
CMF_Y_spectrum_torch = torch.tensor(FastColorMath.CMF_Y, dtype=torch.float32).to(device)
CMF_Z_spectrum_torch = torch.tensor(FastColorMath.CMF_Z, dtype=torch.float32).to(device)

XYZ_D65_to_LMS_torch = torch.tensor(FastColorMath.XYZ_D65_to_LMS, dtype=torch.float32).to(device)
LMS_TO_OKLAB_torch = torch.tensor(FastColorMath.LMS_TO_OKLAB, dtype=torch.float32).to(device)
OKLAB_TO_LMS_torch = torch.tensor(FastColorMath.OKLAB_TO_LMS, dtype=torch.float32).to(device)
LMS_TO_RGB_torch = torch.tensor(FastColorMath.LMS_TO_RGB, dtype=torch.float32).to(device)
RGB_TO_LMS_torch = torch.tensor(FastColorMath.RGB_TO_LMS, dtype=torch.float32).to(device)

EPSILON = FastColorMath.EPSILON
GAMMA = FastColorMath.GAMMA

K, S = FastColorMath.load_standard_K_S()
N_PIGMENTS = len(K)

mse_loss = nn.MSELoss()


# model.to(device)
# K_torch = K_torch.to(device)
# S_torch = S_torch.to(device)
# X_torch = X_torch.to(device)
# Y_torch = Y_torch.to(device)
# CMF_X_spectrum_torch = CMF_X_spectrum_torch.to(device)
# CMF_Y_spectrum_torch = CMF_Y_spectrum_torch.to(device)
# CMF_Z_spectrum_torch = CMF_Z_spectrum_torch.to(device)
# XYZ_D65_to_LMS_torch = XYZ_D65_to_LMS_torch.to(device)
# LMS_TO_OKLAB_torch = LMS_TO_OKLAB_torch.to(device)
# os.environ["PYTOCH_ENABLE_MPS_FALLBACK"] = "1"

def mix_pigments_torch(weights, K_torch, S_torch, alpha=GAMMA, epsilon=EPSILON):
    """
    Mix pigments using nonlinear blending (power-mean) of K and S coefficients.
    
    Parameters:
      weights: Tensor of shape (batch_size, num_pigments)
      K_torch: Tensor of shape (num_pigments, n_waves)
      S_torch: Tensor of shape (num_pigments, n_waves)
      alpha: Exponent for nonlinear blending. (alpha=1 gives linear blending)
      epsilon: Small constant to avoid division by zero.
      
    Returns:
      mixed_R: Tensor of mixed reflectance, shape (batch_size, n_waves)
    """
    # 1) Normalize weights (batch_size, num_pigments)
    sum_w = weights.sum(dim=1, keepdim=True)
    comp_weights = weights / (sum_w + epsilon)
    
    if alpha == 1.0:
        # Linear blending: (batch_size, num_pigments) @ (num_pigments, n_waves)
        mixed_K = comp_weights @ K_torch
        mixed_S = comp_weights @ S_torch
    else:
        # Nonlinear blending:
        # Compute nonlinear weights: (w^alpha) and renormalize (still shape: batch_size x num_pigments)
        nonlin_weights = comp_weights ** alpha
        nonlin_weights = nonlin_weights / (nonlin_weights.sum(dim=1, keepdim=True) + epsilon)
        
        # Blend K and S in a vectorized way.
        # K_torch: (num_pigments, n_waves) -> unsqueeze to (1, num_pigments, n_waves)
        # nonlin_weights: (batch_size, num_pigments) -> unsqueeze to (batch_size, num_pigments, 1)
        weighted_K = nonlin_weights.unsqueeze(2) * (K_torch.unsqueeze(0) ** alpha)
        weighted_S = nonlin_weights.unsqueeze(2) * (S_torch.unsqueeze(0) ** alpha)
        
        # Sum over pigments (axis 1) then take 1/alpha power: result shape (batch_size, n_waves)
        mixed_K = torch.pow(torch.sum(weighted_K, dim=1), 1.0 / alpha)
        mixed_S = torch.pow(torch.sum(weighted_S, dim=1), 1.0 / alpha)
    
    # Clamp to non-negative values
    mixed_K = torch.clamp(mixed_K, min=0.0)
    mixed_S = torch.clamp(mixed_S, min=0.0)
    
    # Compute ratio and reflectance R for each wavelength.
    K_S = mixed_K / (mixed_S + epsilon)
    mixed_R = 1.0 + K_S - torch.sqrt(K_S**2 + 2.0 * K_S)
    
    return mixed_R

def mix_pigments_torch_og(weights, K_torch, S_torch):
  # 1) Sum of weights (to normalize them if needed)
  sum_w = weights.sum(dim=1, keepdim=True)  # shape (batch_size, 1)

  # 2) Normalized weights => shape (batch_size, num_pigments)
  comp_weights = weights / sum_w

  # 3) Compute mixed_K and mixed_S by weighted sum
  # Each is shape (batch_size, n_waves).
  # We'll do a matrix multiply: (batch_size, num_pigments) @ (num_pigments, n_waves)
  mixed_K = comp_weights @ K_torch  # shape (batch_size, n_waves)
  mixed_S = comp_weights @ S_torch  # shape (batch_size, n_waves)

  # 4) K_S = mixed_K / mixed_S, shape (batch_size, n_waves)
  K_S = mixed_K / mixed_S

  # 5) mixed_R = 1.0 + K_S - sqrt(K_S^2 + 2*K_S), all in PyTorch
  mixed_R = 1.0 + K_S - torch.sqrt(K_S**2 + 2.0 * K_S)

  return mixed_R

def reflectance_to_xyz_torch(reflectance):
  # If reflectance is 1D, make it 2D for consistent batch logic
  if reflectance.dim() == 1:
    reflectance = reflectance.unsqueeze(0)  # shape (1, n_waves)

  # Dot product with X_spectrum_torch, etc. We can do elementwise * and sum.
  X = torch.sum(CMF_X_spectrum_torch * reflectance, dim=1)
  Y = torch.sum(CMF_Y_spectrum_torch * reflectance, dim=1)
  Z = torch.sum(CMF_Z_spectrum_torch * reflectance, dim=1)

  # shape (batch_size, 3)
  xyz = torch.stack([X, Y, Z], dim=1)

  # If we originally had a single reflectance, return shape (3,)
  if xyz.shape[0] == 1:
    return xyz[0]
  else:
    return xyz
    
def cbrt_torch(x):
  return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)

def xyz_to_oklab_torch(xyz):
  """
  xyz: shape (batch_size,3) or (3,)

  returns: shape (batch_size,3) or (3,)
  """
  # If 1D, unsqueeze to (1,3)
  if xyz.dim() == 1:
    xyz = xyz.unsqueeze(0)

  # lms = xyz @ XYZ_D65_to_LMS_torch^T
  # shape => (batch_size, 3)
  lms = xyz @ XYZ_D65_to_LMS_torch.T
  
  # lms^(1/3)
  lms_ = cbrt_torch(lms)

  # final = lms_ @ LMS_TO_OKLAB_torch^T
  oklab = lms_ @ LMS_TO_OKLAB_torch.T

  # shape => (batch_size,3)
  if oklab.shape[0] == 1:
    return oklab[0]
  else:
    return oklab

def linear_srgb_to_oklab_torch(lsrgb):
  # If 1D, unsqueeze to (1,3)
  if lsrgb.dim() == 1:
    lsrgb = lsrgb.unsqueeze(0)
    
  lms = lsrgb @ RGB_TO_LMS_torch.T
  lms_ = cbrt_torch(lms)
  oklab = lms_ @ LMS_TO_OKLAB_torch.T
  
  # shape => (batch_size,3)
  if oklab.shape[0] == 1:
    return oklab[0]
  else:
    return oklab
  
def oklab_to_lms_linear_torch(oklab):
  if oklab.dim() == 1:
    oklab = oklab.unsqueeze(0)

  lms = oklab @ OKLAB_TO_LMS_torch.T
  
  if lms.shape[0] == 1:
    return lms[0]
  else:
    return lms

def oklab_to_linear_srgb_torch(oklab):
  """
  oklab: shape (batch_size,3) or (3,)

  returns: shape (batch_size,3) or (3,)
  """
  # If 1D, unsqueeze to (1,3)
  if oklab.dim() == 1:
    oklab = oklab.unsqueeze(0)

  # lms = xyz @ XYZ_D65_to_LMS_torch^T
  # shape => (batch_size, 3)
  lms = oklab @ OKLAB_TO_LMS_torch.T
  
  # lms^3
  lms_ = lms ** 3.0

  # final = lms_ @ LMS_TO_OKLAB_torch^T
  linear_srgb = lms_ @ LMS_TO_RGB_torch.T

  # shape => (batch_size,3)
  if linear_srgb.shape[0] == 1:
    return linear_srgb[0]
  else:
    return linear_srgb

def linear_srgb_to_srgb_torch(lin):
  # If 1D, unsqueeze to (1,3)
  if lin.dim() == 1:
    lin = lin.unsqueeze(0)
  abs_lin = torch.abs(lin)
  abs_gam = torch.where(
      abs_lin <= 0.0031308,
      12.92 * abs_lin,
      1.055 * torch.pow(abs_lin, 1/2.4) - 0.055
  )
  srgb = torch.sign(lin) * abs_gam
  # shape => (batch_size,3)
  if srgb.shape[0] == 1:
    return srgb[0]
  else:
    return srgb

def srgb_to_linear_srgb_torch(gam):
  # If 1D, unsqueeze to (1,3)
  if gam.dim() == 1:
    gam = gam.unsqueeze(0)
  abs_gam = torch.abs(gam)
  abs_lin = torch.where(
      abs_gam <= 0.040449936,
      abs_gam / 12.92,
      torch.pow((abs_gam + 0.055) / 1.055, 2.4)
  )
  lsrgb = torch.sign(gam) * abs_lin
  if lsrgb.shape[0] == 1:
    return lsrgb[0]
  else:
    return lsrgb

def srgb_to_oklab_torch(c):
  return linear_srgb_to_oklab_torch(srgb_to_linear_srgb_torch(c))
  
def oklab_to_srgb_torch(c):
  return linear_srgb_to_srgb_torch(oklab_to_linear_srgb_torch(c))

def oklab_diff_sq_torch(oklab0, oklab1):
  ab_factor = 2.0

  # Ensure both have a batch dim
  if oklab0.dim() == 1:
      oklab0 = oklab0.unsqueeze(0)
  if oklab1.dim() == 1:
      oklab1 = oklab1.unsqueeze(0)

  dL = oklab0[:,0] - oklab1[:,0]
  da = (oklab0[:,1] - oklab1[:,1]) * ab_factor
  db = (oklab0[:,2] - oklab1[:,2]) * ab_factor

  dist_sq = dL * dL + da * da + db * db

  # If we had a single sample, return just a scalar
  if dist_sq.shape[0] == 1:
      return dist_sq[0]
  return dist_sq

def train_cmyk_network(model, optimizer, X_color, Y_cmyk, K_torch, S_torch, epochs=10, batch_size=32, log_interval=10, save_checkpoints=False):
    model.train()
    N = X_color.shape[0]
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        # Shuffle the dataset at the start of each epoch.
        perm = torch.randperm(N)
        X_shuffled = X_color[perm]
        Y_shuffled = Y_cmyk[perm]

        total_loss = 0.0
        num_batches = (N + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            x_batch = X_shuffled[start:end]
            y_batch = Y_shuffled[start:end]

            optimizer.zero_grad()
            
            # Inject noise into the inputs
            noise_std = 0.001
            noise = torch.randn_like(x_batch) * noise_std
            x_batch_noisy = x_batch + noise

            # Forward pass: predict CMYK weights from linear sRGB inputs.
            pred_weights = model(x_batch_noisy)  # shape (batch_size, 4)

            # Compute the loss (MSE between predicted and target CMYK weights)
            cmyk_loss = mse_loss(pred_weights, y_batch)
            
            # compute the loss between reflectance curves
            pred_ref = mix_pigments_torch(pred_weights, K_torch, S_torch)
            y_ref = mix_pigments_torch(y_batch, K_torch, S_torch)
            refl_loss = mse_loss(pred_ref, y_ref)

            # compute the loss between final oklab colors
            pred_oklab = xyz_to_oklab_torch(reflectance_to_xyz_torch(pred_ref))
            y_oklab = xyz_to_oklab_torch(reflectance_to_xyz_torch(y_ref))
            oklab_loss = mse_loss(pred_oklab, y_oklab)
            
            # print(cmyk_loss.item(), refl_loss.item(), oklab_loss.item())
            cmyk_weight = 1.0
            refl_weight = 1.0
            oklab_weight = 1.0
            loss = cmyk_loss * cmyk_weight + refl_loss * refl_weight + oklab_loss * oklab_weight

            # Backpropagation and optimization.
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end - start)

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Batch [{batch_idx+1}/{num_batches}], "
                    f"Loss: {loss.item():.6f}"
                )

        epoch_loss = total_loss / N
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.6f}")

        if save_checkpoints and (epoch + 1) % 100 == 0:
            checkpoint_path = f'data/model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

class PaintModel(nn.Module):
    def __init__(self):
        super(PaintModel, self).__init__()
        hidden_size = 16
        input_size = 3
        output_size = N_PIGMENTS
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

    @property
    def device(self):
        return next(self.parameters()).device

def predict_weights (model, color_input):
  device = model.device
  color_input = torch.tensor(color_input, dtype=torch.float32).unsqueeze(0).to(device)
  weights = FastColorMath.normalized(model(color_input).cpu().detach().numpy()[0])
  return weights

def predict (model, lsrgb, K, S):
  rgbf = FastColorMath.linear_srgb_to_srgb(lsrgb)
  input_oklab = FastColorMath.srgb_to_oklab(rgbf)
  weights = predict_weights(model, lsrgb)
  mixed_R = FastColorMath.mix_pigments(weights, K, S)
  mixed_xyz = FastColorMath.reflectance_to_xyz(mixed_R)
  mixed_oklab = FastColorMath.xyz_to_oklab(np.array(mixed_xyz))
  mixed_srgb = FastColorMath.oklab_to_srgb(mixed_oklab)
  return {
    'oklab': mixed_oklab,
    'srgb': mixed_srgb,
    'residual': mixed_oklab - input_oklab,
    'weights': weights
  }

def predict_rgb_bytes (model, rgb, K, S):
  rgbf = np.array(rgb, dtype) / 255.0
  lsrgb = FastColorMath.srgb_to_linear_srgb(rgbf)
  # oklab = FastColorMath.srgb_to_oklab(rgbf)
  return predict(model, lsrgb, K, S)
