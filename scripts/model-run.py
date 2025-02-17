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
from Gradients import COLORS

dtype = np.float32
# device = torch.device("cpu")
color_type = 'oklab'
K, S = FastColorMath.load_standard_K_S()
is_lms = False

def generate_gradients (model, colors=COLORS, with_model=False):
  steps = 400
  ramp_count = len(colors)
  width = 1350
  height = steps
  scale_factor = 1

  img = Image.new('RGB', (width, height))
  pixels = img.load()
  slice_width = int(width / ramp_count)

  # Generate each color ramp
  for ramp_index, (rgb0, rgb1) in enumerate(colors):
    color0 = FastColorMath.srgb_to_oklab(np.array(rgb0,dtype=np.float32)/255.0)
    color1 = FastColorMath.srgb_to_oklab(np.array(rgb1,dtype=np.float32)/255.0)
    
    if with_model:
      color0_input = FastColorMath.oklab_to_lms_linear(color0) if is_lms else color0
      color1_input = FastColorMath.oklab_to_lms_linear(color1) if is_lms else color1
      w0 = Network.predict_weights(model, color0_input)
      w1 = Network.predict_weights(model, color1_input)
    else:
      oklab0 = color0
      oklab1 = color1
      w0 = FastColorMath.optimize_weights(oklab0, K, S)
      w1 = FastColorMath.optimize_weights(oklab1, K, S)
    
    ref0 = FastColorMath.mix_pigments(w0, K, S)
    ref1 = FastColorMath.mix_pigments(w1, K, S)
    xyz0 = FastColorMath.reflectance_to_xyz(ref0)
    xyz1 = FastColorMath.reflectance_to_xyz(ref1)
    mixed_oklab0 = FastColorMath.xyz_to_oklab(xyz0)
    mixed_oklab1 = FastColorMath.xyz_to_oklab(xyz1)
    
    mixed_color0 = mixed_oklab0
    mixed_color1 = mixed_oklab1
    r0 = color0 - mixed_color0
    r1 = color1 - mixed_color1
    
    latent0 = np.concatenate((w0, r0))
    latent1 = np.concatenate((w1, r1))
    
    # Fill each column in the ramp
    pcount = len(K)
    for i in range(steps):
        t = i / (steps - 1)
        
        latent = FastColorMath.lerp_latent(latent0, latent1, t)
        # latent = FastColorMath.lerp(latent0, latent1, t)
                
        w = latent[:pcount]
        refl = FastColorMath.mix_pigments(w, K, S)
        xyz = FastColorMath.reflectance_to_xyz(refl)
        oklab = FastColorMath.xyz_to_oklab(xyz)
        
        rterm = 1.0
        residual = latent[pcount:pcount+3]
        oklab = oklab + residual * rterm
        result = FastColorMath.oklab_to_srgb(oklab)
        
        rgb = [max(0, min(255, int(n * 255))) for n in result]
                    
        for x in range(slice_width):
            pixels[x + ramp_index * slice_width, i] = tuple(rgb)

  # Scale the image by the scale factor
  resized_width = width * scale_factor
  resized_height = height * scale_factor
  img = img.resize((resized_width, resized_height), Image.NEAREST)

  # Save the image
  fname = f"gradient-network-pigment.png"
  if with_model:
    fname = f"gradient-network-model.png"
  img.save(fname)

model = Network.PaintModel()
model.load_state_dict(torch.load('data/model.pt', weights_only=True))
model.eval()
with torch.no_grad():
  generate_gradients(model, COLORS, with_model=True)
  generate_gradients(model, COLORS, with_model=False)