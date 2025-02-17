import Network
import sys
import json
import math
import os
import numpy as np
import scipy

import FastColorMath
from PIL import Image
from Gradients import COLORS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dtype = np.float32
device = torch.device("mps")

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
N = 32
color_type = 'oklab'
K, S = FastColorMath.load_standard_K_S()
num_pigments = len(K)

# if the polynomial should evaluate to OKLab LMS (Linear) or just plain OKLab
is_lms = False

# load in the CMYK weights from our optimizations
# these will be used as training data, but we will do a forward pass
# to get the closest OKLab from these weights
Y_torch = torch.load(f'data/training-Y-{N}-{color_type}.pt', weights_only=True).to(device)

# convert torch to numpy
base_pigments = Y_torch.detach().cpu().numpy().tolist()

lerp_pigments = []
lerp_count = 48 ** 3
for i in range(lerp_count):
    w0 = np.random.dirichlet(np.ones(num_pigments, dtype=dtype))
    w1 = np.random.dirichlet(np.ones(num_pigments, dtype=dtype))
    l0 = np.concatenate((w0, np.zeros(3)))
    l1 = np.concatenate((w1, np.zeros(3)))
    t = np.random.rand()
    lerped = FastColorMath.lerp_latent(l0, l1, t)
    w = lerped[:num_pigments]
    lerp_pigments.append(w)

pigments = np.array(np.concatenate((base_pigments, lerp_pigments)), dtype=dtype)

def forward (w):
    refl = FastColorMath.forward_r(w, K, S)
    xyz = FastColorMath.reflectance_to_xyz(refl)
    if is_lms:
        return FastColorMath.xyz_to_lms_linear(xyz)
    else:
        return FastColorMath.xyz_to_oklab(xyz)

def generate_multiindices(n_vars, max_degree):
    """
    Generate all exponent tuples for n_vars variables with total degree <= max_degree.
    Returns a list of tuples of length n_vars.
    """
    # This uses a simple recursion.
    def rec(n, d):
        if n == 1:
            yield (d,)
        else:
            for i in range(d + 1):
                for tail in rec(n - 1, d - i):
                    yield (i,) + tail
    multiindices = []
    for total_degree in range(max_degree + 1):
        multiindices.extend(list(rec(n_vars, total_degree)))
    return multiindices

def design_matrix(X, multiindices):
    """
    Given input data X (shape: [N, n_vars]) and a list of multiindices,
    build the design matrix (shape: [N, n_terms]) where each column is
    prod(x_i**e_i) for the corresponding multiindex.
    """
    N, n_vars = X.shape
    n_terms = len(multiindices)
    D = np.empty((N, n_terms), dtype=np.float64)
    for j, exponents in enumerate(multiindices):
        # Compute the monomial for all samples:
        # Start with ones and multiply term-by-term.
        monom = np.ones(N, dtype=np.float64)
        for k in range(n_vars):
            # Only multiply if exponent is nonzero.
            if exponents[k] != 0:
                monom *= X[:, k] ** exponents[k]
        D[:, j] = monom
    return D

print('Building forward data...')

test_colors = [ forward(w) for w in pigments ]
print("Data Count:", len(test_colors))

degree = len(test_colors[0])
n_vars = num_pigments
multiindices = generate_multiindices(n_vars, degree)
print(f"Using {len(multiindices)} polynomial terms for degree <= {degree}.")

D = design_matrix(pigments, multiindices)  # shape: (N, n_terms)

# 4. Solve the linear regression problem: D * coeffs = test_colors
# We solve for coeffs with shape (n_terms, 3).
coeffs, residuals, rank, s = np.linalg.lstsq(D, test_colors, rcond=-1)
print("Least squares fit residuals:", residuals)

# 5. Package the coefficients and corresponding exponent tuples into a JSON structure.
# We output a list of terms; each term includes:
#    - "exponents": list of 5 integers
#    - "coeffs": an object with keys "L", "a", "b" for the three outputs.
terms = []
for j, exponents in enumerate(multiindices):
    term = [
        list(exponents),
        [coeffs[j, 0], coeffs[j, 1], coeffs[j, 2]]
    ]
    terms.append(term)

poly_model = {
    "degree": degree,
    "n_variables": n_vars,
    "terms": terms,
}

json_filename = "data/polynomial.json"
with open(json_filename, "w") as f:
    json.dump(poly_model, f, indent=2)

print(f"Polynomial model saved to {json_filename}")

def evaluate_pigment(weights, model):
    weights = np.array(weights, dtype=np.float64)
    n_vars = model["n_variables"]
    if weights.shape[0] != n_vars:
        raise ValueError(f"Expected weights array of length {n_vars} but got {len(weights)}")
    
    # Convert terms into arrays:
    # exponents: shape (n_terms, n_vars)
    # coeffs: shape (n_terms, 3)
    terms = model["terms"]
    exponents = np.array([term[0] for term in terms], dtype=np.int32)
    coeffs = np.array([term[1] for term in terms], dtype=np.float64)
    
    # Compute each term's monomial value:
    # This computes weights**exponents for each term and then takes the product along axis=1.
    monomials = np.prod(np.power(weights, exponents), axis=1)
    
    # Multiply each monomial by its coefficients and sum over terms.
    result = np.dot(monomials, coeffs)
    return result.tolist()

# another way to write the evaluation function
# def evaluate_pigment_clarity(weights, model):
#     n_vars = model["n_variables"]
#     weights = list(weights)  # ensure we have a list
#     if len(weights) != n_vars:
#         raise ValueError(f"Expected pigment array of length {n_vars} but got {len(weights)}")
#     x, y, z = 0.0, 0.0, 0.0
#     # Loop over every term in the polynomial model.
#     for term in model["terms"]:
#         monomial = 1.0
#         exponents = term[0]
#         coeffs = term[1]
#         for i in range(n_vars):
#             # Using math.pow for clarity.
#             monomial *= math.pow(weights[i], exponents[i])
#         x += coeffs[0] * monomial
#         y += coeffs[1] * monomial
#         z += coeffs[2] * monomial
#     return [x, y, z]


def generate_gradients (poly_model, neural_model, colors=COLORS, with_model=False, with_polynomial=True):
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
        w0 = Network.predict_weights(neural_model, color0_input)
        w1 = Network.predict_weights(neural_model, color1_input)
    else:
        oklab0 = color0
        oklab1 = color1
        w0 = FastColorMath.optimize_weights(oklab0, K, S)
        w1 = FastColorMath.optimize_weights(oklab1, K, S)
    
    if with_polynomial:
        mixed_oklab0 = evaluate_pigment(w0, poly_model)
        mixed_oklab1 = evaluate_pigment(w1, poly_model)
    else:
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
        # latent_r = FastColorMath.lerp(latent0, latent1, t)
        # print(latent[:pcount], latent_r[:pcount])
        
        w = latent[:pcount]
        if with_polynomial:
            oklab = evaluate_pigment(w, poly_model)
        else:
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
  mname = "neural" if with_model else "optimized"
  pname = "fit" if with_polynomial else "km"
  img.save(f"gradient-polynomial-{mname}-{pname}.png")

def test_polynomial(pigments, model):
    count = 1000
    error = 0.0

    # random shuffle of input weights
    shuffled_data = np.random.permutation(pigments)
    dataset = shuffled_data[:count]
    
    for weights in dataset:
        opt_output = forward(weights)
        if is_lms:
          opt_oklab = FastColorMath.lms_linear_to_oklab(opt_output)
        else:
          opt_oklab = opt_output

        pred_output = evaluate_pigment(weights, model)
        if is_lms:
          pred_oklab = FastColorMath.lms_linear_to_oklab(pred_output)
        else:
          pred_oklab = pred_output
        
        error += np.mean(np.square(pred_oklab - opt_oklab))
    error /= count
    print(f'Average Error: {error:.8f}')

nn_model = Network.PaintModel()
nn_model.load_state_dict(torch.load('data/model.pt', weights_only=True))
nn_model.eval()
with torch.no_grad():
    generate_gradients(poly_model, nn_model, COLORS, with_model=False, with_polynomial=False)
    generate_gradients(poly_model, nn_model, COLORS, with_model=True, with_polynomial=True)
    generate_gradients(poly_model, nn_model, COLORS, with_model=True, with_polynomial=False)
    generate_gradients(poly_model, nn_model, COLORS, with_model=False, with_polynomial=True)
    generate_gradients(poly_model, nn_model, COLORS, with_model=False, with_polynomial=False)

print('Testing...')
test_polynomial(pigments, poly_model)
