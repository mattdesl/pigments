import matplotlib.pyplot as plt
import FastColorMath
import numpy as np
import math

from PIL import Image
from scipy.linalg import solve
from scipy.optimize import root
import scipy
from scipy.optimize import minimize

dtype = np.float32
method = 'hybr'
num_wavelengths = len(FastColorMath.wavelengths)
T = np.array([FastColorMath.CMF_X, FastColorMath.CMF_Y, FastColorMath.CMF_Z], dtype=dtype)

COLORS = [
  [[10, 52, 162],[248, 210, 71]],
  [[0, 33, 133],[252, 210, 0]],
  [[243, 240, 247],[0, 176, 0]],
  [[ 26, 10, 83 ],[ 255, 255, 255 ]],
  [[ 0, 66, 170 ],[ 255, 255, 255 ]],
  [[195,209,23],[0,97,255]],
  [[255,170,0],[0,97,255]],
  [[255,106,0],[142,230,255]],
  [[124,42,0],[142,230,255]],
  [[0,0,0],[255,255,255]],
  [[128,128,128],[255,255,255]],
  [[128,128,128],[0,0,0]],
]

def K_S_to_R(K, S):
    """
    Converts K and S curves to R using the provided formula.
    
    Parameters:
    -----------
    K : (36,) ndarray
        Curve K.
    S : (36,) ndarray
        Curve S.
    
    Returns:
    --------
    R : (36,) ndarray
        Reflectance curve R derived from K and S.
    """
    # Ensure S is above the threshold to prevent division by zero
    S_threshold = 0.0001
    S_safe = np.maximum(S, S_threshold)
    
    K_S = K / S_safe
    R = 1.0 + K_S - np.sqrt(K_S**2 + 2.0 * K_S)
    
    # Where S < threshold, set R to 0
    R = np.where(S < S_threshold, 0.0, R)
    
    return R

def plot_KS_pigment(target_srgb, K, S):
    target_srgb = np.array(target_srgb, dtype=dtype)
    Rp = K_S_to_R(K, S)
    xyz = FastColorMath.reflectance_to_xyz(Rp)
    oklab = FastColorMath.xyz_to_oklab(xyz)
    srgb = [ max(0, min(1.0, n)) for n in FastColorMath.oklab_to_srgb(oklab) ]
    color = tuple(srgb)
    input_rgb = tuple(np.array(target_srgb).tolist())
    
    plt.style.use("dark_background")

    plt.figure(figsize=(10, 6))
    # plt.plot(wavelengths, Kp, label=f'{pigment_name} - Absorption (K)', color="gray", linestyle='--')
    plt.plot(FastColorMath.wavelengths, np.array([np.max(Rp)] * len(FastColorMath.wavelengths)), label=f'Input', color=input_rgb, linewidth=2)
    plt.plot(FastColorMath.wavelengths, K, label=f'Absorption (K)', color='gray', linewidth=3, linestyle='--')
    plt.plot(FastColorMath.wavelengths, S, label=f'Scattering (S)', color='gray', linewidth=3, linestyle=':')
    plt.plot(FastColorMath.wavelengths, Rp, label=f'Reflectance (R)', color=color, linewidth=3)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'K S Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pigment(target_srgb, Rp, R2=None):
    xyz = FastColorMath.reflectance_to_xyz(Rp)
    oklab = FastColorMath.xyz_to_oklab(xyz)
    srgb = [ max(0, min(1.0, n)) for n in FastColorMath.oklab_to_srgb(oklab) ]
    color = tuple(srgb)
    input_rgb = tuple(np.array(target_srgb).tolist())
    
    plt.style.use("dark_background")

    plt.figure(figsize=(10, 6))
    # plt.plot(wavelengths, Kp, label=f'{pigment_name} - Absorption (K)', color="gray", linestyle='--')
    plt.plot(FastColorMath.wavelengths, np.array([np.max(Rp)] * len(FastColorMath.wavelengths)), label=f'Input', color=input_rgb, linewidth=3)
    plt.plot(FastColorMath.wavelengths, Rp, label=f'Reflectance (R)', color=color, linewidth=3)
    
    if R2 is not None:
      xyz = FastColorMath.reflectance_to_xyz(R2)
      oklab = FastColorMath.xyz_to_oklab(xyz)
      srgb = [ max(0, min(1.0, n)) for n in FastColorMath.oklab_to_srgb(oklab) ]
      color = tuple(srgb)
      plt.plot(FastColorMath.wavelengths, R2, label=f'Reflectance (R2)', color=color, linewidth=3)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'R Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def smoothness_penalty(curve):
    second_diff = np.diff(curve, n=2)  # Second-order finite differences
    return np.sum(second_diff ** 2)

def LHTSS_scipy(T, sRGB):
    # Ensure T is 3x36
    T = np.asarray(T, dtype=float)
    if T.shape != (3, 36):
        raise ValueError("T must be a 3x36 matrix.")

    # Convert sRGB to float array
    sRGB = np.asarray(sRGB, dtype=float).reshape(3)
    if sRGB.shape != (3,):
        raise ValueError("sRGB must be length-3.")

    # Special cases
    # if np.allclose(sRGB, 0):
    #     return 0.0001 * np.ones(36, dtype=float)
    # if np.allclose(sRGB, 1.0):
    #     return np.ones(36, dtype=float)

    rgb = FastColorMath.srgb_to_linear_srgb(sRGB)

    # Construct difference matrix D (36x36)
    D = 4.0 * np.eye(36)
    D[0,0] = 2.0
    D[-1,-1] = 2.0
    for i in range(35):
        D[i, i+1] = -2.0
        D[i+1, i] = -2.0

    # sech helper
    def sech(x):
        return 1.0 / np.cosh(x)

    # We'll solve for v = [z_0, ..., z_35, lam_0, lam_1, lam_2].
    # Our system F(v) has length 39.
    def system_with_jac(v):
        z   = v[:36]
        lam = v[36:]

        # d0, d1, d2
        d0 = (np.tanh(z) + 1.0)/2.0
        d1 = np.diag((sech(z)**2)/2.0)
        d2 = np.diag(-sech(z)**2 * np.tanh(z))

        # F top and bottom
        F_top = D @ z + d1 @ (T.T @ lam)
        F_bottom = (T @ d0) - rgb
        F_vec = np.concatenate([F_top, F_bottom])  # shape (39,)

        # Jacobian blocks
        tmp = d2 @ (T.T @ lam)  # shape (36,)
        top_left = D + np.diag(tmp)
        top_right = d1 @ T.T
        bottom_left = T @ d1
        bottom_right = np.zeros((3,3))

        J_top = np.hstack([top_left, top_right])
        J_bot = np.hstack([bottom_left, bottom_right])
        J_mat = np.vstack([J_top, J_bot])  # shape (39,39)

        return F_vec, J_mat

    # Initial guess: z=0, lam=0
    v0 = np.zeros(39, dtype=float)

    # Solve
    sol = root(system_with_jac, v0, jac=True, method=method)
    if not sol.success:
        print("Warning: SciPy root did not converge. Info:", sol.message)

    # Extract solution
    z_solution = sol.x[:36]
    # lam_solution = sol.x[36:]  # not needed for final reflectance

    # Final reflectance
    rho = (np.tanh(z_solution) + 1.0)/2.0
    return rho


def optimize_reflectance(target_srgb, R_initial=None, smoothness=0.1):
    target_srgb = np.asarray(target_srgb, dtype=dtype)
    
    # Handle special cases where sRGB is pure black or white
    # if np.allclose(target_srgb, 0):
    #     return 0.0001 * np.ones(36)
    # if np.allclose(target_srgb, 1.0):
    #     return np.ones(36)
    
    # Convert target sRGB to Oklab using FastColorMath utilities
    # This assumes FastColorMath has a function to convert sRGB directly to Oklab
    target_oklab = FastColorMath.srgb_to_oklab(target_srgb)
    
    # Define the objective function
    def objective(R):
        """
        Objective function to minimize: perceptual difference in Oklab + smoothness regularization.
        
        Parameters:
        -----------
        R : (36,) ndarray
            Current reflectance curve being evaluated.
        
        Returns:
        --------
        float
            The computed objective value.
        """
        # Convert R to XYZ
        XYZ = FastColorMath.reflectance_to_xyz(R)
        
        # Convert XYZ to Oklab
        oklab_output = FastColorMath.xyz_to_oklab(XYZ)
        
        # Compute perceptual difference in Oklab
        perceptual_diff = FastColorMath.oklab_diff_sq(target_oklab, oklab_output)
        
        # return perceptual_diff + 0.001 * smoothness_penalty(R)
        # Compute smoothness term: sum of squared differences between adjacent reflectance values
        smooth = np.sum((R[1:] - R[:-1])**2)
        
        # Total objective: perceptual difference + regularization
        return perceptual_diff + smooth * smoothness
    
    # Initial guess: reflectance set to mid-value (0.5) for all wavelengths
    if R_initial is None:
      R_initial = 0.5 * np.ones(36)
    
    # Define bounds for R: each reflectance value must be within (0, 1)
    bounds = [(0.00001, 1.0) for _ in range(36)]
    
    # Perform the optimization using L-BFGS-B algorithm
    result = minimize(
        objective,
        R_initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            # 'maxiter': 1000,
            # 'ftol': 1e-9,
        }
    )
    
    # Check if the optimization was successful
    if not result.success:
        print("Optimization did not converge:", result.message)
    
    # Extract the optimized reflectance curve
    R_opt = result.x
    
    return R_opt

def optimize_K_S(target_srgb, target_R, lambda_reg=0.0):
    target_srgb = np.asarray(target_srgb, dtype=dtype)
    
    # Handle special cases where sRGB is pure black or white
    # if np.allclose(target_srgb, 0):
    #     return None, None, 0.0001 * np.ones(36)
    # if np.allclose(target_srgb, 1.0):
    #     return None, None, np.ones(36)
    
    target_srgb_normalized = target_srgb
    target_oklab = FastColorMath.srgb_to_oklab(target_srgb_normalized)
    
    # Define the objective function
    def objective(x):
        """
        Objective function to minimize: perceptual difference in Oklab + smoothness regularization.
        
        Parameters:
        -----------
        x : (72,) ndarray
            Optimization variables [K_0, ..., K_35, S_0, ..., S_35].
        
        Returns:
        --------
        float
            The computed objective value.
        """
        K = x[:36]
        S = x[36:]
        
        R = K_S_to_R(K, S)
        
        # Convert R to XYZ
        XYZ = FastColorMath.reflectance_to_xyz(R)
        
        # Convert XYZ to Oklab
        oklab_output = FastColorMath.xyz_to_oklab(XYZ)
        
        # Compute perceptual difference in Oklab
        perceptual_diff = FastColorMath.oklab_diff_sq(target_oklab, oklab_output)
        
        # Compute smoothness terms for K and S
        smoothness_K = np.sum((K[1:] - K[:-1])**2)
        smoothness_S = np.sum((S[1:] - S[:-1])**2)
        
        # Total objective: perceptual difference + regularization
        total_objective = perceptual_diff + lambda_reg * (smoothness_K + smoothness_S)
        
        # difference from target R reflectance curve
        total_objective += np.sum((R - target_R) ** 2)
        
        # penalty to ensure that K is always a larger value than S
        # it should not introduce a penalty if K is already larger
        # total_objective += np.sum(np.maximum(0, S - K))
        
        
        return total_objective
    
    # Initial guess: K and S set to 1.0 for all wavelengths
    K_initial = np.array([0.5] * 36,dtype=dtype)
    S_initial = np.zeros(36)
    x0 = np.concatenate([K_initial, S_initial])
    
    # Define bounds for K and S
    # K >=0, S >=0.0001 to prevent division by zero
    bounds_K = [(0.0001, 1.0) for _ in range(36)]
    bounds_S = [(0.0001, 1.0) for _ in range(36)]
    bounds = bounds_K + bounds_S
    
    # Perform the optimization using L-BFGS-B algorithm
    result = minimize(
        fun=objective,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 1000,
            'ftol': 1e-9,
            # 'disp': True  # Set to True to see convergence messages
        }
    )
    
    # Check if the optimization was successful
    if not result.success:
        print("Optimization 2 did not converge:", result.message)
    
    # Extract the optimized K and S
    K_opt = result.x[:36]
    S_opt = result.x[36:]
    
    # Derive the optimized R
    # R_opt = K_S_to_R(K_opt, S_opt)
    
    return K_opt, S_opt

def to_K_S (srgb):
  srgb = np.array(srgb, dtype=dtype)
  R1 = LHTSS_scipy(T, srgb)
  R2 = optimize_reflectance(srgb, R1, 0.01)
  result = optimize_K_S(srgb, target_R=R2, lambda_reg=0.01)
  print(result)
  return result

def generate_gradients (colors=COLORS):
  steps = 400
  ramp_count = len(colors)
  width = 1350
  height = steps
  scale_factor = 1

  types = [ 'optim' ]

  for curType in types:
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    slice_width = int(width / ramp_count)

    # Generate each color ramp
    for ramp_index, (rgb0, rgb1) in enumerate(colors):
        # print(ramp_index)
        # K = FastColorMath.K
        # S = FastColorMath.S
        # col0 = FastColorMath.optimize_rgb_bytes(rgb0, K, S)
        # col1 = FastColorMath.optimize_rgb_bytes(rgb1, K, S)
        # print('weights')
        # print(col0['weights'].tolist())
        # print(col1['weights'].tolist())
        
        # print('oklabs')
        # print(col0['oklab'].tolist())
        # print(col1['oklab'].tolist())
        
        srgb0 = np.array(rgb0, dtype=dtype) / 255.0
        srgb1 = np.array(rgb1, dtype=dtype) / 255.0
        
        print(srgb0, srgb1)
        K0, S0 = to_K_S(srgb0)
        K1, S1 = to_K_S(srgb1)
        
        
        # Fill each column in the ramp
        for i in range(steps):
            t = i / (steps - 1)
            
            weights = np.array([1.0 - t, t])
            K = np.array([K0, K1], dtype=dtype)
            S = np.array([S0, S1], dtype=dtype)
            mixed_r = FastColorMath.mix_pigments(weights, K, S)
            mixed_xyz = FastColorMath.reflectance_to_xyz(mixed_r)
            mixed_oklab = FastColorMath.xyz_to_oklab(mixed_xyz)
            result = FastColorMath.oklab_to_srgb(mixed_oklab)
            
            rgb = [max(0, min(255, int(n * 255))) for n in result]
                        
            for x in range(slice_width):
                pixels[x + ramp_index * slice_width, i] = tuple(rgb)

    # Scale the image by the scale factor
    resized_width = width * scale_factor
    resized_height = height * scale_factor
    img = img.resize((resized_width, resized_height), Image.NEAREST)

    # Save the image
    img.save(f"gradient-{curType}.png")

generate_gradients()

# input_rgb = np.random.randint(0, 256, 3)
# # input_rgb = [0, 50, 54]
# input_srgb = np.array(input_rgb, dtype=np.float32) / 255.0
# print('input', input_rgb)

# R1 = LHTSS_scipy(T,input_srgb)
# R2 = optimize_reflectance(input_srgb, R1, 0.01)
# K, S = optimize_K_S(input_srgb, target_R=R2, lambda_reg=0.01)
# R = K_S_to_R(K, S)


# srgb = FastColorMath.oklab_to_srgb(FastColorMath.xyz_to_oklab(FastColorMath.reflectance_to_xyz(R)))
# # print('curve', R)

# rgb = np.asarray(np.round(srgb * 255.0)).astype(np.uint8)
# print('result', rgb)


# # K, S = FastColorMath.load_standard_K_S()
# plot_KS_pigment(input_srgb, K, S)
# plot_pigment(input_srgb, R, R2)
