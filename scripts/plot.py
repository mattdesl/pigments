import matplotlib.pyplot as plt
# from FastColorMath import K, S, set_pigment_K_S, mix_pigments, optimize_rgb_bytes, mix_K_S, wavelengths, K_S_to_R, reflectance_to_xyz, xyz_to_oklab, oklab_to_srgb
import numpy as np
from scipy.interpolate import CubicSpline
import FastColorMath
import mixbox
# (20, 7, 63)
# (245, 205, 9)
# (190, 12, 0)
# (254, 255, 254)
# print(mixbox.latent_to_rgb([1, 0, 0, 0, 0, 0, 0])) # (20, 7, 63) blue aka cyan
# print(mixbox.latent_to_rgb([0, 1, 0, 0, 0, 0, 0])) # (245, 205, 9) yellow
# print(mixbox.latent_to_rgb([0, 0, 1, 0, 0, 0, 0])) # (190, 12, 0) red ?
# print(mixbox.latent_to_rgb([0, 0, 0, 1, 0, 0, 0])) # (254, 255, 254) white


# wavelengths = range(0, 36, 1)

# weights = optimize_rgb_bytes([0,0,255])['weights']
# K_, S_ = mix_K_S(weights)
# set_pigment_K_S(0, K_, S_)

# print(len(K[0]))
# Labels for pigments

# pigment_index = 1
# Rp = FastColorMath.K_S_to_R(FastColorMath.K[pigment_index], FastColorMath.S[pigment_index])
# xyz = FastColorMath.reflectance_to_xyz(Rp)
# oklab = FastColorMath.xyz_to_oklab(xyz)
# srgb = [ max(0, min(1.0, n)) for n in FastColorMath.oklab_to_srgb(oklab) ]
# rgb = [n * 255 for n in srgb]
# print(FastColorMath.rgb_bytes_to_hex(rgb))

# rgb = FastColorMath.hex_to_rgb_bytes('#ffd483')
# rgb = [190, 12, 0]
# weights = FastColorMath.optimize_rgb_bytes(rgb)['weights']
# K_, S_ = FastColorMath.mix_K_S(weights)

# K_ = FastColorMath.K[pigment_index]
# S_ = FastColorMath.S[pigment_index]

# # Generate equally spaced indices for the original data
# x = np.linspace(0, len(K_) - 1, len(K_))

# # Create a cubic spline interpolator
# cs = CubicSpline(x, K_)

# # Interpolate at the same x positions (smoothed values)
# K_ = cs(x)

# FastColorMath.set_pigment_K_S(pigment_index, K_, S_)


def generate_cmyk_grid(step=0.1):
    """
    Generates a sparse CMYK grid with values in the range [0, 1].
    """
    c = np.arange(0, 1 + step, step)
    m = np.arange(0, 1 + step, step)
    y = np.arange(0, 1 + step, step)
    k = np.arange(0, 1 + step, step)
    grid = np.array(np.meshgrid(c, m, y, k, indexing="ij"))
    return grid.reshape(4, -1).T  # Reshape to (N, 4)

def mix_to_srgb(cmyk):
    r = mix_pigments(cmyk)
    xyz = reflectance_to_xyz(r)
    oklab = xyz_to_oklab(xyz)
    srgb = oklab_to_srgb(oklab)
    
    # srgb = np.array(mixbox.latent_to_rgb([
    #   cmyk[0], cmyk[1], cmyk[2], cmyk[3], 0, 0, 0
    # ])) / 255.0
    return srgb

def plot_cmyk_gamut(step=0.1):
    """
    Plots the gamut of CMYK colors in the sRGB space.
    """
    # Generate CMYK grid
    cmyk_grid = generate_cmyk_grid(step)

    # Convert to sRGB
    srgb_colors = np.array([mix_to_srgb(cmyk) for cmyk in cmyk_grid])
    
    # Clamp sRGB values to [0, 1]
    # srgb_colors = np.clip(srgb_colors, 0, 1)

    # Extract valid points within the sRGB cube
    # valid = np.all((srgb_colors >= 0) & (srgb_colors <= 1), axis=1)
    # valid_srgb = srgb_colors[valid]
    valid_srgb = srgb_colors
    colors = np.clip(srgb_colors, 0, 1)

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(valid_srgb[:, 0], valid_srgb[:, 1], valid_srgb[:, 2],
               c=colors, s=1, marker='o')

    # Set axis labels and limits
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title("CMYK Gamut in sRGB Space")
    plt.show()

# Function to plot for a single pigment
def plot_pigment(pigment_index):
    Kp = K[pigment_index]
    Sp = S[pigment_index]
    Rp = K_S_to_R(Kp, Sp)
    
    CMYK = ['C', 'M', 'Y', 'K']
    pigment_name = CMYK[pigment_index]
    
    xyz = reflectance_to_xyz(Rp)
    oklab = xyz_to_oklab(xyz)
    srgb = [ max(0, min(1.0, n)) for n in oklab_to_srgb(oklab) ]
    color = tuple(srgb)
    
    plt.style.use("dark_background")

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, Kp, label=f'{pigment_name} - Absorption (K)', color="gray", linestyle='--')
    plt.plot(wavelengths, Sp, label=f'{pigment_name} - Scattering (S)', color="gray", linestyle=':')
    plt.plot(wavelengths, Rp, label=f'{pigment_name} - Reflectance (R)', color=color, linewidth=3)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coefficient / Reflectance')
    plt.title(f'{pigment_name} Pigment - K, S, and R Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def logistic_warp(t, k=10, m=0.5):
    """
    Warps the parameter t using a logistic function.
    
    Parameters:
      t : float or numpy array
          Input parameter in the range [0, 1].
      k : float
          Steepness of the logistic curve.
      m : float
          Midpoint of the transition.
    
    Returns:
      Warped t in the range [0, 1].
    """
    return 1 / (1 + np.exp(-k * (t - m)))

def power_warp(t, gamma=2.0):
    """
    Warps the parameter t using a power function.
    
    Parameters:
      t : float or numpy array
          Input parameter in the range [0, 1].
      gamma : float
          Exponent for warping.
    
    Returns:
      Warped t in the range [0, 1].
    """
    return (t**gamma) / (t**gamma + (1-t)**gamma)
    
def mix_colors (latent0, latent1, t):
  # latent0[0:4] = power_warp(latent0[0:4], 1)
  # latent1[0:4] = power_warp(latent1[0:4], 1)
  # latent1 = power_warp(latent1, 1.1)
  # t = power_warp(t, 1.1)
  latent = FastColorMath.lerp_latent(latent0, latent1, t)
  
  # count = len(K)
  # cmyk = latent[0:count]
  # residual = latent[count:-1]
  # mixed_r = FastColorMath.forward_r(cmyk, K, S)
  # mixed_xyz = FastColorMath.reflectance_to_xyz(mixed_r)
  # mixed_oklab = FastColorMath.xyz_to_oklab(mixed_xyz) + np.array(residual, dtype=np.float64)
  # mixed_srgb = FastColorMath.oklab_to_srgb(mixed_oklab)
  # mixed_srgb = np.clip(mixed_srgb, 0.0, 1.0)
  # mixed_oklab = FastColorMath.srgb_to_oklab(mixed_srgb)
  
  # return mixed_oklab
  return FastColorMath.srgb_to_oklab(FastColorMath.latent_to_srgb(latent, K, S))

def mibox_mix_colors (rgb0, rgb1, t):
  rgb = np.array(mixbox.lerp(rgb0, rgb1, t), dtype=np.float32) / 255.0
  return FastColorMath.srgb_to_oklab(rgb)


# def plot_pigment(weight0, weights1, steps, K, S):

#   for i in range(steps):
#     t = i / (steps-1)
#     new_weights = FastColorMath.lerp(weight0, weights1, t)
    
#     Kp, Sp = FastColorMath.mix_K_S(new_weights, K, S)
#     R = FastColorMath.mix_pigments(new_weights, K, S)

#     xyz = FastColorMath.reflectance_to_xyz(R)
#     oklab = FastColorMath.xyz_to_oklab(xyz)
#     srgb = [ max(0, min(1.0, n)) for n in FastColorMath.oklab_to_srgb(oklab) ]
#     color = tuple(srgb)

#     plt.style.use("dark_background")

#     plt.figure(figsize=(10, 6))
#     plt.plot(FastColorMath.wavelengths, Kp, label=f'Absorption (K)', color="gray", linestyle='--')
#     plt.plot(FastColorMath.wavelengths, Sp, label=f'Scattering (S)', color="gray", linestyle=':')
#     plt.plot(FastColorMath.wavelengths, R, label=f'Reflectance (R)', color=color, linewidth=3)

#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Intensity')
#     plt.title(f'Spectral Data')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"tmp/ramps/")
#     plt.close()

def plot_color_interpolations(rgb0, rgb1, count, K, S):
    """
    Computes and plots two interpolation curves between latent0 and latent1:
      - one using the custom mix_colors function,
      - and one using the mixbox_mix function.
      
    Three plots are produced:
      1. The a–b plane (from OKLab)
      2. The L–C (Chroma) plane (from OKLCH)
      3. The Hue (h in OKLCH) vs. interpolation parameter t
      
    Each point is colored with its corresponding sRGB value.
    """

    # Generate t values from 0 to 1.
    t_values = np.linspace(0, 1, count)
    latent0 = FastColorMath.srgb_to_latent(np.array(rgb0)/255.0, K, S)
    latent1 = FastColorMath.srgb_to_latent(np.array(rgb1)/255.0, K, S)
    
    num_pigments = len(K)
    print('Weights0:', [ f"{n:.3f}" for n in latent0[:num_pigments].tolist() ])
    print('Weights1:', [ f"{n:.3f}" for n in latent1[:num_pigments].tolist() ])
    
    
    
    # Compute interpolated OKLab colors for each method.
    custom_oklab = [mix_colors(latent0, latent1, t) for t in t_values]
    mixbox_oklab = [mibox_mix_colors(rgb0, rgb1, t) for t in t_values]
    
    # Convert OKLab colors to OKLCH and sRGB.
    custom_oklch = [FastColorMath.oklab_to_oklch(c) for c in custom_oklab]
    mixbox_oklch = [FastColorMath.oklab_to_oklch(c) for c in mixbox_oklab]
    
    custom_srgb = [np.clip(FastColorMath.oklab_to_srgb(c), 0.0, 1.0) for c in custom_oklab]
    mixbox_srgb = [np.clip(FastColorMath.oklab_to_srgb(c), 0.0, 1.0) for c in mixbox_oklab]
    
    # Extract a and b coordinates (OKLab).
    custom_a = [c[1] for c in custom_oklab]
    custom_b = [c[2] for c in custom_oklab]
    mixbox_a = [c[1] for c in mixbox_oklab]
    mixbox_b = [c[2] for c in mixbox_oklab]
    
    # Extract L (lightness), C (chroma) and h (hue in degrees) from OKLCH.
    custom_L = [c[0] for c in custom_oklch]
    custom_C = [c[1] for c in custom_oklch]
    custom_h = [c[2] for c in custom_oklch]
    mixbox_L = [c[0] for c in mixbox_oklch]
    mixbox_C = [c[1] for c in mixbox_oklch]
    mixbox_h = [c[2] for c in mixbox_oklch]
    
    # Create three side-by-side plots.
    fig, (ax_ab, ax_lc, ax_h) = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Plot on the a–b plane ---
    # ax_ab.scatter(custom_a, custom_b, s=50, c=custom_srgb,
    #               label='Custom Mix', edgecolor='black')
    # ax_ab.scatter(mixbox_a, mixbox_b, s=50, c=mixbox_srgb,
    #               marker='s', label='Mixbox Mix', edgecolor='black')
    # ax_ab.set_xlim(-0.25, 0.25)
    # ax_ab.set_ylim(-0.25, 0.25)
    # ax_ab.set_xlabel("a")
    # ax_ab.set_ylabel("b")
    # ax_ab.set_title("Path in the a–b Plane")
    # ax_ab.grid(True)
    # ax_ab.legend()
    
    # --- Plot on the L–C (Chroma) plane ---
    # ax_lc.scatter(custom_L, custom_C, s=50, c=custom_srgb,
    #               label='Custom Mix', edgecolor='black')
    # ax_lc.scatter(mixbox_L, mixbox_C, s=50, c=mixbox_srgb,
    #               marker='s', label='Mixbox Mix', edgecolor='black')
    # ax_lc.set_xlabel("L (Lightness)")
    # ax_lc.set_ylabel("C (Chroma)")
    # ax_lc.set_title("Path in the L–C (Chroma) Plane")
    # ax_lc.grid(True)
    # ax_lc.legend()
    
    ax_ab.scatter(t_values, custom_L, s=50, c=custom_srgb, edgecolor='black', label='Custom Mix')
    ax_ab.scatter(t_values, mixbox_L, s=50, c=mixbox_srgb, marker='s', edgecolor='black', label='Mixbox Mix')
    ax_ab.set_xlabel("Interpolation Parameter (t)")
    ax_ab.set_ylabel("L (Lightness)")
    ax_ab.set_title("Lightness along Gradient")
    ax_ab.grid(True)
    ax_ab.legend()
    
    ax_lc.scatter(t_values, custom_C, s=50, c=custom_srgb, edgecolor='black', label='Custom Mix')
    ax_lc.scatter(t_values, mixbox_C, s=50, c=mixbox_srgb, marker='s', edgecolor='black', label='Mixbox Mix')
    ax_lc.set_xlabel("Interpolation Parameter (t)")
    ax_lc.set_ylabel("C (Chroma)")
    ax_lc.set_title("Chroma along Gradient")
    ax_lc.grid(True)
    ax_lc.legend()
    
    # --- Plot the Hue shift along the gradient ---
    # Plot hue (in degrees) versus t for both methods.
    ax_h.scatter(t_values, custom_h, s=50, c=custom_srgb, edgecolor='black', label='Custom Mix')
    ax_h.scatter(t_values, mixbox_h, s=50, c=mixbox_srgb, marker='s', edgecolor='black', label='Mixbox Mix')
    ax_h.set_xlabel("Interpolation Parameter (t)")
    ax_h.set_ylabel("Hue (degrees)")
    ax_h.set_title("Hue Shift along the Gradient")
    ax_h.set_ylim(0, 360)
    ax_h.grid(True)
    ax_h.legend()
    
    plt.tight_layout()
    plt.show()

COLORS = [
  [[10, 52, 162], [248, 210, 71]],
  [[0, 33, 133],[252, 210, 0]],
  [[243, 240, 247],[0, 176, 0]],
  [[ 26, 10, 83 ],[ 255, 255, 255 ]],
  [[ 0, 66, 170 ],[ 255, 255, 255 ]],
  [[195,209,23],[0,97,255]],
  [[255,170,0],[0,97,255]],
  [[255,106,0],[142,230,255]],
  [[124,42,0],[142,230,255]],
  [[0,0,0],[255,255,255]],
  [[0,0,5],[255,255,255]],
  [[128,128,128],[255,255,255]],
  [[128,128,128],[0,0,0]]
]

K, S = FastColorMath.load_standard_K_S()
count = 100

# cmyk0 = np.array([1,0,0,0],dtype=np.float64)
# cmyk1 = np.array([0,1,0,0],dtype=np.float64)

# rgb0 = np.clip(FastColorMath.oklab_to_srgb(FastColorMath.forward(cmyk0, K, S)),0.0,1.0) * 255.0
# rgb1 = np.clip(FastColorMath.oklab_to_srgb(FastColorMath.forward(cmyk1, K, S)),0.0,1.0) * 255.0

# rgb0, rgb1 = COLORS[0]
# rgb0, rgb1 = ([ 34.15020236,  29.76379219, 100.31963467], [239.86501094, 185.16527717,  22.28024157])

rgb0 = np.array(mixbox.latent_to_rgb([1,0,0,0,0,0,0]), dtype=np.float64)
rgb1 = np.array(mixbox.latent_to_rgb([0,1,0,0,0,0,0]), dtype=np.float64)

plot_color_interpolations(rgb0, rgb1, count, K, S)

# # Generate the list of interpolated OKLab colors.
# oklab_list = [mix_colors(latent0, latent1, i / (count - 1)) for i in range(count)]

# # Convert each OKLab color to OKLCH.
# oklch_list = [FastColorMath.oklab_to_oklch(color) for color in oklab_list]

# srgb_list = [FastColorMath.oklab_to_srgb(color) for color in oklab_list]

# # Extract the a and b coordinates for the AB plane
# a_coords = [color[1] for color in oklab_list]
# b_coords = [color[2] for color in oklab_list]

# # Extract L and Chroma (C) for the L-C curve.
# L_coords = [color[0] for color in oklch_list]
# C_coords = [color[1] for color in oklch_list]

# # Create two side-by-side plots.
# fig, (ax_ab, ax_lc) = plt.subplots(1, 2, figsize=(12, 6))

# # Plot the path in the AB plane.
# ax_ab.scatter(a_coords, b_coords, s=50, c=srgb_list)
# ax_ab.set_xlim(-0.4, 0.4)
# ax_ab.set_ylim(-0.4, 0.4)
# ax_ab.set_xlabel("a")
# ax_ab.set_ylabel("b")
# ax_ab.set_title("Path in the a-b Plane")
# ax_ab.grid(True)

# # scatter the L vs. Chroma (C) curve.
# ax_lc.scatter(L_coords, C_coords, s=50, c=srgb_list)
# ax_lc.set_xlabel("L (Lightness)")
# ax_lc.set_ylabel("C (Chroma)")
# ax_lc.set_title("Path in the L-C (Chroma) Plane")
# ax_lc.grid(True)

# plt.tight_layout()
# plt.show()

# Run the function to plot
# plot_cmyk_gamut(step=0.05)

# Example usage: Plot for pigment C (index 0) with cyan color
# for i in range(4):
#   plot_pigment(pigment_index=i)

# # Convert CMYK to normalized RGB for Matplotlib
# rgb_colors = {k: (v[0], v[1], v[2]) for k, v in cmyk_colors.items()}
# print(rgb_colors)

# # Plot K (absorption)
# plt.figure(figsize=(12, 6))
# for i, pigment in enumerate(pigments):
#     plt.plot(wavelengths, K[i], label=f'{pigment} - Absorption (K)', color=rgb_colors[pigment], linewidth=2)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Absorption (K)')
# plt.title('Absorption Coefficients (K) for Pigments')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot S (scattering)
# plt.figure(figsize=(12, 6))
# for i, pigment in enumerate(pigments):
#     plt.plot(wavelengths, S[i], label=f'{pigment} - Scattering (S)', color=rgb_colors[pigment], linewidth=2)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Scattering (S)')
# plt.title('Scattering Coefficients (S) for Pigments')
# plt.legend()
# plt.grid(True)
# plt.show()