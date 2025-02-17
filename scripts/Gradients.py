import matplotlib.pyplot as plt
import FastColorMath
import numpy as np
from PIL import Image
# from scipy.interpolate import CubicSpline
import mixbox
import math

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
  [[0,0,5],[255,255,255]],
  [[128,128,128],[255,255,255]],
  [[128,128,128],[0,0,0]],
]

def euclidean_distance(color1, color2):
  return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

def generate_distant_rgb(threshold=200):
  """
  Generate two random RGB values that are distant from each other.

  Args:
      threshold (int): Minimum Euclidean distance between the two colors in RGB space (0-441).

  Returns:
      tuple: Two tuples, each representing an RGB color.
  """
  while True:
    rgb1 = tuple(np.random.randint(0, 256) for _ in range(3))
    rgb2 = tuple(np.random.randint(0, 256) for _ in range(3))
    if euclidean_distance(rgb1, rgb2) >= threshold:
        return rgb1, rgb2


def plot_pigment(weights, fname, K, S):
    Kp, Sp = FastColorMath.mix_K_S(weights, K, S)
    R = FastColorMath.mix_pigments(weights, K, S)
    
    xyz = FastColorMath.reflectance_to_xyz(R)
    oklab = FastColorMath.xyz_to_oklab(xyz)
    srgb = [ max(0, min(1.0, n)) for n in FastColorMath.oklab_to_srgb(oklab) ]
    color = tuple(srgb)
    
    plt.style.use("dark_background")

    plt.figure(figsize=(10, 6))
    plt.plot(FastColorMath.wavelengths, Kp, label=f'Absorption (K)', color="gray", linestyle='--')
    plt.plot(FastColorMath.wavelengths, Sp, label=f'Scattering (S)', color="gray", linestyle=':')
    plt.plot(FastColorMath.wavelengths, R, label=f'Reflectance (R)', color=color, linewidth=3)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title(f'Spectral Data')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

def plot_refl(R, fname):
    xyz = FastColorMath.reflectance_to_xyz(R)
    oklab = FastColorMath.xyz_to_oklab(xyz)
    srgb = [ max(0, min(1.0, n)) for n in FastColorMath.oklab_to_srgb(oklab) ]
    color = tuple(srgb)
    
    plt.style.use("dark_background")

    plt.figure(figsize=(10, 6))
    # plt.plot(wavelengths, Kp, label=f'{pigment_name} - Absorption (K)', color="gray", linestyle='--')
    plt.plot(FastColorMath.wavelengths, R, label=f'Reflectance (R)', color=color, linewidth=3)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'R Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.close()

def generate_pair (iters = 1000):
  best_pair = None
  max_dist = 0
  for i in range(iters):
    rgb1 = tuple(np.random.randint(0, 256) for _ in range(3))
    rgb2 = tuple(np.random.randint(0, 256) for _ in range(3))
    lerped_rgb = np.array(FastColorMath.lerp(np.array(rgb1, dtype=np.float32), np.array(rgb2, dtype=np.float32), 0.5), dtype=np.uint8)
    mixed_rgb = np.array(mixbox.lerp(rgb1, rgb2, 0.5), dtype=np.uint8)
    delta = euclidean_distance(mixed_rgb, lerped_rgb)
    if delta > max_dist:
        max_dist = delta
        best_pair = (rgb1, rgb2)
  return best_pair

def generate_gradients (colors=COLORS):
  steps = 400
  ramp_count = len(colors)
  width = 1350
  height = steps
  scale_factor = 1

  K, S = FastColorMath.load_standard_K_S()
  
  # K[1] *= 4.0
  # S[1] *= 2.0
  
  # print(FastColorMath.cubic_bezier_1D(0.25, 0, 0.5, 1, 0.5))
  # print(FastColorMath.cubic_bezier_1D(0, 0, 1, 1, 0.5))
  # print(FastColorMath.cubic_bezier_1D(0, 0, 1, 1, 0.25))
  # print(FastColorMath.cubic_bezier_1D(0, 0, 1, 1, 0.75))
  # K = FastColorMath.K
  # S = FastColorMath.S


  # w = FastColorMath.optimize_rgb_bytes(np.array([ 53, 168, 47 ]), K, S)['weights']
  # print('w', w.tolist())
  # # w = FastColorMath.normalized(np.array([0.0, 0.0, 0.1, 0, 0], dtype=np.float32))
  # srgb = FastColorMath.oklab_to_srgb(FastColorMath.xyz_to_oklab(FastColorMath.reflectance_to_xyz(FastColorMath.mix_pigments(w, K, S))))
  # rgb = (srgb* 255.0).astype(np.uint8)
  # print(
  #   FastColorMath.rgb_bytes_to_hex(rgb)
  # )

  # white_idx = 3
  # K[white_idx] = np.ones_like(K[white_idx]) * 0.1
  # S[white_idx] = np.ones_like(S[white_idx]) * 0.35

  # black_idx = 4
  # K[black_idx] = np.ones_like(K[black_idx]) * 0.25
  # S[black_idx] = np.ones_like(S[black_idx]) * 0.02

  types = [ 'pigment' ]

  for curType in types:
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    slice_width = int(width / ramp_count)

    # Generate each color ramp
    for ramp_index, (rgb0, rgb1) in enumerate(colors):
        
        # col0 = FastColorMath.optimize_rgb_bytes(rgb0, K, S)
        # col1 = FastColorMath.optimize_rgb_bytes(rgb1, K, S)

        latent0 = FastColorMath.srgb_to_latent(np.array(rgb0)/255.0, K, S)
        latent1 = FastColorMath.srgb_to_latent(np.array(rgb1)/255.0, K, S)
      
        # print(ramp_index)
        # print(rgb0, latent0[:4].tolist())
        # print(rgb1, latent1[:4].tolist())
        # print('----')
      
        # print(latent0.tolist()[:4])    
        # print(latent1.tolist()[:4])    
        # Fill each column in the ramp
        for i in range(steps):
            t = i / (steps - 1)
            # n = np.random.normal(0, 0.01)
            # result = FastColorMath.kmlerp_srgb(col0, col1, t, 1.0, K, S)
            # tvals = [[0.10962330404542064, 0.3112910926171555], [-0.014257685953745863, -0.02505066951313623], [-0.17451345787062877, -0.2062093109716318], [0.226170417538478, 0.12588209993514174]]
            # tvals = None
            # tvals = [[0.5271665527444882, 0.48117265568340767], [0.0851006580117056, 0.06052200666391869], [-0.07412806239343507, -0.009650922295516808], [0.43889793751013695, 0.4238178822762813]]
            latent = FastColorMath.lerp_latent(latent0, latent1, t)
            # latent = FastColorMath.lerp_latent(latent0, latent1, t, tvals)
            result = FastColorMath.latent_to_srgb(latent, K, S)
            
            # if curType == 'pigment' and i % 50 == 0:
            #   count = len(K)
            #   cmyk = latent[0:count]
              # refl = FastColorMath.forward_r(cmyk, K, S, params)
              # plot_refl(refl, f'./tmp/ramps/{ramp_index}-{i}.png')
              # plot_pigment(cmyk, f'./tmp/ramps/{ramp_index}-{i}.png', K, S, params)
              # mean = np.mean(refl)
              # print(f"mean {i} {mean}")
            
            # lerp in rgb space
            if curType == 'pigment-nor':
              latentNoR = np.array([latent[0],latent[1],latent[2],latent[3],0,0,0],dtype=np.float32)
              result = FastColorMath.latent_to_srgb(latentNoR, K, S)
            elif curType == 'srgb':
              result = FastColorMath.lerp(np.array(rgb0, dtype=np.float32)/255.0, np.array(rgb1, dtype=np.float32)/255.0, t)
            # lerp in oklab space
            elif curType == 'oklab':
              oklab0 = FastColorMath.srgb_to_oklab(np.array(rgb0,dtype=np.float32)/255.0)
              oklab1 = FastColorMath.srgb_to_oklab(np.array(rgb1,dtype=np.float32)/255.0)
              oklab = FastColorMath.lerp(oklab0, oklab1, t)
              result = FastColorMath.oklab_to_srgb(oklab)
            elif curType == 'fft':
              pcount = len(K)
              residuals = latent[pcount:]
              r0 = FastColorMath.mix_pigments(latent0[:pcount], K, S)
              r1 = FastColorMath.mix_pigments(latent1[:pcount], K, S)
              fft0 = np.fft.fft(r0)
              fft1 = np.fft.fft(r1)
              R_fft = FastColorMath.lerp(fft0, fft1, t)

              # Set echo/delay parameters.
              # delay = 0.25       # Delay in sample units.
              # echo_gain = i * 0.25  # Echo amplitude factor.              
              # n = len(r0)
              # # Construct the echo filter in the Fourier domain.
              # # For each Fourier component (index k), multiply by (1 + echo_gain * exp(-i*2pi*k*delay/n)).
              # k = np.arange(n)
              # echo_filter = 1 + echo_gain * np.exp(-1j * 2 * np.pi * k * delay / n)
            
              # Apply the echo filter.
              # R_fft = R_fft * echo_filter
              # R_fft += np.random.normal(scale=0.1, size=len(R_fft))
              
              # r = np.fft.ifft(R_fft).real
              r = FastColorMath.lerp(r0,r1,t)
              oklab = FastColorMath.xyz_to_oklab(FastColorMath.reflectance_to_xyz(r))
              oklab += residuals
              result = FastColorMath.oklab_to_srgb(oklab)
              
            rgb = [max(0, min(255, int(n * 255))) for n in result]
            if curType == 'mixbox':
              rgb = mixbox.lerp(rgb0,rgb1,t)
            elif curType == 'mixbox-nor':
              mlatent0 = np.array(mixbox.rgb_to_latent(rgb0), dtype=np.float32)
              mlatent1 = np.array(mixbox.rgb_to_latent(rgb1), dtype=np.float32)
              mlatent = FastColorMath.lerp(mlatent0, mlatent1, t)
              mlatent = [mlatent[0],mlatent[1],mlatent[2],mlatent[3],0,0,0]
              rgb = mixbox.latent_to_rgb(mlatent)
                        
            for x in range(slice_width):
                pixels[x + ramp_index * slice_width, i] = tuple(rgb)

    # Scale the image by the scale factor
    resized_width = width * scale_factor
    resized_height = height * scale_factor
    img = img.resize((resized_width, resized_height), Image.NEAREST)

    # Save the image
    img.save(f"gradient-{curType}.png")

if __name__ == "__main__":
  # np.random.seed(0)
  # rand_colors = []
  # for i in range(10):
  #   rand_colors.append(generate_pair())
  
  # target_oklab = np.array([0,0,0],dtype=np.float32)
  # weights = FastColorMath.optimize_weights(target_oklab, FastColorMath.K, FastColorMath.S)
  # print('target', target_oklab.tolist())
  # print('weights', weights.tolist())
  # R = FastColorMath.mix_pigments(weights, FastColorMath.K, FastColorMath.S)
  # # R = FastColorMath.mix_pigments(np.array([0, 0, 0, 1],dtype=np.float32), FastColorMath.K, FastColorMath.S)
  # # R = FastColorMath.mix_pigments(np.array([0, 0, 0, 0.25],dtype=np.float32), FastColorMath.K, FastColorMath.S)
  # xyz = FastColorMath.reflectance_to_xyz(R)
  # oklab = FastColorMath.xyz_to_oklab(xyz)
  # srgb = FastColorMath.oklab_to_srgb(oklab)
  # rgb = (srgb * 255.0).astype(np.uint8)
  
  # print('mixed oklab', oklab.tolist())
  # print('oklab residuals', (target_oklab - oklab).tolist())
  
  # residuals = (target_oklab - oklab)
  # result_oklab = oklab + residuals
  # print('result oklab', result_oklab.tolist())
  
  print('Generating...')
  generate_gradients()
  
# rgb0 = [ 102, 156, 53 ]
# rgb0 = [ 26, 10, 83 ]
# rgb1 = [ 255, 255, 255 ]

# rgb0 = [0, 33, 133]
# rgb1 = [252,210,0]
# col0 = FastColorMath.optimize_rgb_bytes(rgb0)
# col1 = FastColorMath.optimize_rgb_bytes(rgb1)

# steps = 400
# # Image dimensions (width=64, height=steps)
# width, height = 150, steps
# scale_factor = 1
# img = Image.new('RGB', (width, height))
# pixels = img.load()

# # Fill each row in the image
# for i in range(height):
#   t = i / (steps - 1)
#   result = FastColorMath.kmlerp_srgb(col0, col1, t)
#   rgb = [max(0, min(255, int(n * 255))) for n in result]
#   for x in range(width):
#     # img.putpixel((x, i), tuple(result))
#     pixels[x, i] = tuple(rgb)

# # Scale the image by 4x using nearest neighbor
# resized_width = width * scale_factor
# resized_height = height * scale_factor
# img = img.resize((resized_width, resized_height), Image.NEAREST)

# # Save the image
# img.save('gradient.png')

