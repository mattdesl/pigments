import FastColorMath
import numpy as np
from PIL import Image
import math

dtype = np.float32

def rgb_to_index_3D (srgb, N):
    stepSize = 255.0 / (N-1.0)
    return np.floor(np.array(srgb) / stepSize).astype(np.uint32)

def rgb_to_index_1D (srgb, N):
    return index_3D_to_index_1D(rgb_to_index_3D(srgb, N), N)

def index_3D_to_index_1D (index, N):
    # flatIndex=(iR×N×N)+(iG×N)+iB
    return index[0] * N * N + index[1] * N + index[2]

def index_3D_to_rgb (index, N):
    stepSize = 255.0 / (N-1.0)
    return np.clip(np.floor(np.array(index) * stepSize).astype(np.uint8), 0, 255)

def index_1D_to_3D (index, N):
    iB = index % N
    tmp = (index - iB) / N
    iG = tmp % N
    iR = (tmp - iG) / N
    return np.floor(np.array([iR, iG, iB])).astype(np.uint32)

def build_srgb_to_cmyk_lut(K, S, filename, N=32):
    """
    Builds an NxNxN LUT from sRGB -> CMYK, storing the result in a 2D image (PNG).

    1. Construct sRGB cube of shape (N, N, N, 3)
    2. Convert to OKLab
    3. For each point, compute CMYK via 'optimize_weights'
    4. Reshape and save as an RGBA PNG
    """
    
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
    oklab_flat = FastColorMath.srgb_to_oklab(rgb_flat / 255.0)  # shape (N^3, 3)
    
    # 4) Prepare an array to hold the CMYK results
    rgb_out = np.zeros((N**3, 3), dtype=np.uint8)

    # 5) If your optimize_weights can handle the entire array at once, do it:
    for i in range(N**3):
        cmyk = FastColorMath.optimize_weights(oklab_flat[i], K, S)
        cmy = cmyk[:3]
        cmy_u8 = np.clip(cmy * 255, 0, 255).astype(np.uint8)        
        rgb_out[i] = cmy_u8        
            
        # Optional progress logging
        if i % 5000 == 0:
            print(f"Processed {i} / {N**3} ({100*i/(N**3):.2f}%)")

    width = N
    height = N**2

    # Reshape to (height, width, 3)
    rgb_out = rgb_out.reshape((height, width, 3))

    print("RGBOUT", rgb_out)
    # Create PIL Image
    image = Image.fromarray(rgb_out, 'RGB')

    # 6) Convert to PIL Image, mode='RGB'
    # img_pil = Image.fromarray(cmy_2d, mode='RGB')
    image.save(filename)
    print(f"Saved 3-channel CMY LUT to {filename}")

    
if __name__ == "__main__":
    # N=16
    # i3d = np.array([255, 16, 0])
    # i1d = rgb_to_index_1D(i3d, N)
    # print(i3d)
    # print(i1d)
    # print(index_1D_to_3D(i1d, N))
    # print(index_3D_to_rgb(i3d, N))
    # Example usage
    K, S = FastColorMath.load_standard_K_S()
    build_srgb_to_cmyk_lut(K=K, S=S, filename='data/cmyk_lut.png', N=32)