import matplotlib.pyplot as plt
import time

# import autograd.numpy as np
import numpy as np
# from autoptim import minimize
from numba import njit, config
import scipy

# from scipy.interpolate import CubicSpline
import json

from CubicBezier import cubic_bezier_1D

K1 = 0.206
K2 = 0.03
K3 = (1.0 + K1) / (1.0 + K2)

# DeltaE OKLab factor for the a* b* plane
# DeltaEOK2 is a recent suggestion by Björn Ottosson to improve color difference
# specifically around chroma/hue changes
ab_factor = 1.0

# set to false if you have many thousands/millions of calls to make
# but JIT is just a bit slow for single-color operations and demo purposes
config.DISABLE_JIT = True

dtype = np.float64

# 380 - 730 nm, 10 nm increments
wavelengths = np.array(list(range( 380, 740, 10 )))

EPSILON = 1e-12
GAMMA = 2.2

# CMF 2º standard observer, 380 - 730 nm, 10nm increments, D65 scaled
CMF_X = np.array([
    0.0000646936115727633, 0.000219415369171578, 0.00112060228414359,
    0.00376670730427686, 0.0118808497572766, 0.0232870228938867,
    0.0345602796797156, 0.0372247180152918, 0.0324191842208867,
    0.0212337349018611, 0.0104912522835777, 0.00329591973705558,
    0.000507047802540891, 0.000948697853868474, 0.00627387448845597,
    0.0168650445840847, 0.0286903641895679, 0.0426758762490725,
    0.0562561504260008, 0.0694721289967602, 0.0830552220141023,
    0.0861282432155783, 0.0904683927868683, 0.0850059839999687,
    0.0709084366392777, 0.0506301536932269, 0.0354748461653679,
    0.0214687454102844, 0.0125167687669176, 0.00680475126078526,
    0.00346465215790157, 0.00149764708248624, 0.000769719667700118,
    0.000407378212832335, 0.000169014616182123, 0.0000952268887534793,
], dtype=dtype)

CMF_Y = np.array([
    0.00000184433541764457, 0.0000062054782702308, 0.0000310103776744139,
    0.000104750996050908, 0.000353649345357243, 0.000951495123526191,
    0.00228232006613489, 0.00420743392201395, 0.00668896510747318,
    0.00988864251316196, 0.015249831581587, 0.0214188448516808,
    0.0334237633103485, 0.0513112925264347, 0.0704038388936896,
    0.0878408968669549, 0.0942514030194481, 0.0979591120948518,
    0.094154532672617, 0.0867831869897857, 0.078858499565938,
    0.0635282861874625, 0.0537427564004085, 0.0426471274206905,
    0.0316181374233466, 0.0208857265390802, 0.0138604556350511,
    0.00810284218307029, 0.00463021767605804, 0.002491442109212,
    0.00125933475912608, 0.000541660024106255, 0.000277959820700288,
    0.000147111734433903, 0.0000610342686915558, 0.0000343881801451621,
], dtype=dtype)

CMF_Z = np.array([
    0.000305024750978023, 0.00103683251144092, 0.00531326877604233,
    0.0179548401495523, 0.057079004340659, 0.11365445199637,
    0.173363047597462, 0.196211466514214, 0.186087009289904,
    0.139953964010199, 0.0891767523322851, 0.0478974052884572,
    0.0281463269981882, 0.0161380645679562, 0.00775929533717298,
    0.00429625546625385, 0.00200555920471153, 0.000861492584272158,
    0.000369047917008248, 0.000191433500712763, 0.000149559313956664,
    0.0000923132295986905, 0.0000681366166724671, 0.0000288270841412222,
    0.0000157675750930075, 0.00000394070233244055, 0.00000158405207257727,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=dtype)

# Matrices from @texel/color

RGB_TO_LMS = np.asarray([
    [0.4122214694707629, 0.5363325372617349, 0.051445993267502196],
    [0.2119034958178251, 0.6806995506452345, 0.10739695353694051],
    [0.08830245919005637, 0.2817188391361215, 0.6299787016738223],
], dtype=dtype)

LMS_TO_OKLAB = np.asarray([
    [0.210454268309314, 0.7936177747023054, -0.0040720430116193],
    [1.9779985324311684, -2.42859224204858, 0.450593709617411],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
], dtype=dtype)

OKLAB_TO_LMS = np.asarray([
    [1.0, 0.3963377773761749, 0.2158037573099136],
    [1.0, -0.1055613458156586, -0.0638541728258133],
    [1.0, -0.0894841775298119, -1.2914855480194092],
], dtype=dtype)

LMS_TO_RGB = np.asarray([
    [4.076741636075959, -3.307711539258062, 0.2309699031821041],
    [-1.2684379732850313, 2.6097573492876878, -0.3413193760026569],
    [-0.004196076138675526, -0.703418617935936, 1.7076146940746113],
], dtype=dtype)

XYZ_D65_to_LMS = np.asarray([
    [0.819022437996703, 0.3619062600528904, -0.1288737815209879],
    [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
    [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
], dtype=dtype)

# DEFAULT_INTERPOLATION_PARAMS = None
# DEFAULT_INTERPOLATION_PARAMS = [[0.4453519011908943, 0.42978240210821966], [-0.1466646311316046, -0.15479776554856595], [-0.3033304381844724, -0.3110131988323558], [0.3030503125818883, 0.34924068176182155]] 
DEFAULT_INTERPOLATION_PARAMS = None
# DEFAULT_INTERPOLATION_PARAMS = [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] 

def normalize_KS(K, S):
    # Convert to NumPy arrays for easier manipulation
    K = np.array(K, dtype=dtype)
    S = np.array(S, dtype=dtype)

    # Find the maximum value in K and S across all pigments and wavelengths
    max_K = np.max(K)
    max_S = np.max(S)

    # Find the global scaling factor (maximum of max_K and max_S)
    global_max = max(max_K, max_S)

    # Normalize K and S by the global maximum
    K_norm = K / global_max
    S_norm = S / global_max

    return K_norm, S_norm

def load_standard_K_S ():
  with open('data/adjusted-primaries-optimized-feb17-contrast.json', 'r') as json_file:
    ksdata = json.load(json_file)
    K = np.array(ksdata['K'], dtype=dtype)
    S = np.array(ksdata['S'], dtype=dtype)
  return K, S

@njit(nopython=True)
def parameterize(t, p):
    exp_val = 1.0 + p
    num = t ** exp_val
    den = num + (1.0 - t) ** exp_val
    return np.where(p == 0.0, t, num / den)

@njit(nopython=True)
def forward_r(weights, K_, S_):
    return mix_pigments(weights, K_, S_)
  
@njit(nopython=True)
def forward(weights, K_, S_):
    mixed_r = forward_r(weights, K_, S_)
    xyz = reflectance_to_xyz(mixed_r)
    return xyz_to_oklab(xyz)
  
@njit()
def linear_srgb_to_srgb(lin):
    abs_lin = np.abs(lin)
    abs_gam = np.where(
        abs_lin <= 0.0031308,
        12.92 * abs_lin,
        1.055 * np.power(abs_lin, 1/2.4) - 0.055
    )
    return np.sign(lin) * abs_gam

@njit()
def srgb_to_linear_srgb(gam):
    abs_gam = np.abs(gam)
    abs_lin = np.where(
        abs_gam <= 0.040449936,
        abs_gam / 12.92,
        np.power((abs_gam + 0.055) / 1.055, 2.4)
    )
    return np.sign(gam) * abs_lin

@njit(nopython=True)
def lerp_latent(latent0, latent1, t, tvals=DEFAULT_INTERPOLATION_PARAMS):
    latent = lerp(latent0,latent1,t)
    if tvals is None:
      return latent

    num_pigments = len(tvals)
    for i in range(num_pigments):
      tv = tvals[i]
      latent[i] = lerp(latent0[i], latent1[i], sigmoid_transform(t, tv[0], tv[1]))
    
    return latent

@njit()
def srgb_to_latent (srgb, K_, S_):
  oklab = srgb_to_oklab(srgb)
  oklch = oklab_to_oklch(oklab)
  
  c = optimize_weights(oklab, K_, S_)
  mixed_r = forward_r(c, K_, S_)
  mixed_xyz = reflectance_to_xyz(mixed_r)
  mixed_oklab = xyz_to_oklab(mixed_xyz)
  # mixed_oklch = oklab_to_oklch(mixed_oklab)
  # mixed_srgb = oklab_to_srgb(mixed_oklab)
  
  residual = oklab - mixed_oklab
  
  # Y = mixed_xyz[1]
  # C = lch[1]
  return np.concatenate((c, residual))
  # return np.concatenate((c, residual, np.array([Y], dtype=dtype)))

@njit()
def latent_to_srgb (latent, K_, S_):
  count = len(K_)
  cmyk = latent[0:count]
  residual = latent[count:]
  # target_Y = latent[-1]
  
  # mixed_r = forward_r(cmyk, K_, S_, params)
  # mixed_xyz = reflectance_to_xyz(mixed_r)
  # cur_Y = mixed_xyz[1]
  
  # scaling = 1.0
  # if cur_Y > 1e-6:
      # scaling = target_Y / cur_Y
  # mixed_xyz = mixed_xyz * scaling
  # mixed_xyz = reflectance_to_xyz(mixed_r * scaling)
  
  r_term = 1.0
  mixed_oklab = forward(cmyk, K_, S_)
  # mixed_oklab = xyz_to_oklab(mixed_xyz)
  
  # scaling = 1.0
  # if mixed_oklab[0] > 1e-6:
  #     scaling = target_Y / mixed_oklab[0]
  # mixed_oklab = mixed_oklab * scaling
  
  # mixed_oklch = oklab_to_oklch(mixed_oklab)
  # mixed_oklch = mixed_oklch + r_term * np.array(residual, dtype=dtype)
  # mixed_oklch[1] = tanh_sigmoid(mixed_oklch[1],1,2)
  # mixed_oklab = oklch_to_oklab(mixed_oklch)
  
  # mixed_oklch = oklab_to_oklch(mixed_oklab)
  # cur_C = mixed_oklch[1]
  # c_term = 1.0
  # if cur_C > 1e-6:
  #   c_term = target_C / cur_C
  # mixed_oklch[1] = mixed_oklch[1] * c_term
  # mixed_oklab = oklch_to_oklab(mixed_oklch)
  # print("MIX", mixed_oklch)
  
  mixed_oklab = mixed_oklab + r_term * np.array(residual, dtype=dtype)
  
  mixed_srgb = oklab_to_srgb(mixed_oklab)
  return np.clip(mixed_srgb, 0.0, 1.0)

@njit()
def xyz_to_oklab(c):
    # 1) Apply the XYZ->LMS transform
    lms = c @ XYZ_D65_to_LMS.T  # shape (3,)
    
    # 2) Cubic root in LMS space
    lms_ = np.cbrt(lms)  # shape (3,)
    
    # 3) Multiply by LMS->Oklab to get Oklab
    return lms_ @ LMS_TO_OKLAB.T  # shape (3,)

@njit()
def xyz_to_lms_linear(c):
    # 1) Apply the XYZ->LMS transform
    lms = c @ XYZ_D65_to_LMS.T  # shape (3,)
    return lms

@njit()
def lms_linear_to_oklab(lms):
    # 2) Cubic root in LMS space
    lms_ = np.cbrt(lms)
    # 3) Multiply by LMS->Oklab to get Oklab
    return lms_ @ LMS_TO_OKLAB.T

@njit()
def linear_srgb_to_oklab(c):
    lms = c @ RGB_TO_LMS.T
    lms_ = np.cbrt(lms)
    return lms_ @ LMS_TO_OKLAB.T

@njit()
def oklab_to_linear_srgb(c):
    lms_ = c @ OKLAB_TO_LMS.T
    lms = lms_ ** 3
    return lms @ LMS_TO_RGB.T

@njit()
def oklab_to_lms_linear(c):
    return c @ OKLAB_TO_LMS.T
  
@njit()
def srgb_to_oklab(c):
    return linear_srgb_to_oklab(srgb_to_linear_srgb(c))
  
@njit()
def oklab_to_srgb(c):
    return linear_srgb_to_srgb(oklab_to_linear_srgb(c))

@njit()
def oklch_to_oklab(lch):
    l, c, h = lch
    h_rad = np.radians(h)
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    return np.array([l, a, b], dtype=dtype)
  
@njit()
def oklab_to_oklch(lab):
    l, a, b = lab
    c = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360
    return np.array([l,c,h], dtype=dtype)

@njit()
def oklab_diff_sq (oklab0, oklab1, cur_ab_factor=ab_factor):
    dL = oklab0[0] - oklab1[0]
    da = (oklab0[1] - oklab1[1]) * cur_ab_factor
    db = (oklab0[2] - oklab1[2]) * cur_ab_factor
    return dL * dL + da * da + db * db

@njit()
def oklab_diff (oklab0, oklab1):
    return oklab_diff_sq(oklab0, oklab1)

@njit()
def reflectance_to_xyz(reflectance):
    X = np.dot(CMF_X, reflectance)
    Y = np.dot(CMF_Y, reflectance)
    Z = np.dot(CMF_Z, reflectance)
    return np.array([X, Y, Z], dtype=dtype)

@njit()
def K_S_to_R (K, S, epsilon=EPSILON):
  K_S_ratio = np.where(S > epsilon, K / S, 0.0)
  return np.where(
    S > epsilon,
    1.0 + K_S_ratio - np.sqrt(K_S_ratio**2 + 2.0 * K_S_ratio),
    0.0
  )

@njit()
def mix_K_S (weights, K_, S_, gamma=GAMMA, epsilon=EPSILON):
  total_weight = np.sum(weights)
  if total_weight == 0:
      total_weight = 1.0
  comp_weights = weights / total_weight

  if gamma == 1.0:
      # Linear blending using dot products.
      mixed_K = np.dot(comp_weights, K_)
      mixed_S = np.dot(comp_weights, S_)
  else:
      # Compute nonlinear weights: (w_i ** gamma) normalized.
      nonlin_weights = comp_weights ** gamma
      sum_nonlin = np.sum(nonlin_weights)
      if sum_nonlin < epsilon:
          sum_nonlin = epsilon
      nonlin_weights /= sum_nonlin

      # Blend K and S nonlinearly for each wavelength.
      # (nonlin_weights[:,None] * (K_ ** gamma)) has shape (n_pigments, n_wavelengths)
      mixed_K = np.sum(nonlin_weights[:, None] * (K_ ** gamma), axis=0) ** (1.0 / gamma)
      mixed_S = np.sum(nonlin_weights[:, None] * (S_ ** gamma), axis=0) ** (1.0 / gamma)

  # Ensure non-negative values.
  mixed_K = np.maximum(mixed_K, 0)
  mixed_S = np.maximum(mixed_S, 0)
  return mixed_K, mixed_S

@njit()
def mix_K_S_simple(weights, K_, S_):
    total_weight = np.sum(weights)
    if total_weight == 0:
        total_weight = 1.0
    comp_weights = weights / total_weight
    mixed_K = comp_weights @ K_
    mixed_S = comp_weights @ S_
    return mixed_K, mixed_S

@njit(nopython=True)
def sigmoid_transform(R, a=0.0, b=0.0):
    """
    A two-parameter sigmoidal transformation that is identity when (a, b) = (0,0).
    
    f(R; a, b) = R^(1+a) / ( R^(1+a) + (1-R)^(1+b) )
    
    Parameters:
      R : numpy array or scalar, assumed to be in [0, 1].
      a : parameter controlling curvature on the R side.
      b : parameter controlling curvature on the (1-R) side.
      
    Returns:
      Transformed values, in [0, 1].
    """
    # Avoid division by zero issues by working on the assumption R is in [0,1].
    numerator = R ** (1 + a)
    denominator = numerator + (1 - R) ** (1 + b)
    return numerator / denominator

@njit(nopython=True)
def mix_pigments(weights, K_, S_, gamma=GAMMA, epsilon=EPSILON):
  """
  Mix pigments using nonlinear blending of K and S coefficients.

  Parameters:
      weights : 1D array of pigment weights (length = number of pigments)
      K_      : 2D array of K coefficients (shape: [n_pigments, n_wavelengths])
      S_      : 2D array of S coefficients (shape: [n_pigments, n_wavelengths])
      gamma   : Exponent for nonlinear blending (gamma=1 is linear, >1 gives sharper transitions)
      epsilon : Small constant to avoid division-by-zero

  Returns:
      R : 1D array of mixed reflectance values (per wavelength)
  """
  mixed_K, mixed_S = mix_K_S(weights, K_, S_, gamma, epsilon)
  # Compute R from Kubelka-Munk infinite thickness formula
  return K_S_to_R(mixed_K, mixed_S, epsilon)

@njit(nopython=True)
def mix_pigments_og(weights, K_, S_, epsilon=1e-12):
    total_weight = np.sum(weights)
    if total_weight == 0:
        total_weight = 1.0
    comp_weights = weights / total_weight
    mixed_K = comp_weights @ K_
    mixed_S = comp_weights @ S_
    
    mixed_K = np.maximum(mixed_K, 0)
    mixed_S = np.maximum(mixed_S, 0)
    
    K_S_ratio = np.where(mixed_S > epsilon, mixed_K / mixed_S, 0.0)
    R = np.where(
      mixed_S > epsilon,
      1.0 + K_S_ratio - np.sqrt(K_S_ratio**2 + 2.0 * K_S_ratio),
      0.0
    )
    return R
  
@njit()
def normalized (v):
  summed = np.sum(v)
  return np.array([ n / summed for n in v ], dtype)

@njit(nopython=True)
def compute_chroma(oklab):
    return np.sqrt(oklab[1]**2 + oklab[2]**2)

@njit(nopython=True)
def objective_cmywk(weights, target_oklab, K_, S_):
  epsilon=0.05  # threshold below which we are "achromatic"
  scale=0.04   # span of the transition region
  lambda_penalty=1
  norm_weights = normalized(weights)
  
  # mixed_r = mix_pigments(norm_weights, K_, S_)
  # mixed_xyz = reflectance_to_xyz(mixed_r)
  # mixed_oklab = xyz_to_oklab(mixed_xyz)
  
  mixed_r = forward_r(normalized(weights), K_, S_)
  mixed_xyz = reflectance_to_xyz(mixed_r)
  mixed_oklab = xyz_to_oklab(mixed_xyz)

  color_diff = oklab_diff_sq(mixed_oklab, target_oklab, 2.0)
  target_chroma = compute_chroma(target_oklab)
  
  # apply a penalty for when achromatic colors introduce CMY values
  chroma_penalty_factor = 1.0 - smoothstep(target_chroma, epsilon - scale / 2, epsilon + scale / 2)
  cmy_penalty = np.sum(norm_weights[0:3])
  total_error = color_diff + lambda_penalty * chroma_penalty_factor * cmy_penalty
  
  return total_error
  
@njit(nopython=True)
def objective(weights, target_oklab, K_, S_):
  mixed_oklab = forward(normalized(weights), K_, S_)
  color_diff = oklab_diff_sq(mixed_oklab, target_oklab)
  return color_diff

def optimize_weights (target_oklab, K_, S_):
  bounds = [(0.000001, 1)]
  count = len(K_)
  obj = objective_cmywk
  weights = scipy.optimize.minimize( obj, np.array( [ 1.0 / count ] * count, dtype=dtype ), args=(target_oklab, K_, S_), method='L-BFGS-B', bounds = bounds )['x']
  return normalized(np.array(weights,dtype=dtype))

def optimize (target_oklab, K_, S_):
  weights = optimize_weights(target_oklab, K_, S_)
  xyz = reflectance_to_xyz(mix_pigments(weights, K_, S_))
  oklab = xyz_to_oklab(xyz)
  return {
    'residual': target_oklab - oklab,
    'oklab': oklab,
    'weights': weights
  }
  
def optimize_rgb_bytes (rgb, K_, S_):
  rgbf = np.array(rgb, dtype) / 255.0
  oklab = srgb_to_oklab(rgbf)
  return optimize(oklab, K_, S_)

@njit()
def lerp(a, b, t):
    return a * (1.0 - t) + b * t

def hex_to_rgb_bytes(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove '#' if present
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_bytes_to_hex(rgb):
    rgb = np.array(rgb).astype(np.uint8).tolist()
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def OKLToLr (x):
  return 0.5 * (K3 * x - K1 + np.sqrt((K3 * x - K1) * (K3 * x - K1) + 4 * K2 * K3 * x))

def OKLrToL (x):
  return (x ** 2 + K1 * x) / (K3 * (x + K2))

@njit
def clamp( x, lowBound = 0.0, highBound = 1.0 ):
    return max(min(x, highBound), lowBound)

@njit
def smoothstep( x, lowBound = 0.0, highBound = 1.0 ):
    x = clamp((x - lowBound) / (highBound - lowBound), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
