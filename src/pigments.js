import * as Color from "@texel/color";
import { K, S } from "../data/adjusted-primaries-optimized-feb17-contrast.json";
import { LToLr, LrToL } from "./color/util.js";
import weights from "../data/model_weights.json";
// import weightsReLUHS16_from32 from "../data/model_weights_hs16.json";
import polynomialModel from "../data/polynomial.json";
// const L0_IN = 3;
// const L0_OUT = 12;
// const L2_OUT = 12;
// const L4_OUT = 5;

// K[3] = K[3].map((n) => 0.0025);
// S[3] = S[3].map((n) => 0.5);

// K[4] = K[4].map((n) => 0.5);
// S[4] = S[4].map((n) => 0.0025);

// console.log((window.json = JSON.stringify({ K, S })));

// const DEFAULT_INTERPOLATION_PARAMS = null;
export const DEFAULT_INTERPOLATION_PARAMS = [
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0],
];
export const OKLab_to_LMS_M = [
  [1.0, +0.3963377774, +0.2158037573],
  [1.0, -0.1055613458, -0.0638541728],
  [1.0, -0.0894841775, -1.291485548],
];

// Convert 2D weight arraysZ to 1D Float32Arrays for faster indexing.
export function flatten2D(arr) {
  const flat = new Float32Array(arr.length * arr[0].length);
  let index = 0;
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr[i].length; j++) {
      flat[index++] = arr[i][j];
    }
  }
  return flat;
}

export const toData = (weights, params, silu = false) => ({
  params,
  silu,
  layer0Weight: flatten2D(weights["model.0.weight"]),
  layer0Bias: new Float32Array(weights["model.0.bias"]),
  layer2Weight: flatten2D(weights["model.2.weight"]),
  layer2Bias: new Float32Array(weights["model.2.bias"]),
  layer4Weight: flatten2D(weights["model.4.weight"]),
  layer4Bias: new Float32Array(weights["model.4.bias"]),
});

// const dataStd = toData(weightsReLUHS16_from32, [3, 16, 16, 5], false);

//
export const NUM_PIGMENTS = 5;
export const dataReLU = toData(weights, [3, 16, 16, NUM_PIGMENTS], false);

export function SiLU(x) {
  return x * (1 / (1 + Math.exp(-x)));
}

export function ReLU(x) {
  return Math.max(x, 0);
}

export function predictNeuralNetwork(input, data) {
  const [L0_IN, L0_OUT, L2_OUT, L4_OUT] = data.params;
  const silu = data.silu;

  // input = Color.transform(input, OKLab_to_LMS_M);

  // Ensure input is a Float32Array of length 3.
  const x = new Float32Array(L0_IN);
  for (let i = 0; i < L0_IN; i++) {
    x[i] = input[i];
  }

  // --- Layer 0: Linear (input: 3 -> output: 16) + ReLU ---
  const layer0Out = new Float32Array(L0_OUT);
  for (let i = 0; i < L0_OUT; i++) {
    let sum = data.layer0Bias[i];
    for (let j = 0; j < L0_IN; j++) {
      sum += data.layer0Weight[i * L0_IN + j] * x[j];
    }

    // activation.
    layer0Out[i] = silu ? SiLU(sum) : ReLU(sum);
  }

  // --- Layer 2: Linear (16 -> 16) + ReLU ---
  const layer2Out = new Float32Array(L2_OUT);
  for (let i = 0; i < L2_OUT; i++) {
    let sum = data.layer2Bias[i];
    for (let j = 0; j < L0_OUT; j++) {
      sum += data.layer2Weight[i * L0_OUT + j] * layer0Out[j];
    }
    // activation.
    layer2Out[i] = silu ? SiLU(sum) : ReLU(sum);
  }

  // --- Layer 4: Linear (16 -> 4) ---
  const layer4Out = new Float32Array(L4_OUT);
  for (let i = 0; i < L4_OUT; i++) {
    let sum = data.layer4Bias[i];
    for (let j = 0; j < L2_OUT; j++) {
      sum += data.layer4Weight[i * L2_OUT + j] * layer2Out[j];
    }
    layer4Out[i] = sum;
  }

  // --- Softmax ---
  let maxVal = -Infinity;
  for (let i = 0; i < L4_OUT; i++) {
    if (layer4Out[i] > maxVal) maxVal = layer4Out[i];
  }
  let sumExp = 0;
  for (let i = 0; i < L4_OUT; i++) {
    layer4Out[i] = Math.exp(layer4Out[i] - maxVal);
    sumExp += layer4Out[i];
  }
  for (let i = 0; i < L4_OUT; i++) {
    layer4Out[i] /= sumExp;
  }

  // Return the predicted [c, m, y, k] as a normal array.
  return [...layer4Out];
  // return [layer4Out[0], layer4Out[1], layer4Out[2], layer4Out[3]];
}

export const CMF = {
  wavelengths: {
    min: 380,
    max: 730,
    steps: 10,
    data: [
      [
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
      ],
      [
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
      ],
      [
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
      ],
    ],
  },
};

// prettier-ignore
// export function evalPolynomial(cmyk) {
//   const [c0, c1, c2, c3] = cmyk;
//   var r = 0.0;
//   var g = 0.0;
//   var b = 0.0;

//   var c00 = c0 * c0;
//   var c11 = c1 * c1;
//   var c22 = c2 * c2;
//   var c33 = c3 * c3;
//   var c01 = c0 * c1;
//   var c02 = c0 * c2;
//   var c12 = c1 * c2;

//   var w = 0.0;
//   w = c0*c00; r += +0.07717053*w; g += +0.02826978*w; b += +0.24832992*w;
//   w = c1*c11; r += +0.95912302*w; g += +0.80256528*w; b += +0.03561839*w;
//   w = c2*c22; r += +0.74683774*w; g += +0.04868586*w; b += +0.00000000*w;
//   w = c3*c33; r += +0.99518138*w; g += +0.99978149*w; b += +0.99704802*w;
//   w = c00*c1; r += +0.04819146*w; g += +0.83363781*w; b += +0.32515377*w;
//   w = c01*c1; r += -0.68146950*w; g += +1.46107803*w; b += +1.06980936*w;
//   w = c00*c2; r += +0.27058419*w; g += -0.15324870*w; b += +1.98735057*w;
//   w = c02*c2; r += +0.80478189*w; g += +0.67093710*w; b += +0.18424500*w;
//   w = c00*c3; r += -0.35031003*w; g += +1.37855826*w; b += +3.68865000*w;
//   w = c0*c33; r += +1.05128046*w; g += +1.97815239*w; b += +2.82989073*w;
//   w = c11*c2; r += +3.21607125*w; g += +0.81270228*w; b += +1.03384539*w;
//   w = c1*c22; r += +2.78893374*w; g += +0.41565549*w; b += -0.04487295*w;
//   w = c11*c3; r += +3.02162577*w; g += +2.55374103*w; b += +0.32766114*w;
//   w = c1*c33; r += +2.95124691*w; g += +2.81201112*w; b += +1.17578442*w;
//   w = c22*c3; r += +2.82677043*w; g += +0.79933038*w; b += +1.81715262*w;
//   w = c2*c33; r += +2.99691099*w; g += +1.22593053*w; b += +1.80653661*w;
//   w = c01*c2; r += +1.87394106*w; g += +2.05027182*w; b += -0.29835996*w;
//   w = c01*c3; r += +2.56609566*w; g += +7.03428198*w; b += +0.62575374*w;
//   w = c02*c3; r += +4.08329484*w; g += -1.40408358*w; b += +2.14995522*w;
//   w = c12*c3; r += +6.00078678*w; g += +2.55552042*w; b += +1.90739502*w;

//   return [r, g, b];
// }

export function reflectance_to_xyz(reflectance) {
  let X = 0,
    Y = 0,
    Z = 0;
  const cmfData = CMF.wavelengths.data;
  for (let i = 0; i < reflectance.length; i++) {
    var intensity = reflectance[i];
    X += cmfData[0][i] * intensity;
    Y += cmfData[1][i] * intensity;
    Z += cmfData[2][i] * intensity;
  }
  return [X, Y, Z];
}

export function xyz_to_oklab(xyz) {
  return Color.convert(xyz, Color.XYZ, Color.OKLab);
}

export function normalized(weights) {
  let sumW = weights.reduce((sum, w) => sum + w, 0);
  if (sumW == 0) sumW = 1;
  return weights.map((w) => w / sumW);
}

export function mix_pigments_to_K_S(weights, K, S) {
  // Step 1: Calculate sum of weights
  let sumW = weights.reduce((sum, w) => sum + w, 0);
  if (sumW == 0) sumW = 1;
  let normalizedWeights = weights.map((w) => w / sumW);

  // Step 3: Perform matrix multiplication for K and S

  // For K values
  const mixedK = new Array(K[0].length).fill(0);
  for (let wavelength = 0; wavelength < K[0].length; wavelength++) {
    for (let pigment = 0; pigment < weights.length; pigment++) {
      mixedK[wavelength] += normalizedWeights[pigment] * K[pigment][wavelength];
    }
  }

  // For S values
  const mixedS = new Array(S[0].length).fill(0);
  for (let wavelength = 0; wavelength < S[0].length; wavelength++) {
    for (let pigment = 0; pigment < weights.length; pigment++) {
      mixedS[wavelength] += normalizedWeights[pigment] * S[pigment][wavelength];
    }
  }

  // Handle zero denominators in S values
  const EPSILON = 1e-12; // Same small value as Python version
  for (let i = 0; i < mixedS.length; i++) {
    if (mixedS[i] === 0) {
      console.warn("mix has zero denominator for at least one wavelength");
      mixedS[i] = EPSILON;
    }
  }

  return [mixedK, mixedS];
}

export function mix_pigments(weights, K, S) {
  const [mixedK, mixedS] = mix_pigments_to_K_S(weights, K, S);
  return K_S_to_R(mixedK, mixedS);
}

export function mix_pigments_v2(weights, K, S) {
  const numPigments = K.length;
  const numWavelengths = K[0].length;

  const R = new Array(numWavelengths).fill(0);
  const versionA = true;

  for (let j = 0; j < numWavelengths; j++) {
    let total = 0;
    let Kv = 0;
    let Sv = 0;

    for (let i = 0; i < numPigments; i++) {
      const curK = K[i];
      const curS = S[i];
      const w = weights[i];
      Kv += w * curK[j];
      Sv += w * curS[j];
      total += w;
    }

    if (total !== 0) {
      Kv /= total;
      Sv /= total;
    }

    // opt A:
    if (versionA) {
      if (Math.abs(Sv) < 0.00001) {
        R[j] = 0.0;
      } else {
        let ratio = Kv / Sv;
        R[j] = 1.0 + ratio - Math.sqrt(Math.abs(ratio * ratio + 2.0 * ratio));
      }
    } else {
      let a = 1.0 + Kv / Sv;
      let b = Math.sqrt(a * a - 1.0);

      function coth(f) {
        return Math.cosh(f) / Math.sinh(f);
      }

      let substrate_refl = 0.1;
      let paint_thickness = 0.001;

      let Rv = 1.0 - substrate_refl * (a - b * coth(b) * Sv * paint_thickness);
      Rv /= a - substrate_refl + b * coth(b) * Sv * paint_thickness;
      R[j] = Rv;
    }
  }
  return R;
}

export function K_S_to_R(K, S) {
  const R = Array(K.length).fill(0);
  for (let i = 0; i < K.length; i++) {
    const K_S = Math.abs(K[i] / (S[i] + 1e-12));
    const Rf = 1.0 + K_S - Math.sqrt(K_S ** 2 + 2.0 * K_S);
    R[i] = Rf;
  }
  // if (R.some((n) => isNaN(n))) debugger;
  return R;
}

export async function loadImage(src) {
  return new Promise((r) => {
    const img = new Image();
    img.onload = () => r(img);
    img.src = src;
  });
}

export function getImageData(image) {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  return ctx.getImageData(0, 0, image.width, image.height);
}

export function lerp(a, b, t) {
  return a * (1.0 - t) + b * t;
}

export function lerpArray(a, b, t) {
  return a.map((n, i) => lerp(n, b[i], t));
}

export function clamp(a, min, max) {
  return Math.max(Math.min(a, max), min);
}

export function smoothstep(x, lowBound = 0, highBound = 1) {
  x = clamp((x - lowBound) / (highBound - lowBound), 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

// 0..255 to index 3D
export function rgb_to_index_3D(srgb, N) {
  const stepSize = 255.0 / (N - 1.0);
  return srgb.map((v) => Math.floor(v / stepSize));
}

export function rgb_to_index_1D(srgb, N) {
  return index_3D_to_index_1D(rgb_to_index_3D(srgb, N), N);
}

export function index_3D_to_index_1D(index, N) {
  // flatIndex=(iR×N×N)+(iG×N)+iB
  return index[0] * N * N + index[1] * N + index[2];
}

export function index_3D_to_rgb(index, N) {
  const stepSize = 255.0 / (N - 1.0);
  return index.map((v) => clamp(Math.floor(v * stepSize), 0, 255));
}

export function index_1D_to_3D(index, N) {
  let iB = index % N;
  let tmp = (index - iB) / N;
  let iG = tmp % N;
  let iR = (tmp - iG) / N;
  return [iR, iG, iB].map((n) => clamp(Math.floor(n), 0, N - 1));
}

export function sample(data, x, y, z, N, channels = 4) {
  x = clamp(x, 0, N - 1);
  y = clamp(y, 0, N - 1);
  z = clamp(z, 0, N - 1);
  const idx1d = index_3D_to_index_1D([x, y, z], N);
  const idx = idx1d * channels;
  const c8 = data[idx + 0];
  const m8 = data[idx + 1];
  const y8 = data[idx + 2];
  // Convert to [0..1] floats
  const C = c8 / 255.0;
  const M = m8 / 255.0;
  const Y = y8 / 255.0;
  let K = 1 - (C + M + Y);
  K = Math.max(0.0, K); // or clamp to ensure no negatives
  return [C, M, Y, K];
}

export function getCMYKFromLUT(srgb, N, dataRGBA) {
  const step = N - 1;
  const r = Math.max(0, Math.min(step, Math.floor(srgb[0] * step)));
  const g = Math.max(0, Math.min(step, Math.floor(srgb[1] * step)));
  const b = Math.max(0, Math.min(step, Math.floor(srgb[2] * step)));
  return sample_lut([r, g, b], N, dataRGBA);
  // const channels = 4;
  // return sample(dataRGBA, r, g, b, width, channels);
}

// Convert sRGB [0,255] to floating indices based on step size
export function rgb_to_float_indices(srgb, N) {
  const stepSize = 255.0 / (N - 1.0);
  return srgb.map((v) => v / stepSize);
}

// Get indices and fractional parts for interpolation
export function rgb_to_indices_and_fractions(srgb, N) {
  const floatIndices = rgb_to_float_indices(srgb, N);
  const indices = floatIndices.map((fi) => Math.floor(fi));
  const fractions = floatIndices.map((fi) => fi - Math.floor(fi));

  // Ensure indices do not exceed N-2 to prevent out-of-bounds
  return indices.map((idx, i) => {
    if (idx >= N - 1) {
      return { index0: N - 2, index1: N - 1, fraction: 1.0 };
    } else {
      return { index0: idx, index1: idx + 1, fraction: fractions[i] };
    }
  });
}

// Sample CMYK from LUT at a given 1D index
export function sample_lut(xyz, N, data, channels = 4) {
  const offset = index_3D_to_index_1D(xyz, N) * channels;
  const C = data[offset + 0] / 255.0;
  const M = data[offset + 1] / 255.0;
  const Y = data[offset + 2] / 255.0;
  const K = 1 - (C + M + Y);
  // const K =
  //   channels >= 4 ? data[offset + 3] / 255.0 : 1 - Math.max(C, Math.max(M, Y));
  return [C, M, Y, K];
}

export function linearToConcentration(oklab0, oklab1, t, exponent = 2.0) {
  // Compute weighted contributions with the adjustable exponent
  const l1 = LToLr(oklab0[0]);
  const l2 = LToLr(oklab1[0]);
  const t1 = l1 * (1.0 - t) ** exponent;
  const t2 = l2 * t ** exponent;
  return t1 + t2 <= 0 ? t : t2 / (t1 + t2);
}

export function evalPolynomial(pigments) {
  pigments = pigments.slice();
  // pigments[0] = smoothstep(pigments[0], 0, 1);
  // pigments[1] = smoothstep(pigments[1], 0, 1);
  // pigments[2] = smoothstep(pigments[2], 0, 1);
  // const oklab0 = xyz_to_oklab(reflectance_to_xyz(mix_pigments(pigments, K, S)));
  const oklab1 = evalPolynomial2(pigments);

  // console.log("weights", pigments, oklab0, oklab1);
  return oklab1;
}

function evalPolynomial2(pigments, model = polynomialModel) {
  const N = model.n_variables;
  if (pigments.length !== N) {
    throw new Error(
      `Expected pigment array of length ${N} but got ${pigments.length}`
    );
  }

  // in OKLab
  let L = 0,
    a = 0,
    b = 0;

  // Loop over every term in the polynomial model.
  for (let j = 0; j < model.terms.length; j++) {
    let monomial = 1;
    const term = model.terms[j];
    const exponents = term[0];
    const coeffs = term[1];
    // For each variable, raise to the required exponent.
    for (let i = 0; i < N; i++) {
      // If exponent is zero, Math.pow returns 1.
      monomial *= Math.pow(pigments[i], exponents[i]);
    }
    // Accumulate the weighted contributions for each output channel.
    L += coeffs[0] * monomial;
    a += coeffs[1] * monomial;
    b += coeffs[2] * monomial;
  }

  return [L, a, b];
}

export function smootherstep(x, edge0, edge1) {
  // Scale, and clamp x to 0..1 range
  x = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return x * x * x * (x * (6.0 * x - 15.0) + 10.0);
}

// console.log("input", [255, 128, 50]);
// console.log("mixbox", mixbox.rgbToLatent([255, 128, 50]));
// console.log("mixbox", mixbox.latentToRgb(mixbox.rgbToLatent([255, 128, 50])));

// console.log("neural", neuralRGBToLatent([255, 128, 50]));
// console.log("neural", neuralLatentToRGB(neuralRGBToLatent([255, 128, 50])));

export function sigmoid(R, a = 0.0, b = 0.0) {
  const numerator = Math.pow(R, 1 + a);
  const denominator = numerator + Math.pow(1 - R, 1 + b);
  return numerator / denominator;
}

export function lerpLatents(
  latent0,
  latent1,
  t,
  tvals = DEFAULT_INTERPOLATION_PARAMS
) {
  // Basic linear interpolation over the entire latent array.
  const latent = latent0.map((val, i) => lerp(val, latent1[i], t));

  // If no tvals provided, return the linearly interpolated latent.
  if (!tvals) {
    return latent;
  }

  const numPigments = tvals.length;
  for (let i = 0; i < numPigments; i++) {
    const [a, b] = tvals[i];
    // Use the custom sigmoid transform on t and then interpolate the i-th component.
    latent[i] = lerp(latent0[i], latent1[i], sigmoid(t, a, b));
  }

  return latent;
}

export function lerpLatents2(
  oklab0,
  oklab1,
  latent0,
  latent1,
  t,
  linear = false
) {
  const rcount = 3;
  const gamma = 2.2;

  const rawT = t;
  // t = smoothstep(t, 0.0, 1.0);
  // t = linearToConcentration(oklab0, oklab1, t, 2);
  const pcount = latent0.length - rcount;
  // const factor = 0.1; //0.05;
  const factor = 0.0; //0.05;
  // const cmyT = smoothstep(t, 0, 1);
  const cmyT = t;
  // const cmyT = linear ? t : smootherstep(t, 0.0 - factor, 1.0 + factor);
  // const cmyT = linear ? t : smoothstep(t, 0.0 - factor, 1.0 + factor);
  // const cmyT = linearToConcentration(oklab0, oklab1, t, 2);

  // const cmyT = smoothstep(t, 0, 1);
  // const l1 = LToLr(oklab0[0]);
  // const l2 = LToLr(oklab1[0]);
  // const t1 = l1 * (1.0 - t) ** 2;
  // const t2 = l2 * t ** 2;
  // const cmyT = t1 + t2 <= 0 ? t2 : t2 / (t1 + t2);

  // const c = lerp(latent0[0], latent1[0], smoothstep(t, 0, 1));
  // const m = lerp(latent0[1], latent1[1], smoothstep(t, 0, 1));
  // const y = lerp(latent0[2], latent1[2], smoothstep(t, 0, 1));
  // const cmy = [c, m, y];
  const cmy = lerpArray(latent0.slice(0, 3), latent1.slice(0, 3), t);
  const kw = lerpArray(latent0.slice(3, pcount), latent1.slice(3, pcount), t);
  const r = lerpArray(latent0.slice(pcount), latent1.slice(pcount), t);
  return [...cmy, ...kw, ...r];
  // return [...cmy.map((t) => smoothstep(t, 0, 1)), ...kw, ...r];
  // return [
  //   ...cmy.map((t) => linearToConcentration(oklab0, oklab1, t)),
  //   ...kw,
  //   ...r,
  // ];
  // return [...cmy.map((t) => smoothstep(t, 0, 1)), ...kw, ...r];
  // return a.map((n, i) => lerp(n, b[i], t));
}

export function OKLtoConcentration(oklab0, oklab1, t) {
  return smoothstep(t, 0.0, 1.0);
  // const l1 = LToLr(oklab0[0]);
  // const l2 = LToLr(oklab1[0]);
  // const t1 = l1 * (1.0 - t) ** 2;
  // const t2 = l2 * t ** 2;
  // return t2 / (t1 + t2);
}

// Perform trilinear interpolation
export function getCMYKFromLUT_Trilinear(srgb, N, dataRGBA, channels = 4) {
  const step = N - 1;
  const rw = srgb[0] * step;
  const gw = srgb[1] * step;
  const bw = srgb[2] * step;
  // index
  const ri = Math.max(0, Math.min(step, Math.floor(rw)));
  const gi = Math.max(0, Math.min(step, Math.floor(gw)));
  const bi = Math.max(0, Math.min(step, Math.floor(bw)));
  // fractional
  const tR = rw - ri;
  const tG = gw - gi;
  const tB = bw - bi;

  const r0 = Math.max(0, Math.min(step, ri));
  const g0 = Math.max(0, Math.min(step, gi));
  const b0 = Math.max(0, Math.min(step, bi));
  const r1 = Math.max(0, Math.min(step, ri + 1));
  const g1 = Math.max(0, Math.min(step, gi + 1));
  const b1 = Math.max(0, Math.min(step, bi + 1));

  const C000 = sample_lut([r0, g0, b0], N, dataRGBA, channels); // idx000
  const C001 = sample_lut([r0, g0, b1], N, dataRGBA, channels); // idx001
  const C010 = sample_lut([r0, g1, b0], N, dataRGBA, channels); // idx010
  const C011 = sample_lut([r0, g1, b1], N, dataRGBA, channels); // idx011
  const C100 = sample_lut([r1, g0, b0], N, dataRGBA, channels); // idx100
  const C101 = sample_lut([r1, g0, b1], N, dataRGBA, channels); // idx101
  const C110 = sample_lut([r1, g1, b0], N, dataRGBA, channels); // idx110
  const C111 = sample_lut([r1, g1, b1], N, dataRGBA, channels); // idx111

  // Step 3: Interpolate along the B axis
  const C00 = lerpArray(C000, C001, tB);
  const C01 = lerpArray(C010, C011, tB);
  const C10 = lerpArray(C100, C101, tB);
  const C11 = lerpArray(C110, C111, tB);

  // Step 4: Interpolate along the G axis
  const C0 = lerpArray(C00, C01, tG);
  const C1 = lerpArray(C10, C11, tG);

  // Step 5: Interpolate along the R axis
  const C_final = lerpArray(C0, C1, tR);

  return C_final; // [C, M, Y, K]
}

export function neuralRGBToLatent(rgb) {
  const oklab = Color.convert(
    rgb.map((n) => n / 0xff),
    Color.sRGB,
    Color.OKLab
  );
  const weights = predictNeuralNetwork(oklab, dataReLU);
  const predicted = xyz_to_oklab(
    reflectance_to_xyz(mix_pigments(weights, K, S))
  );
  return [
    ...weights,
    oklab[0] - predicted[0],
    oklab[1] - predicted[1],
    oklab[2] - predicted[2],
  ];
}

// console.log(neuralRGBToLatent([50, 32, 172]).slice(0, 5));
// console.log(neuralRGBToLatentPolynomial([50, 32, 172]).slice(0, 5));
// console.log(
//   neuralLatentToRGBPolynomial(neuralRGBToLatentPolynomial([50, 32, 172]))
// );

export function neuralRGBToLatentPolynomial(rgb) {
  const oklab = Color.convert(
    rgb.map((n) => n / 0xff),
    Color.sRGB,
    Color.OKLab
  );
  const weights = predictNeuralNetwork(oklab, dataReLU);
  const predicted = evalPolynomial(weights);
  return [
    ...weights,
    oklab[0] - predicted[0],
    oklab[1] - predicted[1],
    oklab[2] - predicted[2],
  ];
}

export function neuralLatentToRGB(latent) {
  const rcount = 3;
  const pcount = latent.length - rcount;
  const weights = latent.slice(0, pcount);
  const residual = latent.slice(pcount);
  const oklab = xyz_to_oklab(reflectance_to_xyz(mix_pigments(weights, K, S)));
  oklab[0] += residual[0];
  oklab[1] += residual[1];
  oklab[2] += residual[2];
  return oklab_to_srgb_mapped(oklab).map((n) => Color.floatToByte(n));
}

export function neuralLatentToRGBPolynomial(latent) {
  const rcount = 3;
  const pcount = latent.length - rcount;
  const weights = latent.slice(0, pcount);
  const residual = latent.slice(pcount);
  const oklab = evalPolynomial(weights);
  oklab[0] += residual[0];
  oklab[1] += residual[1];
  oklab[2] += residual[2];
  return oklab_to_srgb_mapped(oklab).map((n) => Color.floatToByte(n));
}

export function oklab_to_srgb_mapped(oklab, residual = [0, 0, 0]) {
  let srgb = Color.convert(oklab, Color.OKLab, Color.sRGB);
  srgb[0] += residual[0];
  srgb[1] += residual[1];
  srgb[2] += residual[2];

  const lch = Color.convert(srgb, Color.sRGB, Color.OKLCH);
  srgb = Color.gamutMapOKLCH(
    lch,
    Color.sRGBGamut,
    undefined,
    undefined,
    // Color.MapToGray
    Color.MapToAdaptiveCuspL
  );
  return srgb;
}

export function LUTMixer(K, S, N, dataRGBA) {
  // const tmp3A = [0, 0, 0];
  // const tmp3B = [0, 0, 0];

  function delta(a, b, out = []) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
  }

  function predictWithWeight(srgb, c) {
    const oklab = Color.convert(srgb, Color.sRGB, Color.OKLab);
    const mixed_r = mix_pigments(c, K, S);
    const mixed_xyz = reflectance_to_xyz(mixed_r);
    const mixed_oklab = xyz_to_oklab(mixed_xyz);
    const residual = delta(oklab, mixed_oklab);
    return {
      srgb,
      target: oklab,
      oklab: mixed_oklab,
      weights: c,
      residual,
    };
  }

  function predict(srgb) {
    const c = getCMYKFromLUT(srgb, N, dataRGBA);
    // const c = getCMYKFromLUT_Trilinear(srgb, N, dataRGBA);
    return predictWithWeight(srgb, c);
  }

  function kmlerp(a, b, t) {
    const c0 = a.weights;
    const c1 = b.weights;
    const r0 = a.residual;
    const r1 = b.residual;

    // const l1 = a.oklab[0];
    // const l2 = b.oklab[0];

    // const t1 = l1 * (1.0 - t) ** 2;
    // const t2 = l2 * t ** 2;
    // t = t2 / (t1 + t2);
    // t = smoothstep(t, 0, 1);
    // t = adjustableSmoothStep(t, 0.5, 2);

    const concentration = lerpArray(c0, c1, t);
    const residual = lerpArray(r0, r1, t);
    const residualTerm = 1;
    const mixed_r = mix_pigments(concentration, K, S);
    const mixed_xyz = reflectance_to_xyz(mixed_r);
    const mixed_oklab = xyz_to_oklab(mixed_xyz);
    mixed_oklab[0] += residual[0] * residualTerm;
    mixed_oklab[1] += residual[1] * residualTerm;
    mixed_oklab[2] += residual[2] * residualTerm;
    const oklch = Color.convert(mixed_oklab, Color.OKLab, Color.OKLCH);
    return Color.gamutMapOKLCH(
      oklch,
      Color.sRGBGamut,
      Color.sRGB,
      undefined,
      Color.MapToL
    );
  }

  return {
    toLatent(srgb) {
      // const c = getCMYKFromLUT(srgb, N, dataRGBA);
      const c = getCMYKFromLUT_Trilinear(srgb, N, dataRGBA);
      const target_oklab = Color.convert(srgb, Color.sRGB, Color.OKLab);
      const mixed_r = mix_pigments(c, K, S);
      const mixed_xyz = reflectance_to_xyz(mixed_r);
      const mixed_oklab = xyz_to_oklab(mixed_xyz);
      // const mixed_srgb = Color.convert(mixed_oklab, Color.OKLab, Color.sRGB);
      // const rgb = mixed_srgb.map((n) => clamp(n, 0, 1));
      // const rgb = evalPolynomial(c);
      // const r0 = srgb[0] - rgb[0];
      // const r1 = srgb[1] - rgb[1];
      // const r2 = srgb[2] - rgb[2];
      const r0 = target_oklab[0] - mixed_oklab[0];
      const r1 = target_oklab[1] - mixed_oklab[1];
      const r2 = target_oklab[2] - mixed_oklab[2];
      // console.log([r0, r1, r2], mixed_srgb);

      return [...c, r0, r1, r2];
    },
    fromLatent(latent) {
      const residual = latent.slice(4);
      const concentration = latent.slice(0, 4);
      const residualTerm = 1;
      const mixed_r = mix_pigments(concentration, K, S);
      const mixed_xyz = reflectance_to_xyz(mixed_r);
      const mixed_oklab = xyz_to_oklab(mixed_xyz);
      mixed_oklab[0] += residual[0] * residualTerm;
      mixed_oklab[1] += residual[1] * residualTerm;
      mixed_oklab[2] += residual[2] * residualTerm;

      const oklch = Color.convert(mixed_oklab, Color.OKLab, Color.OKLCH);
      console.log(mixed_oklab, oklch);
      return Color.gamutMapOKLCH(
        oklch,
        Color.sRGBGamut,
        Color.sRGB,
        undefined,
        Color.MapToL
      );
      // const c = latent.slice(0, 4);
      // const rgb = evalPolynomial(c);
      // const newRGB = [
      //   rgb[0] + latent[4],
      //   rgb[1] + latent[5],
      //   rgb[2] + latent[6],
      // ];
      // return newRGB;
    },
    predictWithWeight,
    predict,
    kmlerp,
  };
}
