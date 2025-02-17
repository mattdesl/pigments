// Data is already in 380+5+780 format

import CMF_5 from "./cie1931-xyzbar-380+5+780.js";
import illuminant_d65_5 from "./illuminant-d65-380+5+780.js";
import illuminant_d50_5 from "./illuminant-d50-380+5+780.js";
// import illuminant_c_5 from "./illuminant-c-380+5+780.js";
import { clamp01, lerp } from "./math.js";
import basis from "./cie1931-basis-bt709-380+5+780.js";

export const DATA_WAVELENGTH_MIN = 380;
export const DATA_WAVELENGTH_MAX = 780;
export const DATA_WAVELENGTH_STEP = 5;
export const DATA_WAVELENGTH_COUNT =
  (DATA_WAVELENGTH_MAX + DATA_WAVELENGTH_STEP - DATA_WAVELENGTH_MIN) /
  DATA_WAVELENGTH_STEP;

export const WAVELENGTH_MIN = 380;
export const WAVELENGTH_MAX = 730;
export const WAVELENGTH_STEP = 10;
export const WAVELENGTH_COUNT =
  (WAVELENGTH_MAX + WAVELENGTH_STEP - WAVELENGTH_MIN) / WAVELENGTH_STEP;

const cmf2 = {
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

function interpolateSpectrum(oldWavelengths, oldReflectances, newWavelengths) {
  const newReflectances = [];

  // A simple helper function for linear interpolation between (x0,y0) and (x1,y1)
  function lerp(x, x0, x1, y0, y1) {
    return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
  }

  // For each new wavelength, find where it fits in the old array
  for (const nm of newWavelengths) {
    // If nm exactly matches an old wavelength, just push that reflectance
    let exactIndex = oldWavelengths.indexOf(nm);
    if (exactIndex !== -1) {
      newReflectances.push(oldReflectances[exactIndex]);
      continue;
    }

    // Otherwise, find the two old wavelengths it is between
    for (let i = 0; i < oldWavelengths.length - 1; i++) {
      const x0 = oldWavelengths[i];
      const x1 = oldWavelengths[i + 1];

      if (x0 <= nm && nm < x1) {
        const y0 = oldReflectances[i];
        const y1 = oldReflectances[i + 1];
        const y = lerp(nm, x0, x1, y0, y1);
        newReflectances.push(y);
        break;
      }
    }
  }

  return newReflectances;
}

function createWavelengths(min, max, steps) {
  const wavelengths = [];
  for (let i = min; i <= max; i += steps) {
    wavelengths.push(i);
  }
  return wavelengths;
}

function resample(data) {
  if (data.length !== DATA_WAVELENGTH_COUNT) {
    throw new Error("data has invalid wavelength step");
  }
  if (data.length === WAVELENGTH_COUNT) {
    // same as output
    return data.slice();
  }
  const oldWavelengths = createWavelengths(
    DATA_WAVELENGTH_MIN,
    DATA_WAVELENGTH_MAX,
    DATA_WAVELENGTH_STEP
  );
  const newWavelengths = createWavelengths(
    WAVELENGTH_MIN,
    WAVELENGTH_MAX,
    WAVELENGTH_STEP
  );
  return interpolateSpectrum(oldWavelengths, data, newWavelengths);
}

export const CMF = resample(CMF_5);
export const illuminant_d65 = resample(illuminant_d65_5);
export const illuminant_d50 = resample(illuminant_d50_5);
// export const illuminant_e = resample(illuminant_d65_5).map(() => 1);
// export const illuminant_c = resample(illuminant_c_5);
export const basis_bt709 = resample(basis);

const CIE_CMF_Y = CMF.map((c) => c[1]);

function dotproduct(a, b) {
  let r = 0;
  for (let i = 0; i < a.length; i++) {
    r += a[i] * b[i];
  }
  return r;
}

function linear_to_concentration(l1, l2, t) {
  let t1 = l1 * (1 - t) ** 2;
  let t2 = l2 * t ** 2;
  return t2 / (t1 + t2);
}

export function mix_spectra(R1, R2, t) {
  let l1 = Math.max(1e-14, dotproduct(R1, CIE_CMF_Y));
  let l2 = Math.max(1e-14, dotproduct(R2, CIE_CMF_Y));
  t = linear_to_concentration(l1, l2, t);

  const R = new Array(WAVELENGTH_COUNT);
  for (let i = 0; i < WAVELENGTH_COUNT; i++) {
    const R1_nz = Math.max(1e-14, R1[i]);
    const R2_nz = Math.max(1e-14, R2[i]);

    // single-constant KM theory
    const KS0 = (1 - R1_nz) ** 2 / (2 * R1_nz);
    const KS1 = (1 - R2_nz) ** 2 / (2 * R2_nz);

    // simple interpolation of KS coefficient
    const KS = lerp(KS0, KS1, t);

    // output reflectance
    R[i] = reflectance_mix(KS);
  }
  return R;
}

function reflectance_mix(ks) {
  return 1.0 + ks - Math.sqrt(ks ** 2 + 2.0 * ks);
}

export function linear_sRGB_to_spectra(rgb) {
  const [r, g, b] = rgb;
  const spec = new Array(WAVELENGTH_COUNT);

  // in 0..1 range
  for (var i = 0; i < WAVELENGTH_COUNT; ++i) {
    const xyz = basis_bt709[i];
    spec[i] = xyz[0] * r + xyz[1] * g + xyz[2] * b;
  }
  return spec;
}

// https://stackoverflow.com/questions/1472514/convert-light-frequency-to-rgb
/**
 * A multi-lobe, piecewise Gaussian fit of CIE 1931 XYZ Color Matching Functions by Wyman el al. from Nvidia. The
 * code here is adopted from the Listing 1 of the paper authored by Wyman et al.
 *
 * Reference: Chris Wyman, Peter-Pike Sloan, and Peter Shirley, Simple Analytic Approximations to the CIE XYZ Color
 * Matching Functions, Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 1-11, 2013.
 *
 * @param wavelength wavelength in nm
 * @return XYZ in a double array in the order of X, Y, Z. each value in the range of [0.0, 1.0]
 */
export function wavelength_to_XYZ(wavelength) {
  let x;
  {
    const t1 = (wavelength - 442.0) * (wavelength < 442.0 ? 0.0624 : 0.0374);
    const t2 = (wavelength - 599.8) * (wavelength < 599.8 ? 0.0264 : 0.0323);
    const t3 = (wavelength - 501.1) * (wavelength < 501.1 ? 0.049 : 0.0382);
    x =
      0.362 * Math.exp(-0.5 * t1 * t1) +
      1.056 * Math.exp(-0.5 * t2 * t2) -
      0.065 * Math.exp(-0.5 * t3 * t3);
  }

  let y;
  {
    const t1 = (wavelength - 568.8) * (wavelength < 568.8 ? 0.0213 : 0.0247);
    const t2 = (wavelength - 530.9) * (wavelength < 530.9 ? 0.0613 : 0.0322);

    y = 0.821 * Math.exp(-0.5 * t1 * t1) + 0.286 * Math.exp(-0.5 * t2 * t2);
  }

  let z;
  {
    const t1 = (wavelength - 437.0) * (wavelength < 437.0 ? 0.0845 : 0.0278);
    const t2 = (wavelength - 459.0) * (wavelength < 459.0 ? 0.0385 : 0.0725);

    z = 1.217 * Math.exp(-0.5 * t1 * t1) + 0.681 * Math.exp(-0.5 * t2 * t2);
  }

  return [x, y, z];
}

// export function spectra_to_XYZ(reflectance, illuminant = illuminant_d65) {
//   let X = 0,
//     Y = 0,
//     Z = 0,
//     S = 0;
//   let scaleByIlluminant = illuminant !== false;
//   let hasIlluminant = scaleByIlluminant && Array.isArray(illuminant);
//   for (let i = 0; i < reflectance.length; i++) {
//     let I = 1;
//     if (hasIlluminant)
//       I = illuminant[i]; // SPD
//     else if (typeof illuminant === "number") I = illuminant; // fixed value
//     var intensity = reflectance[i] * I;
//     X += CMF[i][0] * intensity;
//     Y += CMF[i][1] * intensity;
//     Z += CMF[i][2] * intensity;
//     S += CMF[i][1];
//   }
//   if (scaleByIlluminant) {
//     X /= S;
//     Y /= S;
//     Z /= S;
//   }
//   // scale down by 100 so that XYZ remains in 0..1 range as with rest of lib
//   return [X / 100, Y / 100, Z / 100];
// }

export function spectra_to_XYZ(reflectance) {
  let X = 0,
    Y = 0,
    Z = 0;
  const CMF = cmf2.wavelengths.data;
  for (let i = 0; i < reflectance.length; i++) {
    var intensity = reflectance[i];
    X += CMF[0][i] * intensity;
    Y += CMF[1][i] * intensity;
    Z += CMF[2][i] * intensity;
  }
  return [X, Y, Z];
}

// export function wavelength_to_XYZ(len) {
//   if (len < WAVELENGTH_MIN || len > WAVELENGTH_MAX) return [0, 0, 0];

//   len -= WAVELENGTH_MIN;
//   const index = Math.floor(len / WAVELENGTH_STEP);
//   const offset = len - WAVELENGTH_STEP * index;
//   if (indx < 0) return CMF[0];
//   if (index >= CMF.length - 1) return CMF[CMF.length - 1];

//   const x0 = index * WAVELENGTH_STEP;
//   const x1 = x0 + WAVELENGTH_STEP;
//   const v0 = CMF[index];
//   const v1 = CMF[index + 1];

//   // const xyz = [0, 0, 0];
//   // for (let i = 0; i < 3; i++) {
//   //   // xyz[i] = lerp(v0[i], )
//   //   // xyz[i] = v0[i] + (offset * (v1[i] - v0[i])) / (x1 - x0);
//   // }
//   // return xyz;
// }
