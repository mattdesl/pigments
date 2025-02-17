// import CMF_5 from "./cie1931-xyzbar-380+5+780.js";
// import illuminant_d65_5 from "./color/illuminant-d65-380+5+780.js";

// const targetWavelength = [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730]

// export const WAVELENGTH_MIN = 380;
// export const WAVELENGTH_MAX = 780;
// export const WAVELENGTH_STEP = 5;

import * as spectra from "./color/spectra.js";
import * as Color from "@texel/color";

const srgbIn = [0, 87, 238];
const srgbl = Color.convert(
  srgbIn.map((n) => n / 0xff),
  Color.sRGB,
  Color.sRGBLinear
);
const refl = spectra.linear_sRGB_to_spectra(srgbl);
const srgbOut = Color.convert(
  spectra.spectra_to_XYZ(refl),
  Color.XYZ,
  Color.sRGB
).map((n) => Color.floatToByte(n));
console.log(srgbIn);
console.log(srgbOut);
console.log(
  spectra.WAVELENGTH_MIN,
  spectra.WAVELENGTH_MAX,
  spectra.WAVELENGTH_STEP
);
console.log(refl);

// const srgbOut2 = Color.convert(
//   spectra.spectra_to_XYZ2(refl),
//   Color.XYZ,
//   Color.sRGB
// ).map((n) => Color.floatToByte(n));
// console.log(srgbOut2);

// const
