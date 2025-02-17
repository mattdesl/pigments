import { lerpArray } from "./color/math.js";
import { spectra_to_XYZ, linear_sRGB_to_spectra } from "./color/spectra.js";
import dataset from "./data/paintmixing-data.json" with { type: "json" };
import gradient from "./data/gradient.json" with { type: "json" };
import * as Color from "@texel/color";
import { createCanvas } from "canvas";
import * as fs from "fs/promises";

// console.log(
//   Color.convert(
//     [0, 87, 238].map((n) => n / 0xff),
//     Color.sRGB,
//     Color.OKLab
//   )
// );

const data = {};
dataset.forEach((d) => {
  data[d.name] = d;
});

const PIGMENTS = ["white", "cold yellow", "magenta", "blue green shade"];

const concentrationsUnnormalized = [0.58, 0.859, 0.175, 0.001];

function mixSpectrum(paints) {
  let totalWeight = paints.map((p) => p[1]).reduce((a, b) => a + b, 0);
  if (totalWeight == 0) totalWeight = 1;

  let mixed_K = data[paints[0][0]].K.wavelengths.map(() => 0);
  let mixed_S = data[paints[0][0]].S.wavelengths.map(() => 0);
  if (mixed_K.length !== mixed_S.length)
    throw new Error("expected similar wavelengths");
  for (let paint of paints) {
    let K = data[paint[0]].K.values;
    let S = data[paint[0]].S.values;
    let componentWeight = paint[1] / totalWeight;
    for (let i = 0; i < K.length; i++) {
      mixed_K[i] += K[i] * componentWeight;
      mixed_S[i] += S[i] * componentWeight;
    }
  }

  const mixed_R = new Array(mixed_K.length).fill(0);
  for (let i = 0; i < mixed_R.length; i++) {
    const denom = mixed_K[i] + mixed_S[i];
    const omega = mixed_S[i] / denom;

    const R = omega / (2 - omega + 2 * Math.sqrt(1 - omega));
    mixed_R[i] = R;
  }
  return mixed_R;
}

// const paints = [
//   ["white", 0.58],
//   ["cold yellow", 0.083],
//   ["magenta", 0.175],
//   ["blue green shade", 0.001],
// ];

const mixes = [
  {
    target: [22, 73, 130],
    paints: [
      ["white", 0.313],
      ["cold yellow", 0.001],
      ["magenta", 0.001],
      ["blue green shade", 1],
    ],
  },
  {
    target: [242, 204, 0],
    paints: [
      ["white", 0.007],
      ["cold yellow", 1],
      ["magenta", 0.001],
      ["blue green shade", 0.001],
    ],
  },
];

function mixRGB(paints) {
  return Color.convert(
    spectra_to_XYZ(mixSpectrum(paints)),
    Color.XYZ,
    Color.sRGB
  ).map((n) => Color.floatToByte(n));
}

function kmlerp(a, b, t) {
  const total0 = a.paints.map((p) => p[1]).reduce((a, b) => a + b, 0);
  const c0 = a.paints.map((p) => p[1] / total0);
  const total1 = b.paints.map((p) => p[1]).reduce((a, b) => a + b, 0);
  const c1 = b.paints.map((p) => p[1] / total1);
  const rgb0 = a.target;
  const rgb1 = b.target;

  const paints0 = c0.map((c, i) => [a.paints[i][0], c]);
  const predictedRGB0 = mixRGB(paints0);
  const paints1 = c1.map((c, i) => [b.paints[i][0], c]);
  const predictedRGB1 = mixRGB(paints1);

  const ck = lerpArray(c0, c1, t);
  const paintsMixed = ck.map((c, i) => [a.paints[i][0], c]);
  const mixedRGB = mixRGB(paintsMixed);

  const residuals0 = predictedRGB0.map((n, i) => rgb0[i] - n);
  const residuals1 = predictedRGB1.map((n, i) => rgb1[i] - n);

  const residualsMixed = lerpArray(residuals0, residuals1, t);
  const residualScale = 0;
  const finalRGB = mixedRGB.map(
    (n, i) => n + residualScale * residualsMixed[i]
  );

  return finalRGB;
  // const mixed = mixSpectrum(paints);
  // const xyz = spectra_to_XYZ(mixed);
  // const rgbOut = Color.convert(xyz, Color.XYZ, Color.sRGB).map((n) =>
  //   Color.floatToByte(n)
  // );
  // // const spectrum1 = mixSpectrum(a.paints);
}

function toPaints(paintNames, v) {
  const target = v.target;
  const predicted = v.predicted.map((x) => x * 0xff);
  const concentrations = v.concentrations;
  const totalConcentrations = concentrations.reduce((a, b) => a + b, 0);
  const paintsConcentrations = concentrations.map(
    (c) => c / totalConcentrations
  );
  const paints = paintsConcentrations.map((p, i) => [
    paintNames[i],
    paintsConcentrations[i],
  ]);
  return {
    target,
    predicted,
    paints,
  };
}

const canvas = createCanvas(64, gradient.values.length);
const paintNames = gradient.paint_names;
const a = toPaints(paintNames, gradient.values[0]);
const b = toPaints(paintNames, gradient.values[gradient.values.length - 1]);
const ctx = canvas.getContext("2d");

console.log(a, b);

for (let i = 0; i < gradient.values.length; i++) {
  const v = gradient.values[i];
  const y = i;
  const t = i / (gradient.values.length - 1);
  // const target = v.target;
  // const predicted = v.predicted.map((x) => Color.floatToByte(x));

  const rgb = kmlerp(a, b, t);
  ctx.fillStyle = `rgb(${rgb.join(",")})`;
  ctx.fillRect(0, y, canvas.width, 1);

  // const p = toPaints(paintNames, v);
  // const predictedJS = mixRGB(p.paints);
  // console.log(predicted, predictedJS);
  // const rgbExpected = kmlerp(a, b, t);
  // console.log(predicted.map((n) => Math.round(n)));
}

const png = canvas.toBuffer();
await fs.writeFile("gradient.png", png);
// const srgbOut = kmlerp(mixes[0], mixes[1], 0.5);
// const hex = Color.RGBToHex(srgbOut.map((n) => n / 0xff));
// console.log(srgbOut);
// console.log(hex);
// const paints = [
//   ["white", 0.313],
//   ["cold yellow", 0.001],
//   ["magenta", 0.001],
//   ["blue green shade", 1],
// ];
// console.log(mixRGB(paints));
// // const rgbIn = [196, 103, 141];
// const spec = mixSpectrum(paints);
// const xyz = spectra_to_XYZ(spec);
// const rgb2 = Color.convert(xyz, Color.XYZ, Color.sRGB).map((n) =>
//   Color.floatToByte(n)
// );
// console.log(rgb2);

// const totalWeight = paints.map((p) => p[1]).reduce((a, b) => a + b, 0);
// const concentrations = paints.map((c) => c[1] / totalWeight);
// const hex1 = "#002185";
// const hex2 = "#FCD200";

// spectra_to_XYZ

// const rgb = [72, 25, 128];
// const lrgb = Color.convert(
//   rgb.map((n) => n / 0xff),
//   Color.sRGB,
//   Color.sRGBLinear
// );
// const spec = linear_sRGB_to_spectra(lrgb);
// const XYZ = spectra_to_XYZ(spec);
// const rgb2 = Color.convert(XYZ, Color.XYZ, Color.sRGB).map((n) =>
//   Color.floatToByte(n)
// );
// console.log(rgb, rgb2);

// const XYZ = spectra_to_XYZ(spec);
