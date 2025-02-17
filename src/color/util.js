import { OKLab } from "@texel/color";

export const OK2_SCALE = 2; // how much to scale A and B factors by

const K1 = 0.206;
const K2 = 0.03;
const K3 = (1.0 + K1) / (1.0 + K2);

export const LToLr = (x) =>
  0.5 *
  (K3 * x - K1 + Math.sqrt((K3 * x - K1) * (K3 * x - K1) + 4 * K2 * K3 * x));

export const LrToL = (x) => (x ** 2 + K1 * x) / (K3 * (x + K2));

// // https://github.com/Myndex/deltaphistar
// export const contrast = (l0, l1) => {
//   const v = (Math.abs((LToLr(l0)*100) ** 1.618 - (LToLr(l1)*100) ** 1.618) ** 0.618) * 1.414 - 40;
//   return v;
// };

export const OKLrab = {
  id: "oklrab",
  base: OKLab,
  toBase: (oklrab, out = [0, 0, 0]) => {
    out[0] = LrToL(oklrab[0]);
    out[1] = oklrab[1];
    out[2] = oklrab[2];
    return out;
  },
  fromBase: (oklab, out = [0, 0, 0]) => {
    out[0] = LToLr(oklab[0]);
    out[1] = oklab[1];
    out[2] = oklab[2];
    return out;
  },
};

export function deltaEOKSquared(oklab1, oklab2, distanceFactor = [1, 2, 2]) {
  let dL = (oklab1[0] - oklab2[0]) * distanceFactor[0];
  let da = (oklab1[1] - oklab2[1]) * distanceFactor[1];
  let db = (oklab1[2] - oklab2[2]) * distanceFactor[2];
  return dL * dL + da * da + db * db;
}
