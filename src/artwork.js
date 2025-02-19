import CMYK_LUT from "../data/cmyk_lut.png";
import mixbox from "mixbox";
import spectral from "spectral.js";
import canvasSketch from "canvas-sketch";
import * as Color from "@texel/color";
import { K, S } from "../data/adjusted-primaries-optimized-feb13-black.json";
import {
  neuralRGBToLatentPolynomial,
  neuralLatentToRGBPolynomial,
  lerpArray,
  lerpLatents,
  getImageData,
  loadImage,
  neuralLatentToRGB,
  normalized,
  NUM_PIGMENTS,
} from "./pigments.js";

// function getRandomPartition(n) {
//   // Generate n-1 random breakpoints
//   const breaks = [];
//   for (let i = 0; i < n - 1; i++) {
//     breaks.push(Math.random());
//   }

//   // Sort the breakpoints
//   breaks.sort((a, b) => a - b);

//   // Compute the differences between the breakpoints
//   const result = [];
//   result.push(breaks[0]); // from 0 to the first breakpoint

//   for (let i = 1; i < breaks.length; i++) {
//     result.push(breaks[i] - breaks[i - 1]);
//   }

//   // The last segment from the last breakpoint to 1
//   result.push(1 - breaks[breaks.length - 1]);

//   return result;
// }

// // Gamma sampler using the Marsaglia and Tsang method (for alpha >= 1)
// // For alpha < 1, one common trick is to use:
// //   Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha), where U ~ Uniform(0, 1)
// function gammaSample(alpha) {
//   if (alpha < 1) {
//     // Use the boost method
//     const u = Math.random();
//     return gammaSample(alpha + 1) * Math.pow(u, 1 / alpha);
//   }

//   const d = alpha - 1 / 3;
//   const c = 1 / Math.sqrt(9 * d);
//   while (true) {
//     let x = 0;
//     let v = 0;
//     do {
//       x = randn_bm();
//       v = 1 + c * x;
//     } while (v <= 0);
//     v = v * v * v;
//     const u = Math.random();
//     if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
//     if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
//   }
// }

// // Standard Normal sampler using Box-Muller
// function randn_bm() {
//   let u = 0,
//     v = 0;
//   while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
//   while (v === 0) v = Math.random();
//   return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
// }

// function sampleDirichlet(alpha, n) {
//   const samples = [];
//   for (let i = 0; i < n; i++) {
//     samples.push(gammaSample(alpha));
//   }
//   const sum = samples.reduce((acc, val) => acc + val, 0);
//   return samples.map((x) => x / sum);
// }

canvasSketch(
  ({ width, height }) => {
    const residual = Array(3).fill(0);

    return ({ context }) => {
      const xSize = NUM_PIGMENTS;
      const ySize = NUM_PIGMENTS;
      for (let y = 0; y < ySize; y++) {
        for (let x = 0; x < xSize; x++) {
          const u = x / xSize;
          const v = y / ySize;

          let pigments = Array(5)
            .fill(0)
            .map(() => 0);
          pigments[x] = 1;
          pigments[y] = 1;

          pigments = normalized(pigments);

          const latent = [...pigments, ...residual];
          const rgb = neuralLatentToRGB(latent);

          const cellWidth = width / xSize;
          const cellHeight = height / ySize;
          context.fillStyle = `rgb(${rgb.join(",")})`;
          context.fillRect(width * u, height * v, cellWidth, cellHeight);

          context.fillStyle = "white";
          const fontSize = width * 0.01;
          context.font = `${fontSize}px monospace`;
          context.textAlign = "center";
          context.textBaseline = "middle";
          // context.fillText(
          //   pigments.map((n) => n).join(", "),
          //   width * u + cellWidth / 2,
          //   height * v + cellHeight / 2
          // );
        }
      }
    };
  },
  {
    dimensions: [2048, 2048],
    animate: false,
  }
);
