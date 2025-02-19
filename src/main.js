import CMYK_LUT from "../data/cmyk_lut.png";
import mixbox from "mixbox";
import spectral from "spectral.js";

import * as Color from "@texel/color";
import { K, S } from "../data/adjusted-primaries-optimized-feb13-black.json";
import {
  neuralRGBToLatentPolynomial,
  neuralLatentToRGBPolynomial,
  lerpArray,
  lerpLatents,
  getImageData,
  loadImage,
} from "./pigments.js";

async function run() {
  const image = await loadImage(CMYK_LUT);
  const pixels = getImageData(image);
  const LUT_N = pixels.width;
  // const cmyk = getCMYKFromLUT([1, 0.5, 0.5], LUT_N, pixels.data, image.width);

  // const idx1d = 59;
  // const idx = idx1d * 4;
  // const c8 = pixels.data[idx + 0];
  // const m8 = pixels.data[idx + 1];
  // const y8 = pixels.data[idx + 2];

  // // Convert to [0..1] floats
  // const c = c8 / 255.0;
  // const m = m8 / 255.0;
  // const y = y8 / 255.0;
  // let k = 1 - (c + m + y);
  // k = Math.max(0.0, k); // or clamp to ensure no negatives
  // console.log("DATA", [c, m, y, k]);

  // const mixer = LUTMixer(K, S, LUT_N, pixels.data, image.width);
  // console.log(srgb0);
  // const aa = mixer.predict([10, 52, 162].map((n) => n / 255));
  // const b = mixer.predict(srgb1);
  // console.log(aa.weights);

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  const types = [
    "oklab",
    "spectral.js",
    // "lut",
    // "neural net",
    "neural net std",
    "mixbox",
    // "neural net nor",
    // "mixbox nor",
    // "srgb-linear",
  ];
  canvas.width = 80 * types.length;
  canvas.height = 256;
  const steps = canvas.height;

  // prettier-ignore
  const COLORS = [
    // [[255, 255, 255],[0, 0, 0]],
    [[0,33,133],[252,210,0]]

    // [[10, 52, 162],[248, 210, 71]],
    // [[0, 33, 133],[252, 210, 0]],
    // [[243, 240, 247],[0, 176, 0]],
    // [[ 26, 10, 83 ],[ 255, 255, 255 ]],
    // [[ 0, 66, 170 ],[ 255, 255, 255 ]],
    // [[195,209,23],[0,97,255]],
    // [[255,170,0],[0,97,255]],
    // [[255,106,0],[142,230,255]],
    // [[124,42,0],[142,230,255]],
  ]

  const srgbColors = COLORS.map((pair) =>
    pair.map((rgb) => rgb.map((n) => n / 255))
  );

  // const mixed_r = mix_pigments([0, 0, 0, 1], K, S);
  // const mixed_xyz = reflectance_to_xyz(mixed_r);
  // const mixed_oklab = xyz_to_oklab(mixed_xyz);
  // context.fillStyle = Color.serialize(mixed_oklab, Color.OKLab, Color.sRGB);
  // context.fillRect(0, 0, canvas.width, canvas.height);

  const pair = srgbColors[0].slice();
  // const [srgb0, srgb1] = pair;
  const hexes = pair.map((rgb) => Color.RGBToHex(rgb));
  const tstep = document.createElement("input");
  tstep.setAttribute("checked", "checked");

  draw(...pair, tstep.checked);
  document.body.appendChild(canvas);

  for (let i = 0; i < hexes.length; i++) {
    const hexInitial = hexes[i];
    const input = document.createElement("input");
    input.type = "color";
    input.value = hexInitial;
    input.addEventListener("input", (e) => {
      const hex = e.target.value;
      const rgb = Color.hexToRGB(hex);
      pair[i] = rgb;
      draw(...pair, tstep.checked);
    });
    document.body.appendChild(input);
  }

  tstep.type = "checkbox";
  const label = document.createElement("label");
  label.textContent = "Non-linear";
  // document.body.appendChild(label);
  tstep.oninput = () => draw(...pair, tstep.checked);
  label.appendChild(tstep);

  function draw(srgb0, srgb1, smoothT = false) {
    // const w0 = [
    //   0.9983604973519696, 9.983604973519697e-5, 9.983604973519697e-5,
    //   0.0014398305485599293,
    // ];
    // const w1 = [
    //   0.0013280275771837744, 8.61645098880271e-5, 0.861645098880271,
    //   0.1369407090326572,
    // ];

    // console.log("rgb", rgb_to_index_3D(rgbPair[0], LUT_N));
    // console.log("rgb 1d", rgb_to_index_1D(rgbPair[0], LUT_N));
    // console.log("");
    // const a = mixer.predict(srgb0);
    // const b = mixer.predict(srgb1);

    // const latentA = mixer.toLatent(srgb0);
    // const latentB = mixer.toLatent(srgb1);

    const oklab0 = Color.convert(srgb0, Color.sRGB, Color.OKLab);
    const oklab1 = Color.convert(srgb1, Color.sRGB, Color.OKLab);
    // const modelW0ReLU = predictNeuralNetwork(oklab0, dataReLU);
    // const modelW1ReLU = predictNeuralNetwork(oklab1, dataReLU);

    // const modelW0Std = predictNeuralNetwork(oklab0, dataStd);
    // const modelW1Std = predictNeuralNetwork(oklab1, dataStd);

    // rgbToLatent(srgb0)

    // console.log(srgb0, modelW0);
    // console.log(srgb1, modelW1);

    // console.log("a weights", a.weights);
    // console.log("b weights", b.weights);

    const rgb0 = srgb0.map((n) => Color.floatToByte(n));
    const rgb1 = srgb1.map((n) => Color.floatToByte(n));
    const neural0 = neuralRGBToLatentPolynomial(rgb0);
    const neural1 = neuralRGBToLatentPolynomial(rgb1);

    // console.log(a, b);
    for (let j = 0; j < types.length; j++) {
      const type = types[j];
      const sliceWidth = canvas.width / types.length;
      for (let i = 0; i < steps; i++) {
        const t = i / (steps - 1);

        let color;
        if (type == "lut") {
          const latent = lerpArray(
            latentA,
            latentB,
            smoothT ? OKLtoConcentration(oklab0, oklab1, t) : t
          );
          const rgb = mixer.fromLatent(latent).map((n) => clamp(n, 0, 1));
          color = Color.serialize(rgb, Color.sRGB);
          // const oklab = mixer.kmlerp(a, b, t);
          // color = Color.serialize(oklab, Color.sRGB);
        } else if (type == "mixbox") {
          const rgb0 = srgb0.map((n) => n * 0xff);
          const rgb1 = srgb1.map((n) => n * 0xff);
          color = mixbox.lerp(rgb0, rgb1, t);
        } else if (type == "mixbox nor") {
          const rgb0 = srgb0.map((n) => n * 0xff);
          const rgb1 = srgb1.map((n) => n * 0xff);
          const cmyk0 = mixbox.rgbToLatent(...rgb0).slice(0, 4);
          const cmyk1 = mixbox.rgbToLatent(...rgb1).slice(0, 4);
          const cmyk = lerpArray(cmyk0, cmyk1, t);
          const oklab = xyz_to_oklab(
            reflectance_to_xyz(
              mix_pigments([cmyk[0], cmyk[2], cmyk[1], cmyk[3]], K, S)
            )
          );
          const lch = Color.convert(oklab, Color.OKLab, Color.OKLCH);
          color = Color.serialize(
            Color.gamutMapOKLCH(
              lch,
              Color.sRGBGamut,
              undefined,
              undefined,
              Color.MapToAdaptiveCuspL
            ),
            Color.sRGB
          );
          // color = Color.serialize(evalPolynomial(cmyk), Color.sRGB);
        } else if (type == "spectral.js") {
          const hex0 = Color.RGBToHex(srgb0);
          const hex1 = Color.RGBToHex(srgb1);
          color = spectral.mix(hex0, hex1, t);
        } else if (type == "neural net" || type == "neural net std") {
          // const neural = lerpLatents(oklab0, oklab1, neural0, neural1, t);

          // const linearL0 = LToLr(oklab0[0]);
          // const linearL1 = LToLr(oklab1[0]);

          // const gamma = 1.0;
          // const avgL = (linearL0 + linearL1) / 2;
          // // Increase gamma for darker colors: darker means lower L.
          // const adaptiveGamma = gamma + 3.0 * (1.0 - avgL); // adjust factor as needed

          // const c1 = linearL0 * Math.pow(1.0 - t, adaptiveGamma);
          // const c2 = linearL1 * Math.pow(t, adaptiveGamma);
          // const tMod = smoothstep(t, 0, 1); //c2 / (c1 + c2);

          // const neural = lerpArray(neural0, neural1, t);
          const neural = lerpLatents(neural0, neural1, t);

          // const rgb = neuralLatentToRGB(neural);
          const rgb = neuralLatentToRGBPolynomial(neural);

          // const t1 = t;
          // const t2 = 1 - t;
          // const cx1 = oklab0[0] ** 3;
          // const cx2 = oklab1[0] ** 3;
          // const GAMMA = 2.2;
          // let c1 = ((1 - t) * cx1 ** (1 / GAMMA)) ** GAMMA;
          // let c2 = (t * cx2 ** (1 / GAMMA)) ** GAMMA;
          // const num_pigments = K.length;
          // let [K1, S1] = mix_pigments_to_K_S(
          //   neural0.slice(0, num_pigments),
          //   K,
          //   S
          // );
          // let [K2, S2] = mix_pigments_to_K_S(
          //   neural1.slice(0, num_pigments),
          //   K,
          //   S
          // );
          // const mK = Array(K1.length)
          //   .fill()
          //   .map((_, i) => {
          //     return (K1[i] * c1 + K2[i] * c2) / (c1 + c2);
          //   });
          // const mS = Array(K1.length)
          //   .fill()
          //   .map((_, i) => {
          //     return (S1[i] * c1 + S2[i] * c2) / (c1 + c2);
          //   });

          // const R = K_S_to_R(mK, mS);
          // const mixedOklab = xyz_to_oklab(reflectance_to_xyz(R));
          // const residual = lerpArray(
          //   neural0.slice(num_pigments),
          //   neural1.slice(num_pigments),
          //   t
          // );
          // mixedOklab[0] += residual[0];
          // mixedOklab[1] += residual[1];
          // mixedOklab[2] += residual[2];
          // const rgb = oklab_to_srgb_mapped(mixedOklab).map((n) =>
          //   Color.floatToByte(n)
          // );
          color = `rgb(${rgb.join(",")})`;
          // console.log(rgb);

          // const isStd = type.includes("std");
          // const modelW0 = isStd ? modelW0Std : modelW0ReLU;
          // const modelW1 = isStd ? modelW1Std : modelW1ReLU;

          // const modelOklab0 = xyz_to_oklab(
          //   reflectance_to_xyz(mix_pigments(modelW0, K, S))
          // );
          // const modelSrgb0 = oklab_to_srgb_mapped(modelOklab0);
          // const modelOklab1 = xyz_to_oklab(
          //   reflectance_to_xyz(mix_pigments(modelW1, K, S))
          // );
          // const modelSrgb1 = oklab_to_srgb_mapped(modelOklab1);

          // // const target0 = srgb0;
          // // const target1 = srgb1;
          // // const result0 = modelSrgb0;
          // // const result1 = modelSrgb1;

          // const target0 = oklab0;
          // const target1 = oklab1;
          // const result0 = modelOklab0;
          // const result1 = modelOklab1;

          // const modelLatent0 = [
          //   ...modelW0,
          //   target0[0] - result0[0],
          //   target0[1] - result0[1],
          //   target0[2] - result0[2],
          // ];
          // const modelLatent1 = [
          //   ...modelW1,
          //   target1[0] - result1[0],
          //   target1[1] - result1[1],
          //   target1[2] - result1[2],
          // ];

          // const latent = lerpLatents(
          //   oklab0,
          //   oklab1,
          //   modelLatent0,
          //   modelLatent1,
          //   t,
          //   !smoothT
          //   // smoothT ? OKLtoConcentration(oklab0, oklab1, t) : t
          // );
          // const rcount = 3;
          // const pigmentCount = latent.length - rcount;
          // const pigments = latent.slice(0, pigmentCount);
          // // .map((n, i) => (i <= 2 && smoothT ? smoothstep(n, 0, 1) : n));
          // const residual = latent.slice(pigmentCount);
          // // debugger;
          // // const cmyk = latent.slice(0, 4);
          // const mixedOklab = xyz_to_oklab(
          //   reflectance_to_xyz(mix_pigments(pigments, K, S))
          // );
          // mixedOklab[0] += residual[0];
          // mixedOklab[1] += residual[1];
          // mixedOklab[2] += residual[2];
          // const mixedSrgb = oklab_to_srgb_mapped(mixedOklab);
          // color = Color.serialize(mixedSrgb, Color.sRGB);
        } else if (type == "neural net nor") {
          // const latent = lerpArray(
          //   modelLatent0,
          //   modelLatent1,
          //   smoothT ? OKLtoConcentration(oklab0, oklab1, t) : t
          // );
          // const cmyk = latent.slice(0, 4);
          // const mixedOklab = xyz_to_oklab(
          //   reflectance_to_xyz(mix_pigments(cmyk, K, S))
          // );
          // // mixedOklab[0] += latent[4];
          // // mixedOklab[1] += latent[5];
          // // mixedOklab[2] += latent[6];
          // const lch = Color.convert(mixedOklab, Color.OKLab, Color.OKLCH);
          // color = Color.serialize(
          //   Color.gamutMapOKLCH(
          //     lch,
          //     Color.sRGBGamut,
          //     undefined,
          //     undefined,
          //     Color.MapToAdaptiveCuspL
          //   ),
          //   Color.sRGB
          // );
        } else if (type == "srgb-linear") {
          const a = Color.convert(srgb0, Color.sRGB, Color.sRGBLinear);
          const b = Color.convert(srgb1, Color.sRGB, Color.sRGBLinear);
          const c = lerpArray(a, b, t);
          color = Color.serialize(
            Color.convert(c, Color.sRGBLinear, Color.sRGB),
            Color.sRGB
          );
        } else if (type == "oklab") {
          const lab = lerpArray(oklab0, oklab1, t);
          const lch = Color.convert(lab, Color.OKLab, Color.OKLCH);

          color = Color.serialize(
            Color.gamutMapOKLCH(
              lch,
              Color.sRGBGamut,
              undefined,
              undefined,
              Color.MapToAdaptiveCuspL
            ),
            Color.sRGB
          );
        }

        context.fillStyle = color;

        const sliceHeight = canvas.height / steps;
        context.fillRect(
          Math.floor(j * sliceWidth),
          Math.floor(i * sliceHeight),
          Math.ceil(sliceWidth),
          Math.ceil(sliceHeight)
        );

        // let color = spectral.mix('#00357B', '#D79900', 0.5);
      }
      context.fillStyle = "white";
      context.textAlign = "center";
      context.fillText(type.toUpperCase(), j * sliceWidth + sliceWidth / 2, 20);
    }
  }
}

run();
