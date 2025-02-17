const EPSILON = Number.EPSILON;

export function clamp(value, min, max) {
  return min < max
    ? value < min
      ? min
      : value > max
      ? max
      : value
    : value < max
    ? max
    : value > min
    ? min
    : value;
}

export function clamp01(v) {
  return clamp(v, 0, 1);
}

export function lerp(min, max, t) {
  return min * (1 - t) + max * t;
}

export function inverseLerp(min, max, t) {
  if (Math.abs(min - max) < EPSILON) return 0;
  else return (t - min) / (max - min);
}

export function smoothstep(min, max, t) {
  var x = clamp(inverseLerp(min, max, t), 0, 1);
  return x * x * (3 - 2 * x);
}

export function lerpArray(min, max, t, out = []) {
  if (min.length !== max.length) {
    throw new TypeError(
      "min and max array are expected to have the same length"
    );
  }
  for (var i = 0; i < min.length; i++) {
    out[i] = lerp(min[i], max[i], t);
  }
  return out;
}

export function mod(a, b) {
  return ((a % b) + b) % b;
}

export function degToRad(n) {
  return (n * Math.PI) / 180;
}

export function radToDeg(n) {
  return (n * 180) / Math.PI;
}

export function fract(n) {
  return n - Math.floor(n);
}

export function step(edge, x) {
  if (edge > x) return 0;
  else return 1;
}

export function sign(n) {
  if (n > 0) return 1;
  else if (n < 0) return -1;
  else return 0;
}

export function pingPong(t, length) {
  t = mod(t, length * 2);
  return length - Math.abs(t - length);
}

export function damp(a, b, lambda, dt) {
  return lerp(a, b, 1 - Math.exp(-lambda * dt));
}

export function dampArray(a, b, lambda, dt, out) {
  out = out || [];
  for (var i = 0; i < a.length; i++) {
    out[i] = damp(a[i], b[i], lambda, dt);
  }
  return out;
}

export function mapRange(
  value,
  inputMin,
  inputMax,
  outputMin,
  outputMax,
  clamp
) {
  // Reference:
  // https://openframeworks.cc/documentation/math/ofMath/
  if (Math.abs(inputMin - inputMax) < EPSILON) {
    return outputMin;
  } else {
    var outVal =
      ((value - inputMin) / (inputMax - inputMin)) * (outputMax - outputMin) +
      outputMin;
    if (clamp) {
      if (outputMax < outputMin) {
        if (outVal < outputMax) outVal = outputMax;
        else if (outVal > outputMin) outVal = outputMin;
      } else {
        if (outVal > outputMax) outVal = outputMax;
        else if (outVal < outputMin) outVal = outputMin;
      }
    }
    return outVal;
  }
}
