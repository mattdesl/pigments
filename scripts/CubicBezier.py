# https://stackoverflow.com/questions/11696736/recreating-css3-transitions-cubic-bezier-curve
import math

class UnitBezier:
    epsilon = 1e-6  # Precision

    def __init__(self, p1x, p1y, p2x, p2y):
        # Endpoints b0=(0,0) and b3=(1,1) are implicit.
        # Pre-calculate the polynomial coefficients.
        self.cx = 3.0 * p1x
        self.bx = 3.0 * (p2x - p1x) - self.cx
        self.ax = 1.0 - self.cx - self.bx

        self.cy = 3.0 * p1y
        self.by = 3.0 * (p2y - p1y) - self.cy
        self.ay = 1.0 - self.cy - self.by

    def sampleCurveX(self, t):
        # x(t) = ((ax*t + bx)*t + cx)*t
        return ((self.ax * t + self.bx) * t + self.cx) * t

    def sampleCurveY(self, t):
        # y(t) = ((ay*t + by)*t + cy)*t
        return ((self.ay * t + self.by) * t + self.cy) * t

    def sampleCurveDerivativeX(self, t):
        # dx/dt = (3*ax*t + 2*bx)*t + cx
        return (3.0 * self.ax * t + 2.0 * self.bx) * t + self.cx

    def solveCurveX(self, x, epsilon):
        # First try a few iterations of Newton's method
        t2 = x
        for i in range(8):
            x2 = self.sampleCurveX(t2) - x
            if abs(x2) < epsilon:
                return t2
            d2 = self.sampleCurveDerivativeX(t2)
            if abs(d2) < epsilon:
                break
            t2 = t2 - x2 / d2

        # Fall back to bisection method
        t0 = 0.0
        t1 = 1.0
        t2 = x
        if t2 < t0:
            return t0
        if t2 > t1:
            return t1
        while t0 < t1:
            x2 = self.sampleCurveX(t2)
            if abs(x2 - x) < epsilon:
                return t2
            if x > x2:
                t0 = t2
            else:
                t1 = t2
            t2 = (t1 - t0) * 0.5 + t0
        return t2

    def solve(self, x, epsilon):
        # Find parameter t for which x(t)=x and then return y(t).
        return self.sampleCurveY(self.solveCurveX(x, epsilon))

def cubic_bezier_1D (p1x, p1y, p2x, p2y, t):
  curve = UnitBezier(p1x, p1y, p2x, p2y)
  return curve.solve(t, UnitBezier.epsilon)

# Example usage:
if __name__ == '__main__':
  print(cubic_bezier_1D(0.25, 0, 0.5, 1, 0.5))
  print(cubic_bezier_1D(0, 0, 1, 1, 0.5))
  print(cubic_bezier_1D(0, 0, 1, 1, 0.25))
  print(cubic_bezier_1D(0, 0, 1, 1, 0.75))