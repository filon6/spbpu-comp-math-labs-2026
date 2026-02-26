import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = np.array([1.0, 1.2, 1.5, 1.6, 1.8, 2.0], dtype=float)
y = np.array([5.000, 6.899, 11.180, 13.133, 18.119, 25.000], dtype=float)

spline = CubicSpline(x, y, bc_type='natural')

def g(t: float) -> float:
    return float(spline(t) - (6*t + 3))

def bisection(function, left_bound, right_bound, tolerance=1e-12, max_iterations=200):
    f_left = function(left_bound)
    f_right = function(right_bound)

    if f_left == 0.0:
        return left_bound, 0
    if f_right == 0.0:
        return right_bound, 0
    if f_left * f_right > 0:
        raise ValueError("На концах интервала нет смены знака. Метод бисекции неприменим.")

    for iteration in range(1, max_iterations + 1):
        midpoint = (left_bound + right_bound) / 2
        f_mid = function(midpoint)

        if abs(f_mid) < tolerance or (right_bound - left_bound) / 2 < tolerance:
            return midpoint, iteration

        if f_left * f_mid <= 0:
            right_bound = midpoint
            f_right = f_mid
        else:
            left_bound = midpoint
            f_left = f_mid

    return midpoint, max_iterations

solution_x, iterations = bisection(g, 1.0, 2.0)

print("Корень уравнения S(x) = 6x + 3 на [1,2]:")
print(f"x* = {solution_x:.12f}")
print(f"S(x*) = {float(spline(solution_x)):.12f}")
print(f"6x*+3 = {(6*solution_x+3):.12f}")

xx = np.linspace(1.0, 2.0, 400)
plt.figure(figsize=(10, 6))
plt.plot(xx, spline(xx), label="Spline S(x)")
plt.plot(xx, 6*xx + 3, label="Line 6x+3")
plt.scatter(x, y, label="Table nodes", zorder=3)
plt.axvline(solution_x, linestyle="--", label=f" x*={solution_x:.6f}")
plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
