import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Табличные данные
x = np.array([1.0, 1.2, 1.5, 1.6, 1.8, 2.0], dtype=float)
y = np.array([5.000, 6.899, 11.180, 13.133, 18.119, 25.000], dtype=float)

# 1) Натуральный кубический сплайн S(x)
spline = CubicSpline(x, y, bc_type='natural')

# 2) Функция для уравнения: S(x) = 6x + 3  ->  g(x)=0
def g(t: float) -> float:
    return float(spline(t) - (6*t + 3))

# 3) Бисекция (метод половинного деления)
def bisection(func, lo, hi, tol=1e-12, max_iter=200):
    f_lo = func(lo)
    f_hi = func(hi)

    if f_lo == 0.0:
        return lo, 0
    if f_hi == 0.0:
        return hi, 0
    if f_lo * f_hi > 0:
        raise ValueError("На концах [lo, hi] нет смены знака. Бисекция не применима.")

    for it in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        f_mid = func(mid)

        # остановка по невязке или по длине интервала
        if abs(f_mid) < tol or (hi - lo) / 2 < tol:
            return mid, it

        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return mid, max_iter

# 4) Запуск
root, iters = bisection(g, 1.0, 2.0)

print("Корень уравнения S(x) = 6x + 3 на [1,2]:")
print(f"x* = {root:.12f}")
print(f"итераций = {iters}")
print(f"S(x*) = {float(spline(root)):.12f}")
print(f"6x*+3  = {(6*root+3):.12f}")
print(f"невязка g(x*) = {g(root):.3e}")

# 5) График: S(x) и прямая 6x+3
xx = np.linspace(1.0, 2.0, 400)
plt.figure(figsize=(10, 6))
plt.plot(xx, spline(xx), label="Spline S(x)")
plt.plot(xx, 6*xx + 3, label="Line 6x+3")
plt.scatter(x, y, label="Table nodes", zorder=3)
plt.axvline(root, linestyle="--", label=f" x*={root:.6f}")
plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
