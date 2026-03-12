import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.linalg import lu_factor, lu_solve


def froot(x):
    return x**2 - np.tan(np.pi * x / 3.0)

def integrand_tau(x):
    return np.sqrt(x - x**3)

def compute_tau():
    integral = quad(integrand_tau, 0.0, 0.5)[0]
    return 0.4493982 * integral

def bisection(function, left_bound, right_bound, tolerance = 1e-7, max_iterations = 200):
    f_left = function(left_bound)
    f_right = function(right_bound)

    if f_left == 0.0:
        return left_bound
    if f_right == 0.0:
        return right_bound
    if f_left * f_right > 0:
        raise ValueError("На концах интервала нет смены знака.")

    for iteration in range(1, max_iterations + 1):
        midpoint = (left_bound + right_bound) / 2
        f_mid = function(midpoint)

        if abs(f_mid) < tolerance or (right_bound - left_bound) / 2 < tolerance:
            return midpoint

        if f_left * f_mid <= 0:
            right_bound = midpoint
            f_right = f_mid
        else:
            left_bound = midpoint
            f_left = f_mid

    return midpoint

def find_smallest_positive_root():
    offset = 1e-6
    max_intervals = 20

    asymptotes = [1.5 + 3.0 * k for k in range(max_intervals + 1)]
    intervals = [(0.0 + offset , asymptotes[0] - offset )]
    for k in range(1, len(asymptotes)):
        intervals.append((asymptotes[k - 1] + offset , asymptotes[k] - offset ))

    for a, b in intervals:
        x_grid = np.linspace(a, b, 1000)
        f_values = froot(x_grid)

        for i in range(len(x_grid) - 1):
            v1, v2 = f_values[i], f_values[i + 1]

            if np.isnan(v1) or np.isnan(v2) or np.isinf(v1) or np.isinf(v2):
                continue

            if v1 == 0:
                return x_grid[i]

            if v1 * v2 < 0:
                return bisection(froot, x_grid[i], x_grid[i + 1])

    raise RuntimeError("Корень не найден.")

def compute_omega0():
    root_x = find_smallest_positive_root()
    omega0 = 1.142206 * root_x
    return omega0, root_x

def solve_for_ABC():
    M = np.array([
        [46.0, -24.0, -42.0],
        [-24.0, 16.0, 18.0],
        [-42.0, 18.0, 49.0]
    ])

    rhs = np.array([50.0, -30.0, -35.0])

    LU, piv = lu_factor(M)
    sol = lu_solve((LU, piv), rhs)

    A, B, C = sol
    return A, B, C

def rhs_system(t, y, omega0, tau, mu):
    V, Vp, z = y
    zp = (V**2 - z) / tau
    dterm = (1 - z) * Vp - V * zp
    Vpp = -(omega0**2) * V + 2 * mu * dterm
    return np.array([Vp, Vpp, zp])

def solve_system(A, B, C, omega0, tau, mu, T, tol, n_points = 1000):
    t_eval = np.linspace(0.0, T, n_points)

    sol = solve_ivp(
        fun=lambda t, y: rhs_system(t, y, omega0, tau, mu),
        t_span=(0.0, T),
        y0=np.array([A, B, C]),
        method="RK45",
        t_eval=t_eval,
        rtol=tol,
        atol=tol
    )

    if not sol.success:
        raise RuntimeError("solve_ivp не справился: " + str(sol.message))

    t = sol.t
    V = sol.y[0]
    Vp = sol.y[1]
    z = sol.y[2]

    return t, V, Vp, z

def plot_results(t, V, Vp):
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("t")
    plt.ylabel("V(t)")
    plt.title("График (V,t)")
    plt.grid(True)

    plt.figure()
    plt.plot(V, Vp)
    plt.xlabel("V")
    plt.ylabel("V'")
    plt.title("График (V,V')")
    plt.grid(True)

    plt.show()

def max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


T = 10.0
mu = 0.1
tol = 1e-7

tau = compute_tau()
omega0, x_root = compute_omega0()
A, B, C = solve_for_ABC()

print("Параметры:")
print("tau =", tau)
print("x* =", x_root)
print("omega0 =", omega0)
print("A =", A)
print("B =", B)
print("C =", C)
print()

t, V, Vp, z = solve_system(A, B, C, omega0, tau, mu, T, tol)

delta = 0.05
t2, V2, Vp2, z2 = solve_system(
    A * (1 + delta),
    B * (1 + delta),
    C * (1 + delta),
    omega0,
    tau,
    mu,
    T,
    tol,
    len(t)
)

sens_V = max_abs_diff(V, V2)
sens_Vp = max_abs_diff(Vp, Vp2)
sens_z = max_abs_diff(z, z2)

print("Оценка устойчивости:")
print("max |ΔV| =", sens_V)
print("max |ΔV'| =", sens_Vp)
print("max |Δz| =", sens_z)
print()

plot_results(t, V, Vp)
