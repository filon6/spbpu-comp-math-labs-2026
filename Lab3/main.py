import numpy as np
from scipy.integrate import solve_ivp

def f(t, z):
    x1, x2 = z
    dx1 = -155.0 * x1 - 750.0 * x2 + np.sin(1.0 + t)
    dx2 = x1 + np.cos(1.0 - t) + t + 1.0
    return np.array([dx1, dx2], dtype=float)

def solve_rkf45(t0, t1, z0, h_print, eps):
    t_eval = np.arange(t0, t1 + 1e-12, h_print)

    sol = solve_ivp(
        fun=f,
        t_span=(t0, t1),
        y0=np.array(z0, dtype=float),
        method="RK45",
        t_eval=t_eval,
        rtol=eps,
        atol=eps
    )

    if not sol.success:
        raise RuntimeError("solve_ivp не смог решить задачу")

    return sol.t, sol.y.T

def rk2_step(t_n, z_n, h):
    k1 = h * f(t_n, z_n)
    k2 = h * f(t_n + 2.0 * h / 3.0, z_n + 2.0 * k1 / 3.0)
    return z_n + (k1 + 3.0 * k2) / 4.0

def solve_rk2(t0, t1, z0, h_int, h_print):
    ratio = h_print / h_int
    if abs(ratio - round(ratio)) > 1e-12:
        raise ValueError("h_print должен быть кратен h_int.")

    out_every = int(round(ratio))

    n_steps = int(round((t1 - t0) / h_int))
    if abs(t0 + n_steps * h_int - t1) > 1e-12:
        raise ValueError("Интервал [t0,t1] должен делиться на h_int.")

    t_solution = [t0]
    z_solution = [np.array(z0, dtype=float)]

    t = t0
    z = np.array(z0, dtype=float)

    for step in range(1, n_steps + 1):

        z = rk2_step(t, z, h_int)
        t = t0 + step * h_int

        if step % out_every == 0:
            t_solution.append(t)
            z_solution.append(z.copy())

    return np.array(t_solution), np.array(z_solution)

def print_table(title, t, z):
    print(title)
    print("t         x1                    x2")

    for t_i, (x1, x2) in zip(t, z):
        print(f"{t_i:0.2f}   {x1: .12e}   {x2: .12e}")

    print()

def print_comparison(t, z_ref, z_test, name_ref="RKF45", name_test="RK2"):
    print(f"Сравнение: {name_test} - {name_ref}")
    print("t       dx1                   dx2")

    for t_i, (x1_ref, x2_ref), (x1_test, x2_test) in zip(t, z_ref, z_test):
        print(f"{t_i:0.2f}   {(x1_test-x1_ref): .12e}   {(x2_test-x2_ref): .12e}")

    print()


t0, t1 = 0.0, 1.0
z0 = [5.0, -1.0]

h_print = 0.04
eps = 1e-4

t_rk45, z_rk45 = solve_rkf45(t0, t1, z0, h_print, eps)
print_table("RKF45, EPS=1e-4, h_print=0.04", t_rk45, z_rk45)

h_int1 = 0.02
t_rk2_002, z_rk2_002 = solve_rk2(t0, t1, z0, h_int1, h_print)
print_table("RK2, h_int=0.02, h_print=0.04", t_rk2_002, z_rk2_002)
print_comparison(t_rk45, z_rk45, z_rk2_002)

h_int2 = 0.01
t_rk2_fine, z_rk2_fine = solve_rk2(t0, t1, z0, h_int2, h_print)
print_table("RK2, h_int=0.01, h_print=0.04", t_rk2_fine, z_rk2_fine)
print_comparison(t_rk45, z_rk45, z_rk2_fine)

A = np.array([
    [-155.0, -750.0],
    [1.0, 0.0]
])

eigvals = np.linalg.eigvals(A)
lambda_max = max(abs(eigvals))
h_rk2 = 2 / lambda_max

print("Оценка критического шага:")
print("RK2  h <", h_rk2)