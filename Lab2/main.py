import numpy as np
from scipy.linalg import lu_factor, lu_solve

def build_matrix_A(d, N=5):
    A = np.ones((N, N), dtype=float)
    for i in range(N):
        A[i, i] = d
    return A

def matrix_multiply(A, B):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

    return C

def matrix_subtract(A, B):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            C[i, j] = A[i, j] - B[i, j]

    return C

def compute_matrix_A_inverse(A):
    N = A.shape[0]
    LU, piv = lu_factor(A)

    A_inv = np.zeros((N, N), dtype=float)

    for i in range(N):
        e = np.zeros(N)
        e[i] = 1.0
        x = lu_solve((LU, piv), e)
        A_inv[:, i] = x


    return A_inv

def compute_matrix_R(A, A_inv):
    N = A.shape[0]
    E = np.eye(N)
    term = matrix_multiply(A_inv, A)
    R = matrix_subtract(term, E)
    return R

def norm_matrix_R(R):
    N = R.shape[0]
    max_sum = 0.0

    for k in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += abs(R[k, j])
        if row_sum > max_sum:
            max_sum = row_sum

    return max_sum

N = 5
d_values = [1.01, 1.001, 1.0001]

print("d        ||R||              cond")

for d in d_values:
    A = build_matrix_A(d, N)
    A_inv = compute_matrix_A_inverse(A)
    R = compute_matrix_R(A, A_inv)
    R_norm = norm_matrix_R(R)

    A_norm = norm_matrix_R(A)
    A_inv_norm = norm_matrix_R(A_inv)
    cond = A_inv_norm * A_norm
    print(f"{d:<8g} {R_norm:.10e}   {cond:<8g}")