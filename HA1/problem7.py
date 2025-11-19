#!/usr/bin/env python3
import numpy as np

# -------- images (4x3 each) --------
phi1 = (1/3) * np.array([
    [ 1,  1,  1],
    [ 1,  0,  1],
    [-1, -1, -1],
    [ 0, -1,  0]
], dtype=float)

phi2 = (1/2) * np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 0,  0,  0],
    [ 0,  0,  0]
], dtype=float)

phi3 = (1/2) * np.array([
    [ 0,  0,  0],
    [ 0,  0,  0],
    [ 1,  0, -1],
    [ 1,  0, -1]
], dtype=float)

phi4 = (1/3) * np.array([
    [ 0,  1,  0],
    [ 1,  1,  1],
    [ 1,  0,  1],
    [ 1,  1,  1]
], dtype=float)

f = np.array([
    [11,  5, -3],
    [12,  1, -1],
    [ 1, -4, -6],
    [ 5, -3, -2]
], dtype=float)

phis = [phi1, phi2, phi3, phi4]

# image inner product + norm (Frobenius)
def inner(A, B):
    # <A,B> = sum_ij A_ij B_ij
    return float(np.sum(A * B))

def norm(A):
    # ||A|| = sqrt(<A,A>)
    return float(np.sqrt(inner(A, A)))

# orthonormality check
G = np.array([[inner(a, b) for b in phis] for a in phis])  # Gram matrix
is_orthonormal = np.allclose(G, np.eye(4), atol=1e-12)

# coefficients for best approximation f_a
# If the basis is orthonormal: x_i = <f, phi_i>.
# (General case would solve G x = b with b_i = <f, phi_i>.)
b = np.array([inner(f, p) for p in phis])
if is_orthonormal:
    x = b.copy()
else:
    x = np.linalg.solve(G, b)

# reconstruction and error
f_a = sum(xi * pi for xi, pi in zip(x, phis))
res = f - f_a

# report information
np.set_printoptions(precision=4, suppress=True)
print("Gram matrix G = <phi_i, phi_j>:\n", G, "\n")
print("Orthonormal basis? ", is_orthonormal)
print()
print("Coefficients x (so that f_a = Σ x_i phi_i):")
print(x, "\n")
print("Approximation f_a:\n", f_a, "\n")
print("Residual (f - f_a):\n", res, "\n")
print(f"||f|| = {norm(f):.6f}")
print(f"||residual|| = {norm(res):.6f}")
rel = norm(res) / (norm(f) + 1e-12)
print(f"Relative error = {100*rel:.4f}%")