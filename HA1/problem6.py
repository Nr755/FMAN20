#!/usr/bin/env python3
import numpy as np

# define the three "images" (2x2 matrices)
# u = [[ 4, -3],
#      [ 2, -1]]
u = np.array([[4, -3],
              [2, -1]], dtype=float)

# v = 1/2 * [[ 1, -1],
#            [-1,  1]]
v = 0.5 * np.array([[ 1, -1],
                    [-1,  1]], dtype=float)

# w = 1/2 * [[ 1, -1],
#            [ 1, -1]]
w = 0.5 * np.array([[ 1, -1],
                    [ 1, -1]], dtype=float)

# Frobenius inner product and induced norm
def inner(A: np.ndarray, B: np.ndarray) -> float:
    """⟨A,B⟩ = sum_ij A_ij * B_ij"""
    return float(np.sum(A * B))

def norm(A: np.ndarray) -> float:
    """‖A‖ = sqrt(⟨A,A⟩)"""
    return float(np.sqrt(inner(A, A)))

# projection of u onto span{v, w}
def project_onto_span(u: np.ndarray, basis: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal projection of u onto span(basis), for any (not necessarily orthonormal) basis.
    Returns (proj_u, coeffs), where proj_u = sum_k coeffs[k] * basis[k].
    """
    # Build Gram matrix G_ij = ⟨b_i, b_j⟩ and right-hand side b_i = ⟨u, b_i⟩
    k = len(basis)
    G = np.zeros((k, k), dtype=float)
    b = np.zeros((k,), dtype=float)
    for i in range(k):
        b[i] = inner(u, basis[i])
        for j in range(k):
            G[i, j] = inner(basis[i], basis[j])

    # Solve G c = b for coefficients c
    c = np.linalg.solve(G, b)
    proj = sum(ci * bi for ci, bi in zip(c, basis))
    return proj, c

if __name__ == "__main__":
    # norms
    nu, nv, nw = norm(u), norm(v), norm(w)

    # dot products
    uv, uw, vw = inner(u, v), inner(u, w), inner(v, w)

    # orthonormal check for {v,w}
    ortho = np.isclose(vw, 0.0)
    unit_v = np.isclose(nv, 1.0)
    unit_w = np.isclose(nw, 1.0)
    is_orthonormal = ortho and unit_v and unit_w

    # projection of u onto span{v,w}
    P, coeffs = project_onto_span(u, [v, w])
    residual = u - P

    # ----- print results -----
    np.set_printoptions(precision=4, suppress=True)

    print("u =\n", u)
    print("v =\n", v)
    print("w =\n", w)
    print()
    print(f"‖u‖ = {nu:.6f}")
    print(f"‖v‖ = {nv:.6f}")
    print(f"‖w‖ = {nw:.6f}")
    print()
    print(f"⟨u,v⟩ = {uv:.6f}")
    print(f"⟨u,w⟩ = {uw:.6f}")
    print(f"⟨v,w⟩ = {vw:.6f}")
    print()
    print(f"{'{v,w}'} orthonormal?  {is_orthonormal}  "
          f"(⟨v,w⟩≈0: {ortho}, ‖v‖≈1: {unit_v}, ‖w‖≈1: {unit_w})")
    print()
    print("Projection of u onto span{v,w}:")
    print("  coefficients c =", coeffs)       # u_proj = c[0]*v + c[1]*w
    print("  u_proj =\n", P)
    print("  residual (u - u_proj) =\n", residual)
    print(f"  ‖residual‖ = {norm(residual):.6f}")