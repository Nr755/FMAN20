import scipy
import numpy as np
import matplotlib.pyplot as plt


data = scipy.io.loadmat("HA1/inl1_to_students_python/assignment1bases.mat")
stacks = data["stacks"]
bases = data["bases"]

bases = np.array([bases[0][i] for i in range(3)])
bases = np.transpose(bases, (0, 3, 1, 2))

imgs = np.transpose(stacks[0][0], (2, 0, 1))
faces = np.transpose(stacks[0][1], (2, 0, 1))

# ========= helpers: inner product, norm, projection =========

def fro_inner(A, B):
    """Frobenius inner product <A,B> = sum_ij A_ij * B_ij."""
    return float(np.sum(A * B))

def fro_norm(A):
    """Frobenius norm ||A|| = sqrt(<A,A>)."""
    return float(np.sqrt(fro_inner(A, A)))

def project_onto_basis(u, basis4):
    """
    Orthogonal projection of image u (H×W) onto span{basis4[k]} for k=0..3.
    Works even if the basis is not orthonormal.

    Returns:
      up      : projected image (H×W)
      coeffs  : length-4 vector of coordinates
      errnorm : ||u - up||_F
    """
    # Build tall design matrix B with one column per basis element
    # Solve min_c ||vec(u) - B c||_2 via least squares.
    H, W = u.shape
    B = basis4.reshape(4, -1).T            # shape (H*W, 4)
    y = u.ravel()                           # shape (H*W,)
    coeffs, *_ = np.linalg.lstsq(B, y, rcond=None)

    # Reconstruct projection from coefficients
    up = np.tensordot(coeffs, basis4, axes=(0, 0))  # shape (H, W)
    err = fro_norm(u - up)
    return up, coeffs, err

def gram_matrix(basis4):
    """Gram matrix G_ij = <phi_i, phi_j> for a 4-image basis (4,H,W)."""
    return np.array([[fro_inner(basis4[i], basis4[j]) for j in range(4)] for i in range(4)])


# ========= quick sanity about shapes =========
# bases: (3, 4, 19, 19)   -> three bases, each with 4 basis images
# imgs : (400, 19, 19)    -> general images
# faces: (400, 19, 19)    -> face images
print("bases shape:", bases.shape, "imgs shape:", imgs.shape, "faces shape:", faces.shape)

# ========= show the three bases are (nearly) orthonormal =========
for bi, B in enumerate(bases, start=1):
    G = gram_matrix(B)
    print(f"\nBasis {bi} Gram matrix:\n{np.array_str(G, precision=4, suppress_small=True)}")
    print("Orthonormal?", np.allclose(G, np.eye(4), atol=1e-10))

# ========= evaluate mean error norms for each stack vs each basis =========
def evaluate_stack_vs_bases(stack, bases):
    """
    For each basis, project every image in 'stack' and return:
      means: list of mean error norms (len=3)
      all_errors: list of lists; per-basis error norms over the stack
    """
    means, all_errors = [], []
    for bi, B in enumerate(bases, start=1):
        errs = []
        for u in stack:
            _, _, e = project_onto_basis(u, B)
            errs.append(e)
        all_errors.append(errs)
        means.append(float(np.mean(errs)))
        print(f"Basis {bi}: mean error = {means[-1]:.6f}")
    return means, all_errors

print("\nEvaluating general images (stack 1)...")
means_imgs, _ = evaluate_stack_vs_bases(imgs, bases)

print("\nEvaluating faces (stack 2)...")
means_faces, _ = evaluate_stack_vs_bases(faces, bases)

# ========= print a compact table =========
print("\nMean error norms (lower is better)")
print("rows = test set, cols = basis 1..3")
table = np.vstack([means_imgs, means_faces])
print(np.array_str(table, precision=6, suppress_small=True))

# ========= plotting (a few examples + basis elements) =========
def show_examples(stack, title, n=6):
    """Plot n example images from a stack."""
    n = min(n, len(stack))
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.atleast_2d(axes)
    fig.suptitle(title)
    for k in range(rows*cols):
        ax = axes[k // cols, k % cols]
        ax.axis("off")
        if k < n:
            ax.imshow(stack[k], cmap="gray")
    plt.tight_layout()
    plt.show()

def show_basis(B, title):
    """Plot the four basis images of a single basis (4,H,W)."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    fig.suptitle(title)
    for i in range(4):
        axes[i].imshow(B[i], cmap="gray")
        axes[i].set_title(f"$\\phi_{i+1}$")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# show a few images from each test set
show_examples(imgs, "A few general test images (stack 1)", n=6)
show_examples(faces, "A few face images (stack 2)", n=6)

# show the four basis elements for each of the three bases
for bi, B in enumerate(bases, start=1):
    show_basis(B, f"Basis {bi} – four elements")

# ========= optional: inspect one projection visually =========
def visualize_projection(u, basis4, title="Example projection"):
    up, coeffs, err = project_onto_basis(u, basis4)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(u, cmap="gray");  axes[0].set_title("original");   axes[0].axis("off")
    axes[1].imshow(up, cmap="gray"); axes[1].set_title("projection"); axes[1].axis("off")
    axes[2].imshow(u - up, cmap="gray"); axes[2].set_title(f"residual\n||r||={err:.3g}"); axes[2].axis("off")
    fig.suptitle(title + f"\ncoeffs = {np.array2string(coeffs, precision=3)}")
    plt.tight_layout(); plt.show()

# Example visualization: first image of each stack projected on basis 1
visualize_projection(imgs[0],  bases[0], "Stack 1 / Basis 1")
visualize_projection(faces[0], bases[0], "Stack 2 / Basis 1")
