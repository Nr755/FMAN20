#!/usr/bin/env python3
import numpy as np

def dimension(h, w):
    """Number of degrees of freedom (pixels) in an h×w grayscale image."""
    return h * w

def e_pixel(h, w, r, c, dtype=int):
    """
    Canonical 'pixel' basis element e_{r,c} for an h×w image:
    a matrix with a 1 at (r,c) and 0 elsewhere. r,c are 0-based.
    """
    E = np.zeros((h, w), dtype=dtype)
    E[r, c] = 1
    return E

def pretty_small(M):
    """Compact printing for small matrices."""
    return np.array2string(M, formatter={'int':lambda x: f"{x:1d}"},
                           max_line_width=1000)

def show_first_basis_elements(h, w, how_many=4):
    """
    Print the first few canonical basis elements in row-major order:
    (0,0), (0,1), (0,2), (0,3), then (1,0), ...
    For large images, only print the nonzero coordinate and a small crop.
    """
    print(f"\nFirst {how_many} basis elements for {h}×{w}:")
    idx = 0
    r, c = 0, 0
    for k in range(how_many):
        r = k // w
        c = k % w
        print(f"  e[{r},{c}]  (1 at row {r}, col {c}, zeros elsewhere)")
        # show a tiny 5×7 crop around the nonzero (clipped to image bounds)
        R0, R1 = max(0, r-2), min(h, r+3)
        C0, C1 = max(0, c-3), min(w, c+4)
        crop = np.zeros((R1-R0, C1-C0), dtype=int)
        crop[r-R0, c-C0] = 1
        print(pretty_small(crop))

if __name__ == "__main__":
    # Part A: 2×4 images
    HA, WA = 2, 4
    kA = dimension(HA, WA)
    print(f"Part A — dimension k for {HA}×{WA} grayscale images: k = {kA}")

    # An explicit standard basis for A: e_{r,c} with a single 1 at pixel (r,c)
    # We list them in row-major order and print each as a 2×4 matrix.
    basis_A = [e_pixel(HA, WA, r, c) for r in range(HA) for c in range(WA)]
    print("\nExample basis for A (standard pixel basis, row-major order):")
    for i, Ei in enumerate(basis_A, start=1):
        r, c = divmod(i-1, WA)
        print(f"e{i} (which is e[{r},{c}]):")
        print(pretty_small(Ei))

    # Part B: 1200×3000 images
    HB, WB = 1200, 3000
    kB = dimension(HB, WB)
    print(f"\nPart B — dimension k for {HB}×{WB} grayscale images: k = {kB} "
          "(= 1200×3000)")

    print("\nHow the basis elements are chosen (standard construction):")
    print("  For each pixel location (r,c), define e_{r,c} to be 1 at (r,c) and 0 elsewhere.")
    print("  The set { e_{r,c} : 0≤r<1200, 0≤c<3000 } has 1200×3000 elements and spans the space.")
    print("  They are linearly independent, so this set is a basis.")

    # Show the first few basis elements for the large image without allocating all of them.
    show_first_basis_elements(HB, WB, how_many=4)

    print("\nHow the remaining basis elements are constructed:")
    print("  Continue in row-major order:")
    print("    e[0,0], e[0,1], e[0,2], ..., e[0,2999], e[1,0], e[1,1], ..., e[1199,2999].")
    print("  In code, use e_pixel(1200, 3000, r, c) to materialize any specific element on demand.")