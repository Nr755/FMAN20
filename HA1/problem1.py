import numpy as np

def sample_and_quantize(m=7, n=7, levels=16):
    # sampling grid: x from 0 to 1 (left to right), y from 0 to 1 (top to bottom)
    x = np.linspace(0.0, 1.0, n)   # columns: left -> right
    y = np.linspace(1.0, 0.0, m)   # rows: top -> bottom

    # sample f(x, y) = x * (1 - y) on the grid
    F = np.outer(1 - y, x)         # shape (m, n), row 0 = y=0 (bottom), row m-1 = y=1 (top)

    # quantize to 0..levels-1, 0 -> 0, 1 -> levels-1
    Q = np.round((levels - 1) * F).astype(int)

    return Q

if __name__ == "__main__":
    Q = sample_and_quantize(m=7, n=7, levels=16)
    print(Q)