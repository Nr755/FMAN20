import numpy as np
import matplotlib.pyplot as plt

# Define g(x) as given in the assignment
def g(x):
    x = np.array(x)
    result = np.zeros_like(x, dtype=float)
    
    # |x| <= 1: cosine bump
    mask1 = np.abs(x) <= 1
    result[mask1] = np.cos((np.pi / 2) * x[mask1])
    
    # 1 < |x| <= 2: cubic polynomial
    mask2 = (np.abs(x) > 1) & (np.abs(x) <= 2)
    result[mask2] = -(np.pi/2) * (np.abs(x[mask2])**3 - 5*np.abs(x[mask2])**2 + 8*np.abs(x[mask2]) - 4)
    
    # |x| > 2 → remains zero
    return result

# Given signal f(i), i = 1,...,7
f = np.array([1, 4, 6, 8, 7, 5, 3])
indices = np.arange(1, len(f)+1)

# Interpolation function
def F_g(x):
    # sum over all sample points
    return np.sum([g(x - i) * f_i for i, f_i in zip(indices, f)], axis=0)

# Plot g(x)
x_vals = np.linspace(-3, 3, 600)
plt.figure(figsize=(7,4))
plt.plot(x_vals, g(x_vals), color="blue", label=r"$g(x)$")
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.title(r"Kernel $g(x)$ for $-3 \leq x \leq 3$")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# Plot interpolation F_g(x)
x_vals = np.linspace(1, 7, 600)
plt.figure(figsize=(7,4))
plt.plot(x_vals, F_g(x_vals), color="red", label=r"$F_g(x)$")
plt.scatter(indices, f, color="black", zorder=5, label="samples $f(i)$")
plt.title(r"Interpolation $F_g(x)$ using kernel $g(x)$")
plt.xlabel("x")
plt.ylabel("F_g(x)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
