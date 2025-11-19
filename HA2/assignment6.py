import numpy as np

# Define priors
priors = {
    "B": 0.30,
    "0": 0.45,
    "8": 0.25
}

# Noise model
p_white_to_black = 0.35   # white pixel misclassified as black
p_black_to_white = 0.20   # black pixel misclassified as white

def pixel_likelihood(true_pixel, observed_pixel):
    """Likelihood for one pixel given noise model."""
    if true_pixel == 1:  # true black pixel
        return 1 - p_black_to_white if observed_pixel == 1 else p_black_to_white
    else:  # true white pixel
        return 1 - p_white_to_black if observed_pixel == 0 else p_white_to_black

def image_likelihood(true_img, obs_img):
    probs = [pixel_likelihood(t, o)
             for t, o in zip(true_img.flatten(), obs_img.flatten())]
    return np.prod(probs)

# Corrected class templates (5x3)
B = np.array([
    [1,1,0],
    [1,0,1],
    [1,1,0],
    [1,0,1],
    [1,1,0]
])

zero = np.array([
    [0,1,0],
    [1,0,1],
    [1,0,1],
    [1,0,1],
    [0,1,0]
])

eight = np.array([
    [0,1,0],
    [1,0,1],
    [0,1,0],
    [1,0,1],
    [0,1,0]
])

# Observed image x
x = np.array([
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,0]
])

classes = {"B": B, "0": zero, "8": eight}

# Compute likelihoods and posteriors
likelihoods = {c: image_likelihood(img, x) for c, img in classes.items()}
unnormalized = {c: likelihoods[c] * priors[c] for c in classes}
total = sum(unnormalized.values())
posteriors = {c: unnormalized[c] / total for c in classes}

# Print results
for c in classes:
    print(f"Class {c}: prior={priors[c]:.2f}, "
          f"likelihood={likelihoods[c]:.3e}, "
          f"posterior={posteriors[c]:.4f}")

map_class = max(posteriors, key=posteriors.get)
print(f"\nMAP estimate: Class {map_class} with posterior={posteriors[map_class]:.4f}")
