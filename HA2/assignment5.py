import numpy as np

# Observed 4x4 image (black=1, white=0)
obs = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,1,0,0]
])

# Generate the 4 hypotheses: vertical line in column 1-4
classes = []
for col in range(4):
    img = np.zeros((4,4), dtype=int)
    img[:,col] = 1
    classes.append(img)

# Priors: col1=0.3, col2=0.2, col3=0.2, col4=0.3
priors = [0.3, 0.2, 0.2, 0.3]

# Noise probability
eps = 0.2

def pixel_likelihood(true_pixel, observed_pixel, eps):
    if true_pixel == observed_pixel:
        return 1 - eps
    else:
        return eps

def class_likelihood(true_img, obs_img, eps):
    probs = [pixel_likelihood(t, o, eps) 
             for t, o in zip(true_img.flatten(), obs_img.flatten())]
    return np.prod(probs)

# Compute posteriors
likelihoods = [class_likelihood(c, obs, eps) for c in classes]
unnormalized_post = [L * p for L, p in zip(likelihoods, priors)]
total = sum(unnormalized_post)
posteriors = [u / total for u in unnormalized_post]

# Print results
for i, (prior, like, post) in enumerate(zip(priors, likelihoods, posteriors), 1):
    print(f"Class {i}: prior={prior:.2f}, likelihood={like:.4e}, posterior={post:.4f}")

map_class = np.argmax(posteriors) + 1
print(f"\nMAP estimate: Class {map_class} with posterior={max(posteriors):.4f}")
