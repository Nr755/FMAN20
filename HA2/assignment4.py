import numpy as np

# Define the three classes (true images) and their priors
classA = np.array([[0,1],[1,0]])
classB = np.array([[1,0],[0,1]])
classC = np.array([[1,1],[1,0]])
classes = [classA, classB, classC]
priors = [0.25, 0.5, 0.25]

# Observed noisy image
obs = np.array([[0,1],[1,1]])

def pixel_likelihood(true_pixel, observed_pixel, eps):
    if true_pixel == observed_pixel:
        return 1 - eps
    else:
        return eps

def class_likelihood(true_img, obs_img, eps):
    probs = [pixel_likelihood(t, o, eps) 
             for t, o in zip(true_img.flatten(), obs_img.flatten())]
    return np.prod(probs)

def compute_posteriors(obs, classes, priors, eps):
    likelihoods = [class_likelihood(c, obs, eps) for c in classes]
    unnormalized_post = [L * p for L, p in zip(likelihoods, priors)]
    total = sum(unnormalized_post)
    return [u / total for u in unnormalized_post]

# Run for both epsilon = 0.1 and 0.5
for eps in [0.1, 0.5]:
    print(f"\nResults for epsilon = {eps:.2f}:")
    posteriors = compute_posteriors(obs, classes, priors, eps)
    for i, post in enumerate(posteriors, 1):
        print(f"Posterior P(Class {i} | obs) = {post:.4f}")
    map_class = np.argmax(posteriors) + 1
    print(f"MAP estimate: Class {map_class} (prob={max(posteriors):.4f})")
