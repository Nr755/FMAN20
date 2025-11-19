import numpy as np
from scipy.stats import norm

# Data
class1 = np.array([0.4003, 0.3988, 0.3998, 0.3997, 0.4010, 0.3995, 0.3991])
class2 = np.array([0.2554, 0.3139, 0.2627, 0.3802, 0.3287, 0.3160, 0.2924])
class3 = np.array([0.5632, 0.7687, 0.0524, 0.7586, 0.4243, 0.5005, 0.6769])

# Split into train/test
train1, test1 = class1[:4], class1[4:]
train2, test2 = class2[:4], class2[4:]
train3, test3 = class3[:4], class3[4:]
train_sets = [train1, train2, train3]

# --- Nearest Neighbour ---
def classify_nn(x, train_sets):
    dists = [np.min(np.abs(train - x)) for train in train_sets]
    return np.argmin(dists) + 1  # return class index

tests = [(test1, 1), (test2, 2), (test3, 3)]
results_nn = []
correct_nn = 0

print("Nearest Neighbour Results:")
for test_samples, true_class in tests:
    for x in test_samples:
        pred = classify_nn(x, train_sets)
        results_nn.append((x, true_class, pred))
        print(f"x={x:.4f}, true={true_class}, predicted={pred}")
        if pred == true_class:
            correct_nn += 1

print(f"NN Correct: {correct_nn}/{len(results_nn)}\n")


# --- Gaussian Classification ---
means = [0.4, 0.32, 0.55]
stds = [0.01, 0.05, 0.2]

def classify_gauss(x, means, stds):
    likelihoods = [norm.pdf(x, m, s) for m, s in zip(means, stds)]
    return np.argmax(likelihoods) + 1

tests_all = [(class1, 1), (class2, 2), (class3, 3)]
results_gauss = []
correct_gauss = 0

print("Gaussian Classification Results:")
for samples, true_class in tests_all:
    for x in samples:
        pred = classify_gauss(x, means, stds)
        results_gauss.append((x, true_class, pred))
        print(f"x={x:.4f}, true={true_class}, predicted={pred}")
        if pred == true_class:
            correct_gauss += 1

print(f"Gaussian Correct: {correct_gauss}/{len(results_gauss)}")
