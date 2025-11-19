#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from PIL import Image

from benchmarking.benchmark_assignment3 import benchmark_assignment3


# ============================================================
# Utils
# ============================================================

FEATURE_NAMES = [
    "norm_area", "aspect_ratio", "eccentricity", "extent", "solidity", "euler_norm",
    "hu0c", "hu1c",
    *[f"z{i}{j}" for i in range(3) for j in range(3)],  # 3x3 zones
    *[f"hog{i}" for i in range(36)]  # 2x2 cells, 9 orientations
]


def _to_float01(im):
    im = im.astype(np.float32)
    mn, mx = im.min(), im.max()
    if mx > mn:
        im = (im - mn) / (mx - mn)
    else:
        im = np.zeros_like(im, dtype=np.float32)
    return im

def _binarize(im_gray01):
    """White strokes on black background -> foreground=1 (Otsu)."""
    t = threshold_otsu(im_gray01)
    return (im_gray01 >= t).astype(np.uint8)

def _pad_to_square(img, pad_value=0):
    h, w = img.shape
    if h == w:
        return img
    if h > w:
        pad = h - w
        l = pad // 2
        r = pad - l
        return np.pad(img, ((0,0), (l,r)), constant_values=pad_value)
    else:
        pad = w - h
        t = pad // 2
        b = pad - t
        return np.pad(img, ((t,b), (0,0)), constant_values=pad_value)

def _resize_binary(img, size):
    # nearest keeps stroke crisp
    pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    pil = pil.resize((size, size), resample=Image.NEAREST)
    return (np.array(pil) > 127).astype(np.uint8)

def _center_of_mass(binary):
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        h, w = binary.shape
        return (h - 1) / 2.0, (w - 1) / 2.0
    return ys.mean(), xs.mean()

def _translate_to_center(binary):
    """Integer shift so COM -> image center; returns shifted image and (dy,dx)."""
    h, w = binary.shape
    cy, cx = _center_of_mass(binary)
    ty, tx = (h - 1) / 2.0, (w - 1) / 2.0
    dy, dx = int(round(ty - cy)), int(round(tx - cx))
    out = np.zeros_like(binary)
    y0s, y0d = max(0, -dy), max(0, dy)
    x0s, x0d = max(0, -dx), max(0, dx)
    y1s, y1d = min(h, h - dy), min(h, h + dy)
    x1s, x1d = min(w, w - dx), min(w, w + dx)
    out[y0d:y1d, x0d:x1d] = binary[y0s:y1s, x0s:x1s]
    return out, (dy, dx)

def _tight_crop_rows(img_bin):
    rows = img_bin.sum(axis=1)
    nz = np.where(rows > 0)[0]
    if len(nz) == 0:
        return img_bin
    return img_bin[nz[0]:nz[-1]+1, :]

def _preprocess_digit(mask, target=28):
    """
    mask: binary (0/1) for one component (already bbox-cropped)
    Steps: tight row-crop -> pad square -> resize -> COM-center
    """
    m = (mask > 0).astype(np.uint8)
    m = _tight_crop_rows(m)
    m = _pad_to_square(m, pad_value=0)
    m = _resize_binary(m, target)
    before = _center_of_mass(m)
    m, shift = _translate_to_center(m)
    after = _center_of_mass(m)
    return m, {"before_COM": before, "after_COM": after, "shift": shift}

def _hu_compress(h):
    """
    Compress Hu moments with log1p+sign then tanh -> [-1,1], rescale to [0,1].
    Keeps numeric stability and comparability across samples.
    """
    s = np.sign(h)
    v = np.log1p(np.abs(h))  # >= 0
    z = np.tanh(3.0 * s * v)  # [-1,1] with gentle slope control
    return 0.5 * (z + 1.0)


# ============================================================
# Segmentation
# ============================================================

def im2segment(im):
    import numpy as np

    """
    Segment an image of light digits on a dark background into one mask per digit.

    Input
      im : numpy array of shape (H, W) or (H, W, 3)
           The input image. If it is color, we convert it to grayscale.

    Output
      A Python list of masks [S1, S2, ..., Sn].
      Each mask Si is a numpy array of shape (H, W) with values 0 or 1 (uint8).
      A 1 means "this pixel belongs to digit i", 0 means background.

    Method (kept very simple and slide-aligned)
      1) Convert to grayscale and normalize to [0,1].
      2) Apply a single global threshold (t = 0.1) because digits are bright.
         Pixels > t become foreground (1), others become background (0).
      3) Find 8-connected components among the foreground pixels using a small
         depth-first search (DFS). No external libraries used.
      4) Remove tiny components (< 5 pixels) to drop noise.
      5) For stable output order, sort components left→right, then top→bottom.
    """

    # grayscale
    g = im
    if g.ndim == 3:
        g = 0.2126*g[..., 0] + 0.7152*g[..., 1] + 0.0722*g[..., 2]
    g = g.astype(float)

    # normalize to [0,1] so '0.1 threshold' means 10% of dynamic range
    gmin, gmax = g.min(), g.max()
    g = (g - gmin) / (gmax - gmin + 1e-12)

    # threshold at 0.1 of range
    t = 0.1
    bin_im = np.where(g > t, 1, 0).astype(np.uint8)

    H, W = bin_im.shape
    lab = np.zeros((H, W), dtype=np.int32)

    # 8-connected components
    label_id = 0
    masks = []
    cents = []

    # scan every pixel, when we find an unlabeled foreground pixel, we grow a component
    for r in range(H):
        for c in range(W):
            if bin_im[r, c] and lab[r, c] == 0:
                label_id += 1
                stack = [(r, c)]        # pixels to visit
                lab[r, c] = label_id
                pixels = [(r, c)]       # pixels belonging to this component
                
                # DFS over the 8-neighborhood
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < H and 0 <= xx < W and bin_im[yy, xx] and lab[yy, xx] == 0:
                                lab[yy, xx] = label_id
                                stack.append((yy, xx))
                                pixels.append((yy, xx))
                # drop tiny specks (<5 px)
                if len(pixels) < 5:
                    for (yy, xx) in pixels:
                        lab[yy, xx] = 0
                else:
                    mk = np.zeros((H, W), dtype=np.uint8)  # uint8 mask
                    ys, xs = zip(*pixels)
                    mk[ys, xs] = 1                          # 1s for this digit
                    cx = float(np.mean(xs)); cy = float(np.mean(ys))
                    masks.append(mk)
                    cents.append((cx, cy))

    # stable order: left→right, then top→bottom (helps the given benchmark)
    order = sorted(range(len(masks)), key=lambda i: (cents[i][0], cents[i][1]))
    return [masks[i] for i in order]



# ============================================================
# Features (53-D)
# ============================================================

def segment2feature(Si, target_size=48):
    """
    Extract a 53-D, position/scale-invariant feature vector from a binary digit mask.
    Returns a COLUMN vector (53,1) as expected by the benchmark.
    """
    N = len(FEATURE_NAMES)
    if Si is None or Si.size == 0:
        return np.zeros((N, 1), dtype=np.float32)

    # NEW: Get aspect ratio from original segment before resizing for better invariance.
    h_orig, w_orig = Si.shape
    aspect_ratio = float(h_orig) / (w_orig + 1e-8)
    aspect_ratio_norm = np.clip(aspect_ratio, 0, 5.0) / 5.0  # Clip tall '1's and normalize

    # Preprocess: center-of-mass to center, fixed canvas
    centered, _dbg = _preprocess_digit((Si > 0).astype(np.uint8), target=target_size)
    H, W = centered.shape

    # Use regionprops on the final, centered image for most features
    lbl = label(centered)
    props = regionprops(lbl)
    if not props:
        return np.zeros((N, 1), dtype=np.float32)
    p = max(props, key=lambda r: r.area)

    # Basic shape features
    area = float(p.area)
    ecc = float(p.eccentricity or 0.0)
    extent = float(p.extent or 0.0)
    solidity = float(p.solidity or 0.0)
    norm_area = np.clip(area / float(H * W), 0.0, 1.0)

    # NEW: Euler number (holes), normalized to [0, 1].
    # '8' -> -1 -> 0.0; '1'/'2' -> 0 -> 0.5; '0'/'6' -> 1 -> 1.0
    euler_norm = np.clip((float(p.euler_number) + 1.0) / 2.0, 0.0, 1.0)

    # Hu moments (first two), compressed to [0,1]
    hu = getattr(p, "moments_hu", np.zeros(7))
    hu0c = float(_hu_compress(hu[0])) if len(hu) > 0 else 0.0
    hu1c = float(_hu_compress(hu[1])) if len(hu) > 1 else 0.0

    # UPGRADED: 3x3 zone densities for more spatial detail
    z = []
    for i in range(3):
        for j in range(3):
            r0, r1 = i * H // 3, (i + 1) * H // 3
            c0, c1 = j * W // 3, (j + 1) * W // 3
            block = centered[r0:r1, c0:c1]
            z.append(float(block.mean()) if block.size > 0 else 0.0)
    z = np.array(z, dtype=np.float32)

    # UPGRADED: Full 36-D HOG vector for rich gradient information
    # 2x2 cells over the 48x48 image, 9 orientations -> 4 * 9 = 36 features.
    hog_feats = hog(
        centered.astype(float),
        pixels_per_cell=(H // 2, W // 2),  # 24x24 cells
        cells_per_block=(1, 1),
        orientations=9,
        visualize=False,
        feature_vector=True,
    )
    n = np.linalg.norm(hog_feats) + 1e-8 # L2 normalize for stability
    hog_feats_norm = hog_feats / n

    # Assemble the final, improved feature vector
    feats = np.array([
        norm_area, aspect_ratio_norm, ecc, extent, solidity, euler_norm,
        hu0c, hu1c,
        *z.tolist(),
        *hog_feats_norm.tolist()
    ], dtype=np.float32)

    return feats.reshape(-1, 1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def class_train(X, Y, method="rf"):
    """
    Train a classifier on features.
    """
    if method == "knn":
        model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    elif method == "svm":
        model = SVC(kernel="rbf", C=10, gamma="scale")
    elif method == "rf":
        model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, bootstrap=False)
    elif method == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(40, 20), solver="adam", random_state=42, batch_size=20)
    else:
        raise ValueError(f"Unknown method {method}")
    
    model.fit(X, Y)
    return model


def feature2class(x, classification_data):
    """
    Classify a single feature vector using the trained classifier.
    x: (n_features, 1)
    """
    return int(classification_data.predict(x.reshape(1, -1))[0])


# if __name__ == "__main__": 
#     # Choose dataset
#     thisdir = os.path.dirname(os.path.realpath(__file__))
#     datadir = os.path.join(thisdir, 'datasets', 'short1')  # Which folder of examples are you going to test it on?
#     datadir = os.path.join(thisdir, 'datasets', 'home1')  # datadir = os.path.join('datasets','home1')  # Which folder of examples are you going to test it on?
    
#     # Benchmark and visualize
    
#     mode = 0 # debug modes 
#     # 0 with no plots
#     # 1 with some plots
#     # 2 with the most plots || We recommend setting mode = 2 if you get bad
#     # results, where you can now step-by-step check what goes wrong. You will
#     # get a plot showing some letters, and it will step-by-step show you how
#     # the segmentation worked, and what your classifier classified the letter
#     # as. Press any button to go to the next letter, and so on.
       
#     hitrate,confmat,allres,alljs,alljfg,allX,allY = benchmark_assignment3(im2segment, segment2feature, feature2class,0,datadir,mode)
#     print('Hitrate = ' + str(hitrate*100) + '%')

if __name__ == "__main__": 
    thisdir = os.path.dirname(os.path.realpath(__file__))
    datasets = ["short1", "home1"]

    mode = 0  # debug modes: 0=no plots, 1=some plots, 2=step-by-step plots

    for dataset in datasets:
        datadir = os.path.join(thisdir, "datasets", dataset)
        print(f"\n=== Running OCR system on dataset: {dataset} ===")

        # Step 1: Collect features + labels (dummy classifier during collection)
        _, _, _, _, _, allX, allY = benchmark_assignment3(
            im2segment, segment2feature, lambda x, c: 0, None, datadir, mode=0
        )

        # Step 2: Prepare training data
        X_train = allX.T  # shape: (n_samples, n_features)
        Y_train = np.array([int(y) for y in allY], dtype=np.int32)

        # Step 3: Train classifier
        classification_data = class_train(X_train, Y_train)

        # Step 4: Evaluate with trained classifier
        hitrate, confmat, allres, jis, jalls, _, _ = benchmark_assignment3(
            im2segment, segment2feature, feature2class, classification_data, datadir, mode
        )

        print(f"Hitrate = {hitrate*100:.2f}%")
        print("Confusion matrix:\n", confmat)

