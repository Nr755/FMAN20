#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 2 – Image analysis (digits) - IMPROVED VERSION
- Segmentation into connected components
- Position/scale-invariant 53-D features for better separation
- Clear, named prints + optional debug figures + report outputs

Feature vector (53):
[ norm_area, aspect_ratio, eccentricity, extent, solidity, euler_norm,
  hu0c, hu1c,               # log+tanh-compressed Hu 1–2
  z00..z22,                 # 3x3 zone densities
  hog0..hog35 ]             # full 2x2 cell, 9-orientation HOG

All features are computed after COM-centering and resizing
the digit to a fixed canvas (48x48).
"""

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import regionprops, label
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from PIL import Image

from benchmarking.benchmark_assignment2 import benchmark_assignment2


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

def im2segment(im, min_area=10):
    """
    Segment a (possibly grayscale/RGB) image into connected components.
    Returns a list of binary masks, each cropped to its bbox, sorted left→right.
    """
    if im.ndim == 3:
        im = rgb2gray(im)         # [0,1]
    else:
        im = _to_float01(im)

    im_bin = _binarize(im)

    lbl = label(im_bin)
    segs, bboxes = [], []

    for p in regionprops(lbl):
        if p.area < min_area:
            continue
        minr, minc, maxr, maxc = p.bbox
        sub = (lbl[minr:maxr, minc:maxc] == p.label).astype(np.uint8)
        segs.append(sub)
        bboxes.append((minc, minr, maxc, maxr))

    if not segs:
        return [im_bin.astype(np.uint8)]

    # sort by x (left→right)
    order = np.argsort([b[0] for b in bboxes])
    return [segs[i] for i in order]


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


# ============================================================
# Nice printing / quick viz
# ============================================================

def print_named_vector(vec):
    flat = vec.ravel().astype(float)
    if len(flat) != len(FEATURE_NAMES):
        print(f"Error: Vector length ({len(flat)}) does not match feature names ({len(FEATURE_NAMES)}).")
        print(vec)
        return
    pairs = [f"{n}: {v:.6f}" for n, v in zip(FEATURE_NAMES, flat)]
    print(f"{len(flat)}-D feature vector:")
    print("  " + "\n  ".join(pairs))

def show_segments_overlay(im, segments, max_show=12):
    """Optional helper to see segmentation order."""
    if im.ndim == 3:
        im_gray = rgb2gray(im)
    else:
        im_gray = _to_float01(im)
    plt.figure()
    plt.imshow(im_gray, cmap="gray")
    plt.title("Segments (left→right index)")
    ax = plt.gca()
    for idx, _ in enumerate(segments[:max_show]):
        ax.text(5 + 14*idx, 5, str(idx), color="lime", fontsize=10, bbox=dict(fc='k', alpha=0.3))
    plt.axis("off")
    plt.show()


# ============================================================
# Report helpers (CSV + thumbnails + readable prints)
# ============================================================

def _iter_images(datadir):
    # Sorted list of JPGs in the dataset folder
    return sorted(glob.glob(os.path.join(datadir, "*.jpg")))

def dump_vectors_csv(datadir, outdir, segment_func=segment2feature):
    """Create out/features.csv with: file, segment, <N named features>."""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "features_53d.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "segment"] + FEATURE_NAMES)
        for p in _iter_images(datadir):
            im = plt.imread(p)
            segs = im2segment(im)
            base = os.path.basename(p)
            for i, Si in enumerate(segs):
                vec = segment_func(Si).ravel().astype(float)
                writer.writerow([base, i] + [f"{v:.8f}" for v in vec])
    return csv_path

def save_centered_thumbs(datadir, outdir, target_size=48):
    """Save centered 48x48 binary thumbnails per segment."""
    thumbdir = os.path.join(outdir, "thumbs")
    os.makedirs(thumbdir, exist_ok=True)
    for p in _iter_images(datadir):
        im = plt.imread(p)
        segs = im2segment(im)
        base = os.path.splitext(os.path.basename(p))[0]
        for i, Si in enumerate(segs):
            centered, _ = _preprocess_digit((Si > 0).astype(np.uint8), target=target_size)
            Image.fromarray((centered * 255).astype(np.uint8)).save(
                os.path.join(thumbdir, f"{base}_seg{i}.png")
            )
    return thumbdir

def pretty_print_vectors(datadir, segment_func=segment2feature):
    """Readable console dump you can paste into the report."""
    for p in _iter_images(datadir):
        im = plt.imread(p)
        segs = im2segment(im)
        print(f"\n== {os.path.basename(p)} ==")
        for i, Si in enumerate(segs):
            v = segment_func(Si).ravel().astype(float)
            print(f"  segment {i}:")
            for name, val in zip(FEATURE_NAMES, v):
                print(f"    {name:>12s}: {val:.6f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    thisdir = os.path.dirname(os.path.realpath(__file__))

    ap = argparse.ArgumentParser(description="OCR Assignment 2 – segmentation & 53-D features")
    ap.add_argument("--datadir", default=os.path.join(thisdir, "datasets", "short1"),
                    help="Dataset folder (default: datasets/short1)")
    ap.add_argument("--outdir",  default=os.path.join(thisdir, "datasets", "short1", "out"),
                    help="Output folder for report artifacts")
    ap.add_argument("--no-benchmark", action="store_true", help="Skip benchmark run")
    ap.add_argument("--report", action="store_true",
                    help="Generate CSV + thumbnails + readable prints for the report")
    ap.add_argument("--viz", action="store_true", help="Show a quick segmentation overlay for the first image")
    args = ap.parse_args()

    datadir = args.datadir
    outdir  = args.outdir

    # Example: read first image (if present)
    jpgs = _iter_images(datadir)
    if jpgs:
        im = plt.imread(jpgs[0])
        S = im2segment(im)
        # Compute a sample vector for segment #2 (or last if fewer)
        if S:
            Si = S[min(2, len(S) - 1)]
            f = segment2feature(Si)
            print_named_vector(f)
            if args.viz:
                show_segments_overlay(im, S)
        else:
            print(f"No segments found in {jpgs[0]}")


    # Benchmark on all images
    if not args.no_benchmark:
        debug = True  # set False for quieter output
        _ = benchmark_assignment2(segment2feature, datadir, debug)

    # Report artifacts
    if args.report:
        csv_path = dump_vectors_csv(datadir, outdir, segment_func=segment2feature)
        thumbs_dir = save_centered_thumbs(datadir, outdir, target_size=48)
        print("\nReport artifacts:")
        print("  CSV:   ", csv_path)
        print("  Thumbs:", thumbs_dir)
        print("\nReadable feature prints:")
        pretty_print_vectors(datadir, segment_func=segment2feature)