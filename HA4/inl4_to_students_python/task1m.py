#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMAN20 – Task 1: Color correction / white balance
- Gray-World (baseline)
- Modified Gray-World (illumination-smoothed, linear-RGB)
- Learned method (Afifi et al., WB_sRGB)

Requires:
  pip install numpy matplotlib scikit-image opencv-python scipy
Place flip.py in the same folder (provided by course).
"""

from pathlib import Path
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_filter
import cv2

# ----------------- Paths (robust) -----------------
from pathlib import Path
import os, sys

ROOT = Path(__file__).resolve().parent  # folder containing this script

# Set your WB_sRGB repo root (the folder that contains WB_sRGB_Python/)
WB_REPO = Path("/Users/mikolajsinicka/Desktop/FMAN20/HA4/WB_sRGB-master")
WB_PY   = WB_REPO / "WB_sRGB_Python"
MODELS  = WB_PY / "models"

if not WB_PY.exists():
    raise FileNotFoundError(f"WB_sRGB_Python not found at: {WB_PY}")
if not MODELS.exists():
    raise FileNotFoundError(
        f"Model dir not found: {MODELS}\n"
        "Did you run `git lfs install && git lfs pull` in the WB repo?"
    )

# Make the package importable just like their demo
if str(WB_PY) not in sys.path:
    sys.path.insert(0, str(WB_PY))

# IMPORTANT: make relative 'models/...' loads inside WBsRGB.py work
# (They resolve against the current working directory.)
os.chdir(str(WB_PY))

# Course helper (in same folder as this script)
from flip import computeFLIP

# Learned method class
from classes import WBsRGB as wb_srgb

# ----------------- Small helpers -----------------
def to_uint8(img_float01: np.ndarray) -> np.ndarray:
    return (np.clip(img_float01, 0.0, 1.0) * 255.0).astype(np.uint8)

def to_float01(img_uint8: np.ndarray) -> np.ndarray:
    return img_uint8.astype(np.float32) / 255.0

# sRGB <-> linear (same formulas as in flip.py)
def srgb_to_linear(im: np.ndarray) -> np.ndarray:
    limit = 0.04045
    out = np.empty_like(im, dtype=np.float32)
    mask = (im > limit)
    out[mask]  = ((im[mask] + 0.055) / 1.055) ** 2.4
    out[~mask] = im[~mask] / 12.92
    return out

def linear_to_srgb(im: np.ndarray) -> np.ndarray:
    limit = 0.0031308
    out = np.empty_like(im, dtype=np.float32)
    mask = (im > limit)
    out[mask]  = 1.055 * (im[mask] ** (1.0 / 2.4)) - 0.055
    out[~mask] = 12.92 * im[~mask]
    return out

# ----------------- Methods -----------------
def gray_world(im_srgb01: np.ndarray) -> np.ndarray:
    """Simple gray-world in sRGB: scale each channel by its mean."""
    out = im_srgb01.copy()
    means = out.reshape(-1, 3).mean(axis=0)    # (Rmean, Gmean, Bmean)
    m = float(means.mean())
    gains = np.where(means > 1e-8, m / means, 1.0)  # avoid divide-by-zero
    out *= gains[None, None, :]
    return np.clip(out, 0.0, 1.0)

def gray_world_mod(im_srgb01: np.ndarray, blur_sigma: float = 3.0) -> np.ndarray:
    """
    Modified gray-world:
      • convert to linear RGB,
      • Gaussian-blur to estimate illumination,
      • compute channel gains from blurred image,
      • apply gains to ORIGINAL linear image,
      • convert back to sRGB.
    """
    lin = srgb_to_linear(np.clip(im_srgb01, 0.0, 1.0).astype(np.float32))
    lin_blur = np.empty_like(lin)
    for c in range(3):
        lin_blur[:, :, c] = gaussian_filter(lin[:, :, c], blur_sigma, mode='reflect')
    means = lin_blur.reshape(-1, 3).mean(axis=0)
    m = float(means.mean())
    gains = np.where(means > 1e-12, m / means, 1.0)
    lin_corr = np.clip(lin * gains[None, None, :], 0.0, 1.0)
    return np.clip(linear_to_srgb(lin_corr), 0.0, 1.0)

def wb_afifi(im_srgb01: np.ndarray, upgraded_model: int = 1, gamut_mapping: int = 2) -> np.ndarray:
    """
    Afifi et al. WB_sRGB on RGB float in [0,1].
    Their API takes BGR uint8 and returns BGR float [0,1].
    """
    bgr_u8 = cv2.cvtColor(to_uint8(im_srgb01), cv2.COLOR_RGB2BGR)
    wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping, upgraded=upgraded_model)
    out_bgr = wbModel.correctImage(bgr_u8)               # float [0,1], BGR
    out_rgb = cv2.cvtColor(to_uint8(out_bgr), cv2.COLOR_BGR2RGB)
    return to_float01(out_rgb)

# ----------------- Main -----------------
if __name__ == "__main__":
    # Load images next to this script
    bad_path = ROOT / "abbey_badcolor.jpg"
    ok_path  = ROOT / "abbey_correct.jpg"
    assert bad_path.exists(), f"Not found: {bad_path}"
    assert ok_path.exists(),  f"Not found: {ok_path}"

    im_bad = plt.imread(bad_path).astype(np.float32)
    if im_bad.max() > 1.0:  # some readers return uint8
        im_bad = im_bad / 255.0
    im_ok  = plt.imread(ok_path).astype(np.float32)
    if im_ok.max() > 1.0:
        im_ok = im_ok / 255.0

    # Preview
    plt.figure(figsize=(10, 4))
    plt.imshow(np.hstack((im_bad, im_ok)))
    plt.title("Left: bad color   |   Right: correct reference")
    plt.axis("off")
    plt.show()

    # Methods
    im_gray     = gray_world(im_bad)
    im_gray_mod = gray_world_mod(im_bad, blur_sigma=3.0)

    upgraded_model = 1  # set 0/1 per slides
    gamut_mapping  = 2  # 1=scale, 2=clip (paper default)
    im_afifi      = wb_afifi(im_bad, upgraded_model=upgraded_model, gamut_mapping=gamut_mapping)

    # Visualize outputs
    plt.figure(figsize=(15, 5))
    concat = np.hstack((im_gray, im_gray_mod, im_afifi))
    plt.imshow(np.clip(concat, 0, 1))
    plt.title("Gray-World | Modified Gray-World | Afifi WB_sRGB")
    plt.axis("off")
    plt.show()

    # --- Metrics vs. reference ---
    H, W = im_ok.shape[:2]
    def resize_like(im):
        if im.shape[:2] != (H, W):
            return cv2.resize(to_uint8(im), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return im

    im_gray_m   = resize_like(im_gray)
    im_graym_m  = resize_like(im_gray_mod)
    im_afifi_m  = resize_like(im_afifi)

    psnr_gray     = psnr(im_ok, im_gray_m, data_range=1.0)
    psnr_gray_mod = psnr(im_ok, im_graym_m, data_range=1.0)
    psnr_afifi    = psnr(im_ok, im_afifi_m, data_range=1.0)

    ssim_gray     = ssim(im_ok, im_gray_m, channel_axis=2, data_range=1.0)
    ssim_gray_mod = ssim(im_ok, im_graym_m, channel_axis=2, data_range=1.0)
    ssim_afifi    = ssim(im_ok, im_afifi_m, channel_axis=2, data_range=1.0)

    # flip_gray,     _ = computeFLIP(to_uint8(im_ok), to_uint8(im_gray_m))
    # flip_gray_mod, _ = computeFLIP(to_uint8(im_ok), to_uint8(im_graym_m))
    # flip_afifi,    _ = computeFLIP(to_uint8(im_ok), to_uint8(im_afifi_m))

    flip_gray,     _ = computeFLIP(im_ok,    im_gray_m)
    flip_gray_mod, _ = computeFLIP(im_ok,    im_graym_m)
    flip_afifi,    _ = computeFLIP(im_ok,    im_afifi_m)


    print("\n=== Metrics vs. abbey_correct.jpg ===")
    print(f"Gray-World         : PSNR={psnr_gray:.3f} dB | SSIM={ssim_gray:.5f} | FLIP={flip_gray:.5f}")
    print(f"Modified Gray-World: PSNR={psnr_gray_mod:.3f} dB | SSIM={ssim_gray_mod:.5f} | FLIP={flip_gray_mod:.5f}")
    print(f"WB_sRGB (Afifi)    : PSNR={psnr_afifi:.3f} dB | SSIM={ssim_afifi:.5f} | FLIP={flip_afifi:.5f}")

    # Save outputs for the report
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    plt.imsave(out_dir / "gray_world.png",     np.clip(im_gray, 0, 1))
    plt.imsave(out_dir / "gray_world_mod.png", np.clip(im_gray_mod, 0, 1))
    plt.imsave(out_dir / "wb_srgb.png",        np.clip(im_afifi, 0, 1))
    print(f"Saved outputs to {out_dir}")
