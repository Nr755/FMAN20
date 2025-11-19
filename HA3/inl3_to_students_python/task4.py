#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def poly2(x, coeffs):
    """Evaluate quadratic polynomial with coefficients [a, b, c]."""
    return coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]

def least_squares_fit(xm, ym):
    """Fit quadratic using numpy polyfit (library LS)."""
    coeffs = np.polyfit(xm, ym, 2)
    return coeffs

def least_squares_fit_own(xm, ym):
    """
    Build Vandermonde X = [x^2, x, 1] and solve
    min ||X beta - y||_2 using a linear solver (lstsq).
    """
    xm = np.asarray(xm); ym = np.asarray(ym)
    X = np.c_[xm**2, xm, np.ones_like(xm)]
    beta, *_ = np.linalg.lstsq(X, ym, rcond=None)
    return beta  # [a, b, c]

def ransac_fit(xm, ym, n_iter=1000, threshold=1.0):
    """RANSAC for quadratic polynomial."""
    n_points = len(xm)
    best_inliers = []
    best_coeffs = None

    for _ in range(n_iter):
        # Sample 3 points (minimal for quadratic)
        ids = np.random.choice(n_points, 3, replace=False)
        coeffs = np.polyfit(xm[ids], ym[ids], 2)

        # Compute errors
        y_pred = poly2(xm, coeffs)
        residuals = np.abs(ym - y_pred)

        # Identify inliers
        inliers = np.where(residuals < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_coeffs = coeffs

    # Refit polynomial using all inliers
    final_coeffs = np.polyfit(xm[best_inliers], ym[best_inliers], 2)
    return final_coeffs, best_inliers

if __name__ == "__main__":
    # load data
    datadir = 'HA3/inl3_to_students_python/'
    data = scipy.io.loadmat(datadir + 'curvedata.mat')
    
    xm = data['xm'].flatten()
    ym = data['ym'].flatten()
    
    # LS (numpy.polyfit)
    coeffs_ls_np = least_squares_fit(xm, ym)
    y_ls_np = poly2(xm, coeffs_ls_np)

    # LS (lstsq)
    coeffs_ls_own = least_squares_fit_own(xm, ym)
    y_ls_own = poly2(xm, coeffs_ls_own)

    # RANSAC fit
    coeffs_ransac, inliers = ransac_fit(xm, ym, n_iter=1000, threshold=1.0)
    y_ransac = poly2(xm, coeffs_ransac)

    # Errors
    err_ls_np = np.mean((ym - y_ls_np)**2)
    err_ls_own = np.mean((ym - y_ls_own)**2)
    err_ransac_all = np.mean((ym - y_ransac)**2)
    err_ransac_inliers = np.mean((ym[inliers] - poly2(xm[inliers], coeffs_ransac))**2)

    # Sanity check: both LS methods should match closely
    print("Close(polyfit vs own)?", np.allclose(coeffs_ls_np, coeffs_ls_own, atol=1e-8))

    print("LS (numpy.polyfit) coefficients:", coeffs_ls_np)
    print("LS (own lstsq)   coefficients:", coeffs_ls_own)
    print("RANSAC           coefficients:", coeffs_ransac)
    print("Number of inliers (RANSAC):", len(inliers))
    print("LS-numpy  MSE (all points):", err_ls_np)
    print("LS-own    MSE (all points):", err_ls_own)
    print("RANSAC    MSE (all points):", err_ransac_all)
    print("RANSAC    MSE (inliers only):", err_ransac_inliers)

    # Plot: show all three curves
    xs = np.linspace(min(xm), max(xm), 200)
    plt.scatter(xm, ym, label="Data", c="gray")
    plt.plot(xs, poly2(xs, coeffs_ls_np), 'r-',  label="LS (numpy.polyfit)")
    plt.plot(xs, poly2(xs, coeffs_ls_own), 'g--', label="LS (own lstsq)")
    plt.plot(xs, poly2(xs, coeffs_ransac), 'b-',  label="RANSAC")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Quadratic Curve Fit: LS (numpy) vs LS (own) vs RANSAC")
    plt.show()
