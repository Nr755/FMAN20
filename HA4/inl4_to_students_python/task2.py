#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:06:01 2024

@author: magnuso
"""

import numpy as np
import scipy
from scipy.sparse.csgraph import maximum_flow
from scipy import sparse
import matplotlib.pyplot as plt

def edges4connected(height,width,only_one_dir = 0):
    #
    # Generates a 4-connectivity structure for a height*width grid
    #
    # if only_one_dir==1, then there will only be one edge between node i and
    # node j. Otherwise, both i-->j and i<--j will be added.
    #
    
    N = height*width
    I = np.array([])
    J = np.array([])
    
    # connect vertically (down, then up)
    iis = np.delete(np.arange(N),np.arange(height-1,N,height))
    jjs = iis+1;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    # connect horizontally (right, then left)
    
    iis = np.arange(0,N-height)
    jjs = iis+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    return I,J

def edges8connected(height,width,only_one_dir = 0):
    #
    # Generates a 8-connectivity structure for a height*width grid
    #
    # if only_one_dir==1, then there will only be one edge between node i and
    # node j. Otherwise, both i-->j and i<--j will be added.
    #
    
    N = height*width
    I = np.array([])
    J = np.array([])
    
    # connect vertically (down, then up)
    iis = np.delete(np.arange(N),np.arange(height-1,N,height))
    jjs = iis+1;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    # Diagonal
    jjs = iis+1+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))

    jjs = iis+1-height;
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))


    ind = (I>=0) & (I<N)
    I = I[ind]
    J = J[ind]
    ind = (J>=0) & (J<N)
    I = I[ind]
    J = J[ind]
    

    # connect horizontally (right, then left)
    
    iis = np.arange(0,N-height)
    jjs = iis+height
    if ~only_one_dir:
        I = np.hstack((I,iis,jjs))
        J = np.hstack((J,jjs,iis))
    else:
        I = np.hstack((I,iis))
        J = np.hstack((J,jjs))
    
    return I,J


if __name__ == '__main__':

    # --- Example toy flow (can keep for reference, or comment out) ---
    # I = np.array([0,0,1,2], dtype=np.int32)
    # J = np.array([1,2,3,3], dtype=np.int32)
    # V = np.array([5,2,3,7], dtype=np.int32)
    # F = sparse.coo_array((V,(I,J)),shape=(4,4)).tocsr()
    # mf = maximum_flow(F, 0, 3)
    # print("Toy flow value:", mf.flow_value)
    # print(mf.flow.toarray())

    # --- Real problem: Heart segmentation ---
    data = scipy.io.loadmat('heart_data.mat')
    data_chamber = data['chamber_values']
    data_background = data['background_values']
    im = data['im']
    M, N = im.shape
    n = M * N

    # === 1. Gaussian parameters ===
    m_chamber     = float(np.mean(data_chamber))
    s_chamber     = float(np.std(data_chamber, ddof=1))
    m_background  = float(np.mean(data_background))
    s_background  = float(np.std(data_background, ddof=1))

    print(f"Chamber mean={m_chamber:.2f}, std={s_chamber:.2f}")
    print(f"Background mean={m_background:.2f}, std={s_background:.2f}")

    # === 2. Edge structure and regularization ===
    Ie, Je = edges4connected(M, N)   # or edges8connected
    lam = 5.0                        # smoothness weight, tune as needed
    Ve = lam * np.ones_like(Ie, dtype=float)

    # === 3. Data terms ===
    f = im.astype(float).ravel()
    eps = 1e-8
    Vs = ((f - m_chamber)**2) / (2.0 * (s_chamber**2 + eps))      # chamber cost
    Vt = ((f - m_background)**2) / (2.0 * (s_background**2 + eps))  # background cost

    # === 4. Source/sink connections ===
    Is1 = np.arange(n)
    Js1 = (n) * np.ones((n,), dtype=np.int32)
    Is2 = (n) * np.ones((n,), dtype=np.int32)
    Js2 = np.arange(n)

    It1 = np.arange(n)
    Jt1 = (n+1) * np.ones((n,), dtype=np.int32)
    It2 = (n+1) * np.ones((n,), dtype=np.int32)
    Jt2 = np.arange(n)

    # === 5. Build graph ===
    I = np.hstack((Ie, Is1, Is2, It1, It2)).astype(np.int32)
    J = np.hstack((Je, Js1, Js2, Jt1, Jt2)).astype(np.int32)
    V = np.hstack((Ve, Vs, Vs, Vt, Vt))

    sf = 10000   # scale to integers
    V = np.round(V * sf).astype(np.int32)

    F = sparse.coo_array((V,(I,J)), shape=(n+2,n+2)).tocsr()

    # === 6. Solve max flow/min cut ===
    mf = maximum_flow(F, n, n+1)

    seg = mf.flow
    imflow = seg[0:n, n+1].reshape((M,N)).toarray().astype(float)

    # Threshold against sink costs to decide segmentation
    imseg = imflow < (V[-n:].astype(float).reshape(M,N))

    plt.imshow(imseg, cmap='gray')
    plt.title("Graph cut segmentation")
    plt.show()
