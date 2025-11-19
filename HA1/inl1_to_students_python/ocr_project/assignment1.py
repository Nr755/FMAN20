#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from benchmarking.benchmark_assignment1 import benchmark_assignment1


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


if __name__ == "__main__":

    # read example image
    im = plt.imread('datasets/short1/im1.jpg') 
    
    # read ground truth numbers
    gt_file = open('datasets/short1/im1.txt','r') 
    gt = gt_file.read()
    gt = gt[:-1] # remove newline character
    gt_file.close()
    
    # show image with ground truth
    plt.imshow(im)
    plt.title(gt)
    
    # segment example image
    S = im2segment(im)
    
    # Plot all the segments
    fig,axs = plt.subplots(len(S),1) # create n x 1 subplots 
    
    for Si,axi in zip(S,axs):  # loop over segments and subplots
        axi.imshow(Si,cmap = 'gray',vmin = 0, vmax = 1.0)
    
    
    
    # Benchmark your segmentation routine on all images
    
    datadir = os.path.join('datasets','short1')
    debug = False
    stats = benchmark_assignment1(im2segment,datadir,debug)
    if stats != 0:
        print(f'Total mean Jaccard score is {np.mean(stats[0]):.2}')
        
