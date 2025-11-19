import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# ============================================================
# Minimal segment2feature (7-D feature vector)
# ============================================================

def segment2feature(Si, target_size=28):
    if Si is None or Si.size == 0:
        return np.zeros(7, dtype=np.float32)

    # ensure binary
    Si = (Si > 0).astype(np.uint8)

    lbl = label(Si)
    props = regionprops(lbl)
    if not props:
        return np.zeros(7, dtype=np.float32)
    p = max(props, key=lambda r: r.area)

    h, w = Si.shape
    area = float(p.area) / (h * w)
    aspect_ratio = float(h) / (w + 1e-8)
    ecc = float(p.eccentricity or 0.0)
    extent = float(p.extent or 0.0)
    solidity = float(p.solidity or 0.0)

    hu = getattr(p, "moments_hu", np.zeros(7))
    def _hu_compress(val):
        s = np.sign(val)
        v = np.log1p(np.abs(val))
        z = np.tanh(3.0 * s * v)
        return 0.5 * (z + 1.0)

    hu0c = _hu_compress(hu[0]) if len(hu) > 0 else 0.0
    hu1c = _hu_compress(hu[1]) if len(hu) > 1 else 0.0

    return np.array([area, aspect_ratio, ecc, extent, solidity, hu0c, hu1c],
                    dtype=np.float32)


# ============================================================
# Ground-truth test
# ============================================================

def test_with_ground_truth(datadir):
    gtnames = glob.glob(os.path.join(datadir, "*txt"))
    segnames = glob.glob(os.path.join(datadir, "*npy"))
    gtnames.sort()
    segnames.sort()

    allX, allY = [], []

    for segname, gtname in zip(segnames, gtnames):
        Sgt = np.load(segname)
        with open(gtname, "r") as f:
            gt = f.read().strip()
        if len(gt) != len(Sgt):
            continue
        for mask, label in zip(Sgt, gt):
            fvec = segment2feature(mask)
            allX.append(fvec)
            allY.append(int(label))

    allX = np.array(allX, dtype=np.float32)
    allY = np.array(allY, dtype=np.int32)

    # 80/20 split
    X_train, X_test, Y_train, Y_test = train_test_split(
        allX, allY, test_size=0.2, stratify=allY
    )

    # kNN classifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred, labels=range(10))

    print(f"Ground-truth minimal features accuracy: {acc*100:.2f}%")
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    thisdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(thisdir, "datasets", "short1")  # or "home1"
    test_with_ground_truth(datadir)
