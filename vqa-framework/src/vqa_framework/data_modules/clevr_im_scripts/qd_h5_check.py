"""
Quick ugly sanity check.
"""

import numpy as np

import h5py
import matplotlib.pyplot as plt

if __name__ == "__main__":
    feature_h5 = h5py.File("tmp_test.h5", "r")
    features = feature_h5["features"][()]
    plt.imshow(features[0].transpose(1, 2, 0).astype(np.int32))
    plt.show()
    plt.imshow(features[100].transpose(1, 2, 0).astype(np.int32))
    plt.show()
    plt.imshow(features[200].transpose(1, 2, 0).astype(np.int32))
    plt.show()
