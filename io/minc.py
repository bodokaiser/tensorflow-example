import h5py
import numpy as np

SLICE_WIDTH = 466
SLICE_HEIGHT = 394
SLICE_SHAPE = [SLICE_WIDTH, SLICE_HEIGHT, 1]

def read_minc(filename):
    with h5py.File(filename, 'r') as f:
        slices = np.array(f['minc-2.0/image/0/image'], np.float32)
        slices -= np.min(slices)
        slices /= np.max(slices)
        return slices