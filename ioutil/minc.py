import h5py
import numpy as np

def read_minc(filename):
    with h5py.File(filename, 'r') as f:
        slices = np.array(f['minc-2.0/image/0/image'], np.float32)
        slices -= np.min(slices)
        slices /= np.max(slices)
        return slices