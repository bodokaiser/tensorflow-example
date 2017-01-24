import h5py
import numpy as np

def read_minc(filename):
    with h5py.File(filename, 'r') as f:
        slices = np.array(f['minc-2.0/image/0/image'], np.float32)
        vrange = np.array(f['minc-2.0/image/0/image'].attrs['valid_range'])
        return (slices - vrange[0]) / np.sum(np.abs(vrange))