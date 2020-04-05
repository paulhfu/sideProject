import h5py
from affogato.affinities import compute_affinities
import numpy as np

def computeAffs(file_from, offsets):
    file = h5py.File(file_from, 'a')
    keys = list(file.keys())
    file.create_group('masks')
    file.create_group('affs')
    for k in keys:
        data = file[k][:].copy()
        affinities, _ = compute_affinities(data != 0, offsets)
        file['affs'].create_dataset(k, data=affinities)
        file['masks'].create_dataset(k, data=data)
        del file[k]
    return

def get_naive_affinities(raw, offsets):
    affinities = np.zeros([len(offsets)] + list(raw.shape))
    normed_raw = raw / raw.max()
    for y in range(normed_raw.shape[0]):
        for x in range(normed_raw.shape[1]):
            for i, off in enumerate(offsets):
                if 0 <= y+off[0] < normed_raw.shape[0] and 0 <= x+off[1] < normed_raw.shape[1]:
                    affinities[i, y, x] = abs(normed_raw[y, x] - normed_raw[y+off[0], x+off[1]])
    return affinities