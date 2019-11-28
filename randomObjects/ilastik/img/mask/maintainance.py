import numpy as np
import h5py

f = h5py.File('masks.h5', 'a')
for key in list(f.keys()):
    dset = f[key][:]/255
    f[key][:] = dset.round()
f.close()
