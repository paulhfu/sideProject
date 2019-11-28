import imageio
import numpy as np
import h5py

files = ['mask1.jpg', 'mask2.jpg', 'mask3.jpg', 'mask4.jpg', 'mask5.jpg', 'mask6.jpg', 'mask7.jpg', 'mask8.jpg', 'mask9.jpg', 'mask10.jpg', 'mask11.jpg', 'mask12.jpg', 'mask14.jpg', 'mask15.jpg', 'mask16.jpg', 'mask18.jpg', 'mask19.jpg', 'mask20.jpg']
f = h5py.File('masks.h5','a')
for i, file in enumerate(files):
    dset = imageio.imread(file)
    f.create_dataset(str(i), data=dset)
f.close()
