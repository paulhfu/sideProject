import numpy as np
import h5py

files = ['Superpixels1.h5', 'Superpixels2.h5', 'Superpixels3.h5', 'Superpixels4.h5', 'Superpixels5.h5', 'Superpixels6.h5', 'Superpixels7.h5', 'Superpixels8.h5', 'Superpixels9.h5', 'Superpixels10.h5', 'Superpixels11.h5', 'Superpixels12.h5', 'Superpixels14.h5', 'Superpixels15.h5', 'Superpixels16.h5', 'Superpixels18.h5', 'Superpixels19.h5', 'Superpixels20.h5']
f = h5py.File('superpixels.h5','a')
for i, file in enumerate(files):
    dset = h5py.File(file, 'r')['exported_data'][:]
    f.create_dataset(str(i), data=dset)
f.close()
