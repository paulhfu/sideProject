import napari
import h5py

with napari.gui_qt():
    file1 = 'superpixels.h5'
    viewer = napari.Viewer(title=file1)
    f = h5py.File(file1, 'r')
    keys = list(f.keys())
    for key in keys:
        viewer.add_image(f[key],visible=False)