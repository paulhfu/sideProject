import torch
import numpy as np

class StridedRollingRandomPatches2D():

    def __init__(self, strides, patch_shape, shape):
        assert len(strides) == 2
        assert len(patch_shape) == 2
        assert len(shape) == 2

        self.strides = np.array(strides)
        self.shape = np.array(shape)
        self.patch_shape = np.array(patch_shape)
        self.n_patch_per_dim = self.shape // self.strides

    def get_patch(self, image, index):
        idx1 = index // self.n_patch_per_dim[1]
        idx2 = index % self.n_patch_per_dim[1]

        idx1 *= self.strides[0]
        idx2 *= self.strides[1]

        rolled_img = image.roll([idx1, idx2], [-2, -1])
        patch = rolled_img[..., :self.patch_shape[0], :self.patch_shape[1]]
        return patch


class NoPatches2D():

    def __init__(self):
        self.n_patch_per_dim = [1, 1]

    def get_patch(self, image, index):
        return image
