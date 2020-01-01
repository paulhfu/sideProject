import sys
sys.path.append('/usr/local/lib/python3.6/site-packages/')
import affogato
import numpy as np


from affogato.segmentation import compute_mws_segmentation

# seg = compute_mws_segmentation(np.random.rand(2,100,100), [[0,-1],[0,-1]], 1, algorithm='prim')
# a=1

def ind_flat_2_spat(flat_indices, shape):
    spat_indices = np.zeros([len(flat_indices)] + [len(shape)])
    for flat_ind, spat_ind in zip(flat_indices, spat_indices):
        rm = flat_ind
        for dim in range(1, len(shape)):
            sz = np.prod(shape[dim:])
            spat_ind[dim - 1] = rm // sz
            rm -= spat_ind[dim - 1] * sz
        spat_ind[-1] = rm
    return spat_indices

def ind_spat_2_flat(spat_indices, shape):
    flat_indices = np.zeros(len(spat_indices))
    for i, spat_ind in enumerate(spat_indices):
        for dim in range(len(shape)):
            flat_indices[i] += max(1, np.prod(shape[dim + 1:])) * spat_ind[dim]
    return flat_indices

shape = [5,5,5,5,5]

spat = [[3,4,2,4,4]]
flat = [5*5*5*5*3 + 5*5*5*4 + 5*5*2 + 5*4 + 4]

calc_spat = ind_flat_2_spat(flat, shape)
calc_flat = ind_spat_2_flat(spat, shape)

a=1


