from data.mtx_wtsd import get_sp_graph
import h5py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt



def get_pix_data(length=50000, shape=(128, 128), radius=72):
    mp = (56, 56)
    length = length
    shape = shape
    radius = radius
    fidx = 0
    transform = None

    n_disc = np.random.randint(8, 10)
    rads = []
    mps = []
    for disc in range(n_disc):
        radius = np.random.randint(max(shape) // 18, max(shape) // 15)
        touching = True
        while touching:
            mp = np.array([np.random.randint(0 + radius, shape[0] - radius),
                           np.random.randint(0 + radius, shape[1] - radius)])
            touching = False
            for other_rad, other_mp in zip(rads, mps):
                diff = mp - other_mp
                if (diff ** 2).sum() ** .5 <= radius + other_rad + 2:
                    touching = True
        rads.append(radius)
        mps.append(mp)

    data = np.zeros(shape=shape, dtype=np.float)
    gt = np.zeros(shape=shape, dtype=np.float)
    for y in range(shape[0]):
        for x in range(shape[1]):
            bg = True
            for radius, mp in zip(rads, mps):
                ly, lx = y - mp[0], x - mp[1]
                if (ly ** 2 + lx ** 2) ** .5 <= radius:
                    data[y, x] += np.cos(np.sqrt((x - shape[1]) ** 2 + y ** 2) * 50 * np.pi / shape[1])
                    data[y, x] += np.cos(np.sqrt(x ** 2 + y ** 2) * 50 * np.pi / shape[1])
                    # data[y, x] += 6
                    gt[y, x] = 1
                    bg = False
            if bg:
                data[y, x] += np.cos(y * 40 * np.pi / shape[0])
                data[y, x] += np.cos(np.sqrt(x ** 2 + (shape[0] - y) ** 2) * 30 * np.pi / shape[1])
    data += 1

    return torch.from_numpy(data).float().unsqueeze(0), torch.from_numpy(gt).long().unsqueeze(0)


def store_all(base_dir):
    pix_dir = os.path.join(base_dir, 'pix_data')
    graph_dir = os.path.join(base_dir, 'graph_data')


    for i in range(20):
        raw, gt = get_pix_data()
        raw, gt = raw.squeeze().numpy(), gt.squeeze().numpy()

        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, nodes, noisy_affinities = get_sp_graph(raw, gt, 1.01)

        graph_file = h5py.File(os.path.join(graph_dir, "graph_" + str(i) + ".h5"), 'w')
        pix_file = h5py.File(os.path.join(pix_dir, "pix_" + str(i) + ".h5"), 'w')

        pix_file.create_dataset("raw", data=raw, chunks=True)
        pix_file.create_dataset("gt", data=gt, chunks=True)

        graph_file.create_dataset("edges", data=edges, chunks=True)
        graph_file.create_dataset("edge_feat", data=edge_feat, chunks=True)
        graph_file.create_dataset("diff_to_gt", data=diff_to_gt)
        graph_file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
        graph_file.create_dataset("node_labeling", data=node_labeling, chunks=True)

        graph_file.close()
        pix_file.close()


if __name__ == "__main__":
    store_all("/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/data/storage/pix_and_graphs_validation")
