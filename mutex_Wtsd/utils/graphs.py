import torch
import numpy as np

def collate_edges(edges):
    n_offs = [0]
    e_offs = [0]
    for i in range(len(edges)):
        n_offs.append(n_offs[-1] + edges[i].max() + 1)
        e_offs.append(e_offs[-1] + edges[i].shape[1])
        edges[i] += n_offs[-2]

    return torch.cat(edges, 1), (n_offs, e_offs)

def separate_nodes(nodes, n_offs):
    r_nodes = []
    for i in range(len(n_offs) - 1):
        r_nodes.append(nodes[n_offs[i]: n_offs[i+1]] - n_offs[i])
    return r_nodes

def separate_edges(edges, e_offs, n_offs):
    r_edges = []
    for i in range(len(e_offs) - 1):
        r_edges.append(edges[e_offs[:, i]: e_offs[:, i+1]] - n_offs[i])
    return r_edges

def get_edge_indices(edges, edge_list):
    """
    :param edges: list of edges in a graph shape(n, 2). must be sorted like (smaller, larger) and must not contain equal values
    :param edge_list: list of edges. Each edge must be present in edges. Can contain multiple entries of one edge
    :return: the indices for edges that each entry in edge list correpsonds to
    """
    indices = torch.nonzero(((edges[0] == edge_list[0].unsqueeze(-1)) & (edges[1] == edge_list[1].unsqueeze(-1))),
                  as_tuple=True)[1]
    assert indices.shape[0] == edge_list.shape[1], "edges must be sorted and unique"
    return indices


def get_angles_in_rag(edges, segmentation):
    """
    calculates the angles in a region adjacency graph based on the center of mass of each sp.
    Batches are not supported.
    :param edges: torch.Tensor storing the undirectional edges in the rag
    :param segmentation: torch.Tensor storing a single segmentation images
    :return: for each undirectional edge two angles of the line going through the center of masses of the incidental nodes
    and the x-axis w.r.t. each node. W.r.t. the first and the second node is in first and second half of result respectively
    """
    nodes = torch.unique(segmentation)
    cms = torch.empty((len(nodes), 2), dtype=torch.float, device=edges.device)
    for i, (n, mask) in enumerate(zip(nodes, segmentation.unsqueeze(0) == nodes.unsqueeze(-1).unsqueeze(-1))):
        idxs = torch.where(mask)
        cms[i] = torch.stack([torch.sum(idxs[0]), torch.sum(idxs[1])]) / (mask.sum().float() + np.finfo(float).eps)

    vec = cms[edges[0]] - cms[edges[1]]
    angles = torch.atan(vec[:, 0] / (vec[:, 1] + np.finfo(float).eps))
    angles = 2 * angles / np.pi
    return angles


if __name__ == "__main__":
    # edges = np.array([[1, 3], [2, 4], [1, 2], [2, 3], [3, 5], [3, 6], [1, 5], [2, 8], [4, 8], [4, 9], [5, 9], [8, 9]])
    # edge_list = edges[np.random.choice(np.arange(edges.shape[0]), size=(20 * 10))]
    # edge_indices = get_edge_indices(torch.from_numpy(edges).transpose(0, 1), torch.from_numpy(edge_list).transpose(0, 1))

    sp = torch.zeros((100, 100), dtype=torch.float)
    sp[:50, :50] = 0.0
    sp[50:, :50] = 1.0
    sp[:50, 50:] = 2.0
    sp[50:, 50:] = 3.0
    sp[40:60, 40:60] = 4.0
    edges = torch.tensor([[0, 4], [1, 4], [2, 4], [3, 4]], dtype=torch.long).T

    get_angles_in_rag(edges, sp)

