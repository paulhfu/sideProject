from disjoint_set import DisjointSet
import queue
import numpy as np


# make this inline
def check_mutex(ru, rv, mutexes):
    if ru in mutexes and rv in mutexes:
        mtxs_u = mutexes[ru]
        mtxs_v = mutexes[rv]
        return bool(set(mtxs_u).intersection(mtxs_v))

        # itr1, itr2 = 0, 0
        # while itr1 < len(mtxs_u) and itr2 < len(mtxs_v):
        #     if mtxs_u[itr1] < mtxs_v[itr2]:
        #         itr1 += 1
        #     else:
        #         if mtxs_u[itr1] == mtxs_v[itr2]:
        #             return True
        #         itr2 += 1
    return False


def insert_mutex(ru, rv, mutex_edge_id, mutexes):
    for root in (ru, rv):
        if root in mutexes:
            try:
                idx = next(idx for idx, e_id in enumerate(mutexes[root]) if e_id > mutex_edge_id)
            except StopIteration:
                idx = len(mutexes[root])
            mutexes[root].insert(idx, mutex_edge_id)
        else:
            mutexes[root] = [mutex_edge_id]


def merge_mutexes(root_from, root_to, mutexes):
    if root_from not in mutexes or len(mutexes[root_from]) == 0:
        return
    if root_to not in mutexes or len(mutexes[root_to]) == 0:
        mutexes[root_to] = mutexes[root_from]
        return

    merge_buffer = mutexes[root_from] + mutexes[root_to]
    merge_buffer.sort()

    mutexes[root_to] = merge_buffer
    del mutexes[root_from]


def add_neighbours(position, offset_strides, number_of_nodes, edge_weights, valid_edges, ufd, visited, pq):
    ru = ufd.find(position)
    for i in range(offset_strides.size):
        edge_id = position + i * number_of_nodes
        if valid_edges[edge_id] and not visited[edge_id]:
            # get all directly connected neighbours
            neighbour = position + offset_strides[i]
            rv = ufd.find(neighbour)
            if ru != rv:
                pq.put((-edge_weights[edge_id], edge_id, position, neighbour))
        within_bounds = (offset_strides[i] > 0 or position < number_of_nodes + offset_strides[i]) and \
                        (offset_strides[i] < 0 or offset_strides[i] <= position)
        if within_bounds:
            neg_neighbour = position - offset_strides[i]
            neg_edge_id = neg_neighbour + i * number_of_nodes
            if valid_edges[neg_edge_id] and not visited[neg_edge_id]:
                rv = ufd.find(neg_neighbour)
                if ru != rv:
                    pq.put((-edge_weights[neg_edge_id], neg_edge_id, position, neg_neighbour))
    return pq

def compute_partial_mws_prim_segmentation(edge_weight_exp,
                                          valid_edges_exp,
                                          offsets,
                                          number_of_attractive_channels,
                                          image_shape, iterations):

    visited = np.zeros(edge_weight_exp.size, dtype=bool)
    node_labeling = np.zeros(image_shape).ravel()
    number_of_nodes = node_labeling.size
    number_of_attractive_edges = number_of_nodes * number_of_attractive_channels
    ndims = len(offsets[0])
    array_stride = np.empty(ndims, dtype=np.int64)
    current_stride = 1
    mutexes = {}
    for i in range(ndims-1, -1, -1):
        array_stride[i] = current_stride
        current_stride *= image_shape[i]

    offset_strides = []
    for offset in offsets:
        stride = 0
        for i in range(len(offset)):
            stride += offset[i] * array_stride[i]
        offset_strides.append(stride)

    offset_strides = np.asarray(offset_strides)
    node_ufd = DisjointSet()
    for lbl in range(number_of_nodes):
        node_ufd.find(lbl)

    # mutexes = np.ndarray(number_of_nodes)
    pq = queue.PriorityQueue()

    # start prim from top left node
    add_neighbours(0, offset_strides, number_of_nodes, edge_weight_exp, valid_edges_exp, node_ufd, visited, pq)
    itr = 0
    # iterate over all edges
    cut_edges = []
    used_mtxs = []
    while not pq.empty():
        # extract next element from the queue
        position_vector = pq.get()
        edge_id = position_vector[1]
        u = position_vector[2]
        v = position_vector[3]

        if visited[edge_id]:
            continue
        visited[edge_id] = 1
        # find the current reps and skip if identical or mtx exists
        ru = node_ufd.find(u)
        rv = node_ufd.find(v)
        if ru == rv:
            continue
        if check_mutex(ru, rv, mutexes):
            if edge_id <= number_of_attractive_edges:
                # this edge is attractive and neighbour has different class
                cut_edges.append(edge_id)
            continue

        # check whether this edge is mutex via the edge offset
        if edge_id >= number_of_attractive_edges:
            used_mtxs.append(edge_id)
            insert_mutex(ru, rv, edge_id, mutexes)
        else:
            node_ufd.union(u,v)
            if node_ufd.find(ru) == rv:
                rv, ru = ru, rv
            merge_mutexes(rv, ru, mutexes)

        # add the next node to pq
        add_neighbours(v, offset_strides, number_of_nodes, edge_weight_exp, valid_edges_exp, node_ufd, visited, pq)
        itr += 1
        if itr == iterations:
            break

    # create node labeling from disjoint sets
    # 0's indicate no labeling
    for idx, cc in enumerate(node_ufd.itersets()):
        for node in cc:
            node_labeling[node] = idx+1

    return node_labeling, cut_edges, used_mtxs


def compute_mws_prim_segmentation(edge_weight_exp,
                                  valid_edges_exp,
                                  offsets,
                                  number_of_attractive_channels,
                                  image_shape, iterations):

    visited = np.zeros(edge_weight_exp.size, dtype=bool)
    node_labeling = np.zeros(image_shape).ravel()
    number_of_nodes = node_labeling.size
    number_of_attractive_edges = number_of_nodes * number_of_attractive_channels
    ndims = len(offsets[0])
    array_stride = np.empty(ndims, dtype=np.int64)
    current_stride = 1
    mutexes = {}
    for i in range(ndims-1, -1, -1):
        array_stride[i] = current_stride
        current_stride *= image_shape[i]

    offset_strides = []
    for offset in offsets:
        stride = 0
        for i in range(len(offset)):
            stride += offset[i] * array_stride[i]
        offset_strides.append(stride)

    offset_strides = np.asarray(offset_strides)
    node_ufd = DisjointSet()
    for lbl in range(number_of_nodes):
        node_ufd.find(lbl)

    # mutexes = np.ndarray(number_of_nodes)
    pq = queue.PriorityQueue()

    # start prim from top left node
    add_neighbours(0, offset_strides, number_of_nodes, edge_weight_exp, valid_edges_exp, node_ufd, visited, pq)
    itr = 0
    # iterate over all edges
    while not pq.empty():
        # extract next element from the queue
        position_vector = pq.get()
        edge_id = position_vector[1]
        u = position_vector[2]
        v = position_vector[3]

        if visited[edge_id]:
            continue
        visited[edge_id] = 1
        # find the current reps and skip if identical or mtx exists
        ru = node_ufd.find(u)
        rv = node_ufd.find(v)
        if ru == rv or check_mutex(ru, rv, mutexes):
            continue

        # check whether this edge is mutex via the edge offset
        if edge_id >= number_of_attractive_edges:
            insert_mutex(ru, rv, edge_id, mutexes)
        else:
            node_ufd.union(u,v)
            if node_ufd.find(ru) == rv:
                rv, ru = ru, rv
            merge_mutexes(rv, ru, mutexes)

        # add the next node to pq
        add_neighbours(v, offset_strides, number_of_nodes, edge_weight_exp, valid_edges_exp, node_ufd, visited, pq)
        itr += 1
        if itr == iterations:
            break

    # create node labeling from disjoint sets
    # 0's indicate no labeling
    for idx, cc in enumerate(node_ufd.itersets()):
        for node in cc:
            node_labeling[node] = idx+1

    return node_labeling