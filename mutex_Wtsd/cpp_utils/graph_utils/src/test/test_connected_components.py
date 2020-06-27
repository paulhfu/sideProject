import unittest
import os
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pkgutil
from random import randint

# this is the package we are inspecting -- for example 'email' from stdlib
import rag_utils as ru
package = ru
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print("Found submodule %s (is a package: %s)" % (modname, ispkg))



class TestRagUtils(unittest.TestCase):

    def test_cc(self):
        from rag_utils import find_dense_subgraphs
        edges = []
        nodes = []
        for i in range(10):
            file = h5py.File("rags/rag_" + str(i) + ".h5", "r")
            edges.append(file["edges"][:]
            nodes.append(np.unique(edges[-1]))

        bsgs = find_dense_subgraphs(edges, 10, nodes)

        for c, sgs in enumerate(bsgs):
            for i, sg in enumerate(sgs):
                fig = plt.figure()
                
                g = nx.Graph()
                g.add_nodes_from(nodes[c])
                g.add_edges_from(edges[c])
                pos = nx.spring_layout(g)

                nsg = np.unique(sg)

                nx.draw(g, pos, with_labels=True, node_color='b', edge_color='b')
                nx.draw_networkx_nodes(g, pos, nodelist=sg, node_color='r')
                nx.draw_networkx_edges(g, pos, edgelist=nsg, edge_color='r', width=3)

                directory = "graphs_pics/graph_" + str(c)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                plt.savefig(directory + "/sub_graph_" + str(i) + ".png")
                plt.clf()
                g.clear()

if __name__ == '__main__':
    unittest.main()
