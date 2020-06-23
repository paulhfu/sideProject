#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/segmentation/mutex_watershed.hxx"
#include "affogato/segmentation/semantic_mutex_watershed.hxx"
#include "affogato/segmentation/connected_components.hxx"
#include "affogato/segmentation/zwatershed.hxx"
#include "affogato/segmentation/grid_graph.hxx"

namespace py = pybind11;

PYBIND11_MODULE(_segmentation, m)
{
    xt::import_numpy();
    m.doc() = "segmentation module of affogato";

    using namespace affogato;

    m.def("compute_mws_prim_segmentation_dbg_impl",[](const xt::pytensor<float, 1> & edge_weights,
                                                  const xt::pytensor<bool, 1> & valid_edges,
                                                  const std::vector<std::vector<int>> & offsets,
                                                  const size_t number_of_attractive_channels,
                                                  const std::vector<int> & image_shape){
        int64_t number_of_nodes = 1;
        for (auto & s: image_shape){
            number_of_nodes *= s;
        }
        xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({number_of_nodes});
		xt::pytensor<uint64_t, 1> neighbors = xt::zeros<uint64_t>({number_of_nodes * 4});
		xt::pytensor<uint64_t, 1> cutting_edges = xt::zeros<uint64_t>({number_of_nodes * number_of_attractive_channels});
		xt::pytensor<uint64_t, 1> mtxs = xt::zeros<uint64_t>({number_of_nodes * (offsets.size()-number_of_attractive_channels)});
		xt::pytensor<uint64_t, 1> indices = xt::zeros<uint64_t>({number_of_nodes * 4});
        {
            py::gil_scoped_release allowThreads;
            segmentation::compute_mws_prim_segmentation_dbg(edge_weights,
                                                        valid_edges,
                                                        offsets,
                                                        number_of_attractive_channels,
                                                        image_shape,
                                                        node_labeling, 
														neighbors,
														cutting_edges,
														mtxs,
														indices);
        }
        return std::make_tuple(node_labeling, neighbors, cutting_edges, mtxs, indices);
    }, py::arg("edge_weights"),
       py::arg("valid_edges"),
       py::arg("offsets"),
       py::arg("number_of_attractive_channels"),
       py::arg("image_shape"));
}
