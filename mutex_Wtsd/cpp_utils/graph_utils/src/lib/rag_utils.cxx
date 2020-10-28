#define FORCE_IMPORT_ARRAY
#include "connected_components.hxx"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/chrono.h"

#include "xtensor-python/pytensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor-python/pyarray.hpp"
#include <numpy/arrayobject.h>
#include <iostream>
#include <cmath>
#include <typeinfo>

namespace py = pybind11;

PYBIND11_MODULE(_rag_utils, m){
	m.doc() = R"pbdoc(
        Pybind11 plugin for graph computations on region adjacency graphs
        -----------------------

        .. currentmodule:: rag_utils

        .. autosummary::
           :toctree: _generate

           find_connected_components
    )pbdoc";

    xt::import_numpy();

    using namespace graph_utils;

    m.def("find_dense_subgraphs_impl", [](const xt::pyarray<uint64_t> & edges,
                                     const std::vector<uint64_t> & sizes,
									 const std::vector<uint64_t> & n_nodes,
									 const std::vector<uint64_t> & n_edges) {

			std::vector<uint64_t> components;
			std::vector<uint64_t> sep_sgs;
			uint64_t bs = n_edges.size();
			std::vector<uint64_t> n_components(n_edges.size() * sizes.size(), 0);
			{
				py::gil_scoped_release allowThreads;
				find_dense_subgraphs(edges, sizes, n_nodes, n_edges, n_components, components);
				sep_sgs.resize(components.size(), 0);
				separate_subgraphs(bs, n_components, components, sizes, sep_sgs);
			}
			xt::xarray<uint64_t> py_comps = xt::adapt(components, {components.size()});
			xt::xarray<uint64_t> py_n_comps = xt::adapt(n_components, {n_components.size()});
			xt::xarray<uint64_t> py_sep_sgs = xt::adapt(sep_sgs, {sep_sgs.size()});

			py::tuple out = py::make_tuple(py_comps, py_n_comps, py_sep_sgs);
			return out;
    }, py::arg("edges"),
       py::arg("sizes"),
       py::arg("n_nodes"),
       py::arg("shape")
    );
}
