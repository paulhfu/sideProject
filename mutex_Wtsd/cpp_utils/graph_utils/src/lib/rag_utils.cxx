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

using namespace std;
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
                                     const size_t size,
									 const std::vector<uint64_t> & n_nodes,
									 const std::vector<uint64_t> & n_edges) {
			// expecting edges to have shape dim (N, C, 2) 
			
//			cout<<"edge size: "<<n_edges.size()<<endl;
//			int batch_size = 0;
//			for(int i=0; i<n_edges.size(); ++i){
//				n_components[i] = ceil((float)n_edges[i] / (float)size);
//				cout<<"val: "<<n_components[i]<<endl;
//				batch_size += n_components[i];
//			}
//			for(auto & comp : n_components){
//				cout<<"sizes are:  "<<batch_size<<",  "<<comp<<endl;
//			}
//			xt::pyarray<uint64_t> components = xt::zeros<uint64_t>({batch_size * size * 2});

//			xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({(int64_t) number_of_labels});
//			xt::pytensor<uint64_t, 1> n_components = xt::zeros<uint64_t>({(int64_t) n_edges.size()});			
			std::vector<uint64_t> components;
			std::vector<uint64_t> n_components(n_edges.size(), 0);
			{
				py::gil_scoped_release allowThreads;
				find_dense_subgraphs(edges, size, n_nodes, n_edges, n_components, components);
			}
			xt::xarray<uint64_t> py_comps = xt::adapt(components, {components.size()});   
			xt::xarray<uint64_t> py_n_comps = xt::adapt(n_components, {n_components.size()});   

//			return std::make_pair(py_comps, n_components);
			py::tuple out = py::make_tuple(py_comps, py_n_comps);
			return out;
    }, py::arg("edges"),
       py::arg("size"),
       py::arg("n_nodes"),
       py::arg("shape")
    );
}
