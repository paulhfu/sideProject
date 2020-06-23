#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"
#include "affogato/segmentation/mutex_watershed.hxx"
#include <iostream>
#include <tr1/unordered_map>

using namespace std::tr1;

namespace py = pybind11;

int main (int argc, char *argv[]) {
	std::cout << "Hello" << std::endl;
        int64_t number_of_nodes = 1;
		std::vector<int> image_shape({2,2});
        for (auto & s: image_shape){
            number_of_nodes *= s;
        }
	uint64_t n1, n2, r1, r2, sm=1, bg=5;

	class Value{
		public:
			std::vector<uint64_t> cut_edges;
			std::vector<uint64_t> mutexes;
			Value(){ 
			} 
			Value(std::vector<uint64_t> cut_edges, std::vector<uint64_t> mutexes){ 
				this->cut_edges = cut_edges;
				this->mutexes = mutexes; 
			} 
	};
	typedef std::tuple <int, int> key;
	struct key_hash : public std::unary_function<key_t, std::size_t>{
		std::size_t operator()(const key& k) const{
			std::string conc_string = std::to_string(std::get<0>(k)) + std::to_string(std::get<1>(k));
			std::cout << "conc string: " << conc_string << '\n';
			std::hash<std::string> hash_fn;
			return hash_fn(conc_string);
		}
	};
	unordered_map< key, std::shared_ptr<Value>, key_hash > neighbors_features;
	if (neighbors_features.find(std::make_tuple(sm, bg)) != neighbors_features.end()){
		std::cout << "inif1" << std::endl;
	}
	std::cout<<"testOK"<<std::endl;
	xt::pytensor<double, 1> m_data;
/*        xt::pytensor<uint64_t, 1> edge_weights({1,1,1,1, 1,1,1,1, 0,0,0,0});
	xt::pytensor<uint64_t, 1> valid_edges({0,0,0,0, 0,0,0,0, 0,0,0,0});
	std::vector<std::vector<int>> offsets = {{1,0}, {1,0}, {1,0}};
	size_t number_of_attractive_channels = 2;
	xt::pytensor<uint64_t, 1> node_labeling({0,0,0,0});
	xt::pyarray<uint64_t> neighbors = xt::zeros<uint64_t>({0});
	xt::pyarray<uint64_t> cutting_edges = xt::zeros<uint64_t>({0});
	xt::pyarray<uint64_t> mtxs = xt::zeros<uint64_t>({0});
	xt::pyarray<uint64_t> indices = xt::zeros<uint64_t>({0});

        affogato::segmentation::compute_mws_prim_segmentation_dbg(edge_weights,
                                                    valid_edges,
                                                    offsets,
                                                    number_of_attractive_channels,
                                                    image_shape,
                                                    node_labeling, 
													neighbors,
													cutting_edges,
													mtxs,
													indices);
*/
        return 0;
}
