#include "xtensor/xtensor.hpp"
#include <unordered_map>
#include <iostream>
#include <queue>

using namespace std;

namespace graph_utils {
	class EdgeProps{
		public:
			int n_times_used;
	};
	typedef std::tuple <int, int> key;
	struct key_hash : public std::unary_function<key_t, std::size_t>{
		std::size_t operator()(const key& k) const{
			std::string conc_string = std::to_string(std::get<0>(k)) + std::to_string(std::get<1>(k));
			std::hash<std::string> hash_fn;
			return hash_fn(conc_string);
		}
	};

	template<class EDGES>
		inline void find_dense_subgraphs(const xt::xexpression<EDGES> & edges_exp,
				const size_t subgraph_size,
				const std::vector<uint64_t> & n_nodes,
				const std::vector<uint64_t> & n_edges,
//				xt::xexpression<COMPS> & n_components_exp,
				std::vector<uint64_t> & n_components,
				std::vector<uint64_t> & components) {
			// graphs need to be connected, otw stuck in loop
			// expecting shape to be (N, C, 2), edges_exp be of size prod(shape)
			// edges must be non directed unique and ordered like [low_val, high_val]
			cout<<"in function find connected_components"<<endl;
			const auto & edges = edges_exp.derived_cast();
//			auto & n_components = n_components_exp.derived_cast();
//			auto & components = components_exp.derived_cast();

			std::vector<uint64_t> edge_off;
			edge_off.push_back(0);
			for(int i=0; i<n_edges.size()-1; ++i){
				edge_off.push_back(edge_off.back() + n_edges[i]);
			}

			std::vector<uint64_t> node_off;
			node_off.push_back(0);
			for(int i=0; i<n_nodes.size()-1; ++i){
				node_off.push_back(node_off.back() + n_nodes[i]);
			}

			uint64_t n_all_nodes = accumulate(n_nodes.begin(), n_nodes.end(),0);
			vector<vector<uint64_t>> adj_list;
			adj_list.resize(n_all_nodes);
			// add edges to the directed graph
			for(int b=0; b < n_edges.size(); ++b){
				for(int i=0; i < n_edges[b]; ++i){
					adj_list[edges[(i * 2) + edge_off[b]] + node_off[b]]
						.push_back(edges[(i * 2 + 1) + edge_off[b]]);
					adj_list[edges[(i * 2 + 1) + edge_off[b]] + node_off[b]]
						.push_back(edges[(i * 2) + edge_off[b]]);
				}
			}
			// core algorithm
			int anchor;

			for(uint64_t b=0; b < n_edges.size(); ++b){
				std::unordered_map<key, bool, key_hash> unused_edges;
				uint64_t comps = 0;

				for(int i=0; i<n_edges[b]; ++i){
					int sm = std::min(edges[(i * 2) + edge_off[b]], edges[(i * 2 + 1) + edge_off[b]]);
					int bg = std::max(edges[(i * 2 + 1) + edge_off[b]], edges[(i * 2) + edge_off[b]]);
					unused_edges[std::make_tuple(sm, bg)] = true;
				}
				while(unused_edges.size() > 0){
					int i = 0;
					int prio = 0;

					std::unordered_map<uint64_t, bool> drwn_nodes;  // keep current node set
					std::unordered_map<key, bool, key_hash> used_edges; // keep current edge set
					// pq stores neighs to current sub_graph, prio lower the later added as sg grows
					priority_queue<std::pair<int, uint64_t>, 
						std::vector<std::pair<int, uint64_t>>,
					   	std::greater<std::pair<int, uint64_t>>> pq;

					//start every component by a random, so far, unused edge
					int rn = rand() % unused_edges.size();
					auto rn_edge = std::next(std::begin(unused_edges), rn)->first;
					unused_edges.erase(rn_edge);
					used_edges[rn_edge] = true;
					
					drwn_nodes[std::get<0>(rn_edge)] = true;
					drwn_nodes[std::get<1>(rn_edge)] = true;
					
					pq.push(std::make_pair(prio, std::get<0>(rn_edge)));
					pq.push(std::make_pair(prio, std::get<1>(rn_edge)));
					components.push_back(std::get<0>(rn_edge));
					components.push_back(std::get<1>(rn_edge));
					++i;
					int n_samples = 0;

					while(i<subgraph_size){
						std::pair<int, uint64_t> top_pos = pq.top();
						pq.pop();
						++top_pos.first;
						++n_samples;
						++prio;
						pq.push(top_pos);
						
						for(auto neigh : adj_list[top_pos.second + node_off[b]]){
							uint64_t sm, bg;
							sm = std::min(top_pos.second, neigh);
							bg = std::max(neigh, top_pos.second);
							if(used_edges.find(std::make_tuple(sm, bg)) == used_edges.end() & 
								(drwn_nodes.find(neigh)!=drwn_nodes.end() | n_samples >= pq.size())){
								pq.push(std::make_pair(prio, neigh));
								components.push_back(top_pos.second);
								components.push_back(neigh);
								
								unused_edges.erase(std::make_tuple(sm, bg));
								used_edges[std::make_tuple(sm, bg)] = true;
								
								++i;	
								n_samples = 0;
							}
						}
					}
					++comps;
					pq = priority_queue<std::pair<int, uint64_t>,
							std::vector<std::pair<int, uint64_t>>,
							std::greater<std::pair<int, uint64_t>>>();  // priority_queue does not provide clear func 
					used_edges.clear();
					drwn_nodes.clear();
				}
				n_components[b] = comps;
				unused_edges.clear();
			}
		}
}
