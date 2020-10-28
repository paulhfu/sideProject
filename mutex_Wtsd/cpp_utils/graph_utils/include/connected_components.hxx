#include "xtensor/xtensor.hpp"
#include <unordered_map>
#include <iostream>
#include <queue>


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
				const std::vector<uint64_t> & subgraph_size,
				const std::vector<uint64_t> & n_nodes,
				const std::vector<uint64_t> & n_edges,
				std::vector<uint64_t> & n_components,
				std::vector<uint64_t> & components) {
			// graphs need to be connected, otw stuck in loop
			// expecting shape to be (N, C, 2), edges_exp be of size prod(shape)
			// edges must be non directed unique and ordered like [low_val, high_val]

			const auto & edges = edges_exp.derived_cast();

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
			std::vector<std::vector<uint64_t>> adj_list;
			adj_list.resize(n_all_nodes);
			// add edges to the directed graph
			for(int b=0; b < n_edges.size(); ++b){
				for(int i=0; i < n_edges[b]; i+=2){
					adj_list[edges[i + edge_off[b]] + node_off[b]]
						.push_back(edges[i + 1 + edge_off[b]]);
					adj_list[edges[i + 1 + edge_off[b]] + node_off[b]]
						.push_back(edges[i + edge_off[b]]);
				}
			}
			// core algorithm
			for(uint64_t ssize_it=0; ssize_it < subgraph_size.size(); ++ssize_it){
				for(uint64_t b=0; b < n_edges.size(); ++b){
					std::unordered_map<key, bool, key_hash> unused_edges;
					uint64_t comps;
					comps = 0;

					for(int i=0; i<n_edges[b]; i+=2){
						int sm = std::min(edges[i + edge_off[b]], edges[i + 1 + edge_off[b]]);
						int bg = std::max(edges[i + 1 + edge_off[b]], edges[i + edge_off[b]]);
						unused_edges[std::make_tuple(sm, bg)] = true;
					}
					while(unused_edges.size() > 0){
						int rn, n_samples, prio, i;
						key rn_edge;
						rn = 0;
						n_samples = 0;
						prio = 0;
						i = 0;

						std::unordered_map<uint64_t, bool> drwn_nodes;  // keep current node set
						std::unordered_map<key, bool, key_hash> used_edges; // keep current edge set
						// pq stores neighs to current sub_graph, prio lower the later added as sg grows
						std::priority_queue<std::pair<int, uint64_t>, 
							std::vector<std::pair<int, uint64_t>>,
							std::greater<std::pair<int, uint64_t>>> pq;

						//start every component by a random, so far, unused edge
						rn = rand() % unused_edges.size();
						rn_edge = std::next(std::begin(unused_edges), rn)->first;
						unused_edges.erase(rn_edge);
						used_edges[rn_edge] = true;
						
						drwn_nodes[std::get<0>(rn_edge)] = true;
						drwn_nodes[std::get<1>(rn_edge)] = true;
						
						pq.push(std::make_pair(prio, std::get<0>(rn_edge)));
						pq.push(std::make_pair(prio, std::get<1>(rn_edge)));
						components.push_back(std::get<0>(rn_edge));
						components.push_back(std::get<1>(rn_edge));
						++i;

						while(true){
							bool all_added;
							std::pair<int, uint64_t> top_pos = pq.top();

							pq.pop();
							++prio;
							all_added = true;
							
							for(auto neigh : adj_list[top_pos.second + node_off[b]]){
								uint64_t sm, bg;
								sm = std::min(top_pos.second, neigh);
								bg = std::max(neigh, top_pos.second);

								if(used_edges.find(std::make_tuple(sm, bg)) == used_edges.end()){
									if(drwn_nodes.find(neigh)!=drwn_nodes.end() | n_samples > pq.size() + 1){
									
										pq.push(std::make_pair(prio, neigh));
										drwn_nodes[neigh] = true;		 

										components.push_back(sm);
										components.push_back(bg);
										++i;
										if(i==subgraph_size[ssize_it]){break;}
										
										unused_edges.erase(std::make_tuple(sm, bg));
										used_edges[std::make_tuple(sm, bg)] = true;
										
										n_samples = 0;
									}
									else{
										all_added = false;
									}
								}
							}
							if(i==subgraph_size[ssize_it]){break;}
							if(!all_added){
								top_pos.first += 5;  // this value is a empirical result
								pq.push(top_pos);
							}
							++n_samples;
						}
						++comps;
						pq = std::priority_queue<std::pair<int, uint64_t>,
								std::vector<std::pair<int, uint64_t>>,
								std::greater<std::pair<int, uint64_t>>>();  // priority_queue does not provide clear func 
						used_edges.clear();
						drwn_nodes.clear();
					}
					n_components[n_edges.size() * ssize_it + b] = comps;
					unused_edges.clear();
				}
			}
		}

    inline void separate_subgraphs(
			const uint64_t & bs,
			const std::vector<uint64_t> & n_comps,
            const std::vector<uint64_t> & edges,
            const std::vector<uint64_t> & subgraph_size,
            std::vector<uint64_t> & separate_subgraphs) {
        // separates a set of possibly connected subgraphs
//         std::cout<<"num edge vals are  "<<edges.size()<<std::endl;
        std::unordered_map<uint64_t, uint64_t> node_rep;
        int node = 0;
        int iter = 0;
        int _iter = 0;
//         for(uint64_t it=0; it < n_comps.size(); ++it){
//             std::cout<<"comp" << it <<" is  "<<n_comps[it]<<std::endl;
//         }
		for(uint64_t ssize_it=0; ssize_it < subgraph_size.size(); ++ssize_it){
		    _iter = iter;
		    iter += std::accumulate(n_comps.begin() + bs * ssize_it, n_comps.begin() + bs * (ssize_it+1), 0) * subgraph_size[ssize_it]*2;
// 		    std::cout<<"iter is  "<<iter<<std::endl;
			for(int i=_iter; i<iter; i+=subgraph_size[ssize_it]*2){
				for(int j=0; j<subgraph_size[ssize_it]*2; j+=2){
					if(node_rep.find(edges[i+j]) == node_rep.end()){
						node_rep[edges[i+j]] = node;
						node++;
					}
					if(node_rep.find(edges[i+j+1]) == node_rep.end()){
						node_rep[edges[i+j+1]] = node;
						node++;
					}
					separate_subgraphs[i+j] = node_rep[edges[i+j]];
					separate_subgraphs[i+j+1] = node_rep[edges[i+j+1]];
				}
				node_rep.clear();
			}
		}
    }
}
