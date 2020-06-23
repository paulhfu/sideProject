

add_executable(graph_utils graph_utils.cxx)
target_link_libraries(graph_utils ${xtensor-python_INCLUDE_DIRS} pybind11::embed)
target_include_directories(graph_utils PUBLIC ${xtensor-python_INCLUDE_DIRS})
target_include_directories(graph_utils PUBLIC ${NP})
