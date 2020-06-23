
target_link_libraries(test_seg ${xtensor-python_INCLUDE_DIRS} pybind11::embed)
target_include_directories(test_seg PUBLIC ${xtensor-python_INCLUDE_DIRS})

add_library(graph_utils graph_utils.hxx)
