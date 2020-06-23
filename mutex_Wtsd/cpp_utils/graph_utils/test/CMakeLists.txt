
include_directories(${MyProject_SOURCE_DIR}/inc)
link_directories(${MyProject_SOURCE_DIR}/src)
add_executable(test_graph_utils test_graph_utils.cxx)
target_link_libraries(test_graph_utils graph_utils)
