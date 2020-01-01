cmake_minimum_required(VERSION 3.1)

project(experiments)
file(COPY /home/drford/Documents/Masterarbeit/sideProject/affogatoTest/affogato/python/affogato/_affogato.cpython-37m-x86_64-linux-gnu.so DESTINATION /home/drford/Documents/Masterarbeit/sideProject/affogatoTest/affogato/experiments)
#file(RENAME _affogato.cpython-37m-x86_64-linux-gnu.so libaffogato.so)
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#link_directories(/home/drford/Documents/Masterarbeit/sideProject/affogatoTest/affogato/src/python/lib)
#_affogato.cpython-37m-x86_64-linux-gnu.so
find_library(MY_LIB NAMES segmentation PATHS /home/drford/Documents/Masterarbeit/sideProject/affogatoTest/affogato/experiments)
message(STATUS "Found segmentation: ${MY_LIB}")
find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}")
find_library(AFF NAMES affogato PATHS /home/drford/Documents/Masterarbeit/sideProject/affogatoTest/affogato/experiments)
message(STATUS "Found affogato: ${AFF}")
find_package(xtensor-python REQUIRED)
message(STATUS "Found xtensor-python: ${xtensor-python_INCLUDE_DIRS}")
find_library(NP NAMES numpy)
message(STATUS "Found numpy: ${NP}")
#find_package(xtl REQUIRED HINTS /home/drford/miniconda3/)
#find_package(xtensor REQUIRED HINTS /home/drford/miniconda3/)

add_executable(test_seg test_seg.cxx)
target_link_libraries(test_seg ${xtensor-python_INCLUDE_DIRS} pybind11::embed ${AFF})
target_include_directories(test_seg PUBLIC /home/drford/miniconda3/lib/python3.7/site-packages/numpy/core/include)
target_include_directories(test_seg PUBLIC /home/drford/Documents/Masterarbeit/sideProject/affogatoTest/affogato/include)
#target_include_directories(test_seg PUBLIC ${pybind11_INCLUDE_DIR})
target_include_directories(test_seg PUBLIC ${xtensor-python_INCLUDE_DIRS})
#target_include_directories(test_seg PUBLIC ${xtensor_INCLUDE_DIRS})
#target_include_directories(test_seg PRIVATE /home/drford/miniconda3/include/python3.7m)
#target_include_directories(test_seg PRIVATE /home/drford/miniconda3/include)
#target_include_directories(test_seg PUBLIC /home/drford/miniconda3/lib/python3.7/site-packages/numpy/core/include)


