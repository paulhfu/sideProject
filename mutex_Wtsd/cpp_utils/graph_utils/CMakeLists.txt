cmake_minimum_required(VERSION 3.1)

project(rag_utils)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##############################
# Check and enable C++ 14
##############################

include(CheckCXXCompilerFlag)
# make sure the compiler supports c++14
# FIXME I think this won't work for MSVC
CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
if(NOT HAS_CPP14_FLAG)
    message(FATAL_ERROR "Unsupported compiler -- requires C++14 support!")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# This should make the visbility warnings go away,
# but does not do it
# set(CMAKE_CXX_VISIBILITY_PREFIX default)


# set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

execute_process (
    COMMAND bash -c "which python"
	OUTPUT_VARIABLE PYTHON_PATH
)

# Set CMAKE_PREFIX_PATH to the conda env, but allow changing it
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} CACHE PATH "")

MESSAGE(STATUS "Setting cmake prefix path to ${CMAKE_PREFIX_PATH}")
include_directories("${CMAKE_PREFIX_PATH}/include")

# find libraries - pthread
find_package(Threads)

# boost
SET(BOOST_ROOT "${CMAKE_PREFIX_PATH}")
SET(BOOST_LIBRARYDIR "${CMAKE_PREFIX_PATH}/lib")
SET(Boost_NO_SYSTEM_PATHS ON)
find_package(Boost 1.63.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


###############################
# Python-bindings
###############################

find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++14)

find_package(NUMPY REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

find_package(xtensor-python REQUIRED)
include_directories(${xtensor-python_INCLUDE_DIRS})

# find global headers
file(GLOB_RECURSE headers include/*.hxx)
file(GLOB_RECURSE headers ${CMAKE_INSTALL_PREFIX}/include/*.hxx)
include_directories(include)
# add subdirectories
add_subdirectory(src)

# install the headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.hxx"
  PATTERN "*.hpp"
  PATTERN "*.h"
)



