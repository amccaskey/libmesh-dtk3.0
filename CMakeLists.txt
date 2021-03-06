project(libmesh-dtk3.0 CXX)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
message(STATUS "C++ version ${CXX_STANDARD} configured.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
set(BOOST_ROOT $ENV{BOOST_DIR})

include(boost_helper)

find_package(MPI REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

enable_testing()

include_directories(${MPI_INCLUDE_PATH})
include_directories(${DTK_TRILINOS_ROOT}/include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Libmesh_INCLUDE_DIRS})
include_directories(${PETSC_INCLUDE_DIR})

add_subdirectory(adapter)
add_subdirectory(userapp)
