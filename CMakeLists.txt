project(libmesh-dtk3.0 CXX)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_DISABLE_IN_SOURCE_BUILDS ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
message(STATUS "C++ version ${CXX_STANDARD} configured.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

find_package(Boost COMPONENTS system program_options REQUIRED)
message(STATUS "Found Boost Headers = ${Boost_INCLUDE_DIRS}")
message(STATUS "Found Boost Libraries = ${Boost_LIBRARIES}")

include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(src)
