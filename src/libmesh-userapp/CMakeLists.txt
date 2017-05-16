
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
include(boost_helper)
link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Kokkos_LIBRARY_DIRS})
link_directories(${DataTransferKit_LIBRARY_DIRS})
add_boost_tests(tstLibmeshUserApplication.cpp "${CMAKE_SOURCE_DIR}/packages/Kokkos/src;${TPL_Libmesh_INCLUDE_DIRS};${DataTransferKitKokkos_INCLUDE_DIRS};${Kokkos_INCLUDE_DIRS}" "${Boost_LIBRARIES};${TPL_Libmesh_LIBRARIES};${Kokkos_LIBRARIES};${Teuchos_LIBRARIES};${DataTransferKit_LIBRARIES}")

