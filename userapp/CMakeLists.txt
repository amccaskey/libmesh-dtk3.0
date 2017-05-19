set (PACKAGE_NAME "Libmesh User Application Package")
set (PACKAGE_DESCIPTION "This package provides the functional api for DTK to interact with Libmesh")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/adapter)

install(FILES ${HEADERS} DESTINATION include)

link_directories("${Libmesh_LIBRARY_DIRS}")
link_directories("${DTK_TRILINOS_ROOT}/lib")
link_directories("${Boost_LIBRARY_DIRS}")

# Gather tests
file (GLOB test_files test/*.cpp)
add_boost_tests("${test_files}" "${CMAKE_CURRENT_SOURCE_DIR}" "mesh_opt;${MPI_LIBRARIES};${Boost_LIBRARIES};boost_mpi;boost_serialization;teuchoscomm;kokkoscore;teuchoscore;teuchoskokkoscomm;teuchosnumerics;teuchosparameterlist;dtk_interface;dtk_utils;tpetra;tpetraclassic;tpetraclassiclinalg;tpetraext;tpetrainout;tpetrakernels;tpetraclassicnodeapi")
