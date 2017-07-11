set (PACKAGE_NAME "Libmesh Adapter Package")
set (PACKAGE_DESCIPTION "This package provides a convenient adapter for working with a libmesh mesh instance")
set (LIBRARY_NAME libmesh-adapter)

file (GLOB HEADERS *.hpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

install(FILES ${HEADERS} DESTINATION include)

link_directories("${Libmesh_LIBRARY_DIRS}")
link_directories("${DTK_TRILINOS_ROOT}/lib")
link_directories("${PETSC_LIBRARY_DIR}")

# Gather tests
file (GLOB test_files test/*.cpp)
add_boost_tests("${test_files}" "${CMAKE_CURRENT_SOURCE_DIR}" "mesh_opt;${MPI_LIBRARIES};${Boost_LIBRARIES};teuchoscomm;kokkoscore;teuchoscore;teuchoskokkoscomm;teuchosnumerics;teuchosparameterlist;dtk_interface;dtk_utils;tpetra;tpetraclassic;tpetraclassiclinalg;tpetraext;tpetrainout;tpetrakernels;tpetraclassicnodeapi;petsc")
