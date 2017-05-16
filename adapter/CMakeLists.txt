set (PACKAGE_NAME "Libmesh Adapter Package")
set (PACKAGE_DESCIPTION "This package provides a convenient adapter for working with a libmesh mesh instance")
set (LIBRARY_NAME libmesh-adapter)

file (GLOB HEADERS *.hpp)
file (GLOB SRC *.cpp)

include_directories(${CMAKE_SOURCE_DIR})

add_library(${LIBRARY_NAME} SHARED ${SRC})

install(FILES ${HEADERS} DESTINATION include)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

# Gather tests
file (GLOB test_files test/*.cpp)
add_tests("${test_files}" "${CMAKE_CURRENT_SOURCE_DIR}" "${LIBRARY_NAME}")
