set(BOOST_ROOT $ENV{BOOST_DIR})
find_package(Boost COMPONENTS unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})
function(add_boost_tests tests test_include_dirs test_libs)
      add_tests_with_flags("${tests}" "${test_include_dirs}" "${test_libs}" "")
endfunction()
function(add_tests_with_flags tests test_include_dirs test_libs test_cflags)
    if(Boost_FOUND)
      # Include all the required headers and BOOST
      include_directories("${Boost_INCLUDE_DIRS}" "${test_include_dirs}")
      # Make executables and link libraries for testers
      foreach(test ${tests})
         message(STATUS "Adding test ${test}")
         get_filename_component(testName ${test} NAME_WE)
         # Add the executable to the build and the test list
         add_executable(${testName} ${test})
         add_test(${testName} ${testName} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
         # Overwrite the default compile flags if needed.
         if (NOT test_cflags STREQUAL "")
            target_compile_options(${testName} PUBLIC "${test_cflags}")
         endif()
         # Only link libraries if they are provided
         if (test_libs)
            target_link_libraries(${testName} ${test_libs})
         endif(test_libs) 
         # Add a label so the tests can be run on its own by label
         set_property(TEST ${testName} PROPERTY LABELS ${PACKAGE_NAME})
      endforeach(test ${tests})
    else(Boost_FOUND)
      message(WARNING "Tests - ${tests} - not built because BOOST is unavailable!")
    endif()  
endfunction()