cmake_minimum_required(VERSION 2.8.12)
project(_clustering)

### Don't know to what extent we can rely on PyBind to deal with Debug/Release configurations
# Comment/Uncomment in `main.cpp` and `pybind.cpp` too
set(CMAKE_BUILD_TYPE Release)

### Global compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)       # Use of tuple auto binding

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -lblas -llapack")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

### PyBind Support 
set(PYBIND11_CPP_STANDARD -std=c++17)           # Last C++ version
find_package(pybind11 REQUIRED)                 # Use `find_package` if already available
include_directories("${PROJECT_SOURCE_DIR}")    # So that it can find files via the relative path, e.g: `utils/clustering`
# add_subdirectory("${LIB}/pybind11")           # Use `add_subdirectory` if in the same directory
pybind11_add_module(_clustering pybind.cpp) 

### Subproject paths
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(UTILS  ${ROOT}/utils)
set(LIB  ${ROOT}/lib)
set(TESTS ${ROOT}/tests)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ROOT}/build)   # If this is before the pybind import, the module won't be built in the 
                                                    # right place.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ROOT}/build)

### Main 
add_executable(main "${ROOT}/main.cpp")
# Stupid, but could not add them all in a single string
target_compile_options(main PRIVATE "-Wall")
target_compile_options(main PRIVATE "-Wextra")
target_compile_options(main PRIVATE "-pedantic-errors")
target_compile_options(main PRIVATE "-Weffc++")
target_compile_options(main PRIVATE "-Wsign-conversion")
target_compile_options(main PRIVATE "-Werror")
if (CMAKE_BUILD_TYPE MATCHES Debug)
    # Again, that's a bit ugly. The idea is to disable the error on unknown pragma when using
    # Debug variants, where OpenMP is not used. 
    message(STATUS "Debug configuration. Disabling unknown-pragmas warnings as errors.")
    target_compile_options(main PRIVATE "-Wno-unknown-pragmas")
endif()
target_include_directories(main PRIVATE ${UTILS})

### Google test support
# It's possible to remove the code from lib and download it directly. Code and instructions from:
# https://github.com/google/googletest/blob/master/googletest/README.md
# TODO: check if there's not a better way to include the tests? See: https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html
set(GOOGLE "${LIB}/googletest")
add_subdirectory(${GOOGLE})
add_executable(clustering_test ${TESTS}/main.cpp
    "${GOOGLE}/googletest/src/gtest_main.cc"
    )
target_include_directories(clustering_test PUBLIC ${GOOGLE}/googletest/include)
target_link_libraries(clustering_test PRIVATE gtest)
add_test(NAME test COMMAND clustering_test)

### OpenMP Support
find_package(OpenMP)
if((OpenMP_CXX_FOUND) AND (CMAKE_BUILD_TYPE MATCHES Release))
    target_link_libraries(_clustering PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(clustering_test PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()