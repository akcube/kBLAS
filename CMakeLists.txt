# Works with 3.11 and tested through 3.23
cmake_minimum_required(VERSION 3.11...3.23)

# Project name and a few useful settings. Other commands can pick up the results
project(
  KBLAS
  VERSION 1
  DESCRIPTION "Another implementation of the BLAS library in C"
  LANGUAGES C)

# Make all the libraries created go to /build/lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# The KBLAS library code is here
add_subdirectory(src)

# The benchmark library code is here
add_subdirectory(lib)

# This includes the roofline memory analysis for bandwidth and FLOPS
add_subdirectory(roofline-analysis)

# Contains the code for all the KBLAS vs BLIS benchmarks
add_subdirectory(benchmarks)