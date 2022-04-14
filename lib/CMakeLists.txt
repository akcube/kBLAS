# Creates the library 
add_library(benchmark_lib benchmark.c benchmark.h)

# We need this directory, and users of our library will need it too
target_include_directories(benchmark_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Compiles with optimization flags
target_compile_options(benchmark_lib PRIVATE -O3 -ffast-math -mavx2 -fopenmp -march=znver3)
