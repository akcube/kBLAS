# Creates the library 
add_library(benchmark_lib benchmark.c benchmark.h)

# We need this directory, and users of our library will need it too
target_include_directories(benchmark_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link it with OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(benchmark_lib PRIVATE gomp)

# Compiles with optimization flags
target_compile_options(benchmark_lib PRIVATE -O3 -ffast-math -mavx2 -fopenmp -march=znver3)
