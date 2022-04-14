# Makes the binary go to /build/bin/kblas_benchmarks
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/kblas_benchmarks)

# Creates the target 
add_executable(scal_bench scal_bench.c)

# Compiles with optimization flags
target_compile_options(scal_bench PRIVATE -O3 -ffast-math -mavx2 -fopenmp -march=znver3)

# Link it with BLIS
target_link_libraries(scal_bench PRIVATE blis)

# Link it with OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(scal_bench PRIVATE gomp)

# Links it with the benchmark library and kblas
target_link_libraries(scal_bench PRIVATE benchmark_lib)
target_link_libraries(scal_bench PRIVATE kblas)