# Makes the binary go to /build/bin/benchmarks
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks)

# Creates the target 
add_executable(bench sscal_bench.c helper.c)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Compiles with optimization flags
target_compile_options(bench PRIVATE -O3 -mavx2 -fopenmp -march=znver3)

# Link it with BLIS
target_link_libraries(bench PRIVATE blis)

# Link it with OpenMP and pthreads
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(bench PRIVATE gomp)
target_link_libraries(bench PRIVATE m)

# Links it with the benchmark library and kblas
target_link_libraries(bench PRIVATE benchmark_lib)
target_link_libraries(bench PRIVATE kblas)

# Contains the input/verification file generation code
add_subdirectory(generator)