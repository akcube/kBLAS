# Creates the target 
add_executable(mGBPS mgbps.c)

# Compiles with optimization flags
target_compile_options(mGBPS PRIVATE -O3 -ffast-math -mavx2 -fopenmp -march=znver3)

# Link it with OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(mGBPS PRIVATE gomp)

# Links it with the benchmark library
target_link_libraries(mGBPS PRIVATE benchmark_lib)