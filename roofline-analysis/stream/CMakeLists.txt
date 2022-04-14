# Creates the target 
add_executable(stream stream.c)

# Link it with OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(stream PRIVATE gomp)

# Compiles with optimization flags + some necessary defines
target_compile_options(stream PRIVATE -O3 -ffast-math -mavx2 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 -march=znver3 -DTUNED)