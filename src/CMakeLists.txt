# Creates the library 
add_library(kblas kblas_dscal.c kblas_sscal.c kblas_sdot.c kblas_ddot.c kblas_saxpy.c kblas_daxpy.c)

# We need this directory, and users of our library will need it too
target_include_directories(kblas PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Compiles with optimization flags
target_compile_options(kblas PRIVATE -O3 -ffast-math -mavx2 -fopenmp -march=znver3)
