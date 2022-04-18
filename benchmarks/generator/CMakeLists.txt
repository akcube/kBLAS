# Creates it as a new project to compile with C++
project(
  BLAS_FILE_GEN
  VERSION 1
  DESCRIPTION "Generates input/verification files for testing/benchmarking BLAS implementations"
  LANGUAGES CXX)

# Creates the target 
add_executable(gen gen.cpp)

# Compiles with optimization flags
target_compile_options(gen PRIVATE -O0 -g)

# Link it with BLIS
target_link_libraries(gen PRIVATE blis)

# Link it with OpenMP and pthreads
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(gen PRIVATE gomp)
target_link_libraries(gen PRIVATE pthread)

# Create required directories during build
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks/input)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks/verification)
set(INPUT_DIR ${CMAKE_BINARY_DIR}/bin/benchmarks/input)
set(VERIF_DIR ${CMAKE_BINARY_DIR}/bin/benchmarks/verification)

# Input file directories
file(MAKE_DIRECTORY ${INPUT_DIR}/sscal)
file(MAKE_DIRECTORY ${INPUT_DIR}/dscal)
file(MAKE_DIRECTORY ${INPUT_DIR}/sdot)
file(MAKE_DIRECTORY ${INPUT_DIR}/ddot)
file(MAKE_DIRECTORY ${INPUT_DIR}/saxpy)
file(MAKE_DIRECTORY ${INPUT_DIR}/daxpy)

# Verification file directories
file(MAKE_DIRECTORY ${VERIF_DIR}/sscal)
file(MAKE_DIRECTORY ${VERIF_DIR}/dscal)
file(MAKE_DIRECTORY ${VERIF_DIR}/saxpy)
file(MAKE_DIRECTORY ${VERIF_DIR}/daxpy)