# Makes the binary go to /build/bin/benchmarks
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks)

# Check required libraries exist
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Create an interface library containing an interface to all required libs 
add_library(allbenchlib INTERFACE)
target_link_libraries(allbenchlib INTERFACE blis gomp m benchmark_lib kblas)

set(BENCH_EXECUTABLES 
	sscal_bench
	dscal_bench
	sdot_bench
	ddot_bench
	saxpy_bench
	daxpy_bench
	sgemv_bench)

# Create the targets
foreach(exec ${BENCH_EXECUTABLES})
	add_executable(${exec} ${exec}.c helper.c)

	# Allow it to find helper.c
	target_include_directories(${exec} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

	# Link it with all the required libs using the interface lib we made
	target_link_libraries(${exec} allbenchlib)

	# Compile it with the following compiler flags
	target_compile_options(${exec} PRIVATE -O3 -mavx2 -fopenmp -march=znver3)
endforeach()

# Contains the input/verification file generation code
add_subdirectory(generator)