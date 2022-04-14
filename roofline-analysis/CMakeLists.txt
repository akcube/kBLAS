# Makes the binaries go to /build/bin/roofline-analysis
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/roofline-analysis)

# The tuned version of the stream benchmark
add_subdirectory(stream)

# A memory benchmark
add_subdirectory(mGBPS)

