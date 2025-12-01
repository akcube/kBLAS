# kBLAS

This is a pet project where I try my hand at implementing some of the functions as specified by the BLAS standard. It's more a learning exercise than anything else so I'll be toying with several architecture specific optimizations and write the SIMD intrinsics myself to get the job done. 

I document my entire journey working on this project here: [Optimizing kBLAS](https://akcube.github.io/blog/mega-project-kblas-writing-a-benchmark-library-in-c-optimizing-l1-l2-basic-linear-algebra-subprograms/)

As of now, these are the functions I plan on implementing. As the project matures and I learn more I'll consider expanding to and implementing more functions. 

## Level 1

- [x] sscalv, dscalv
- [x] sdot, ddot
- [x] saxpy, daxpy

## Level 2

- [ ] sgemv, dgemv

## Level 3

- [ ] sgemm, dgemm

## Benchmark information

A few things to keep in mind when reading any results I post. 

1. All benchmarks were run with BLIS configured using `-t OpenMP` and with the environment variable `$BLIS_NUM_THREADS=16`. 

2. All my vectors / matrix data are aligned to 64 byte boundaries by default. In practice, I would have to write a small scalar loop to perform computations on the beginning portion of a vector till I hit a 64 byte boundary separately. To save myself the hassle, I'm currently only benchmarking on 64 byte aligned vector and matrices. 

3. I attempt to populate the entire cache with the input vectors before calling the functions. Perhaps later I will consider benchmarking both with cold cache and hot cache. However, I think hot cache benchmarking is good enough for now. Since I benchmark over a large range of vector sizes (From ~40kb to 1GB) we see the effects of the memory wall on the larger data. For smaller data it is reasonable to assume it is usually in cache and helps us better compare peak performance of two functions in my opinion.

4. The benchmark is designed to run like this:
```
 BENCH_START(FRUN, MRUN, DUR, NAME) 
     reset_vars(...) // This includes flushing unnecessary variables from cache and populating it with input data
     START_RECORD
         fun(arg1, arg2. ...) 
     END_RECORD
 BENCH_END
```
This construct is run as many times as necessary until `DUR` seconds has been reached (by the accumulative sum of the recording region). This way we can dynamically run a benchmark `x` number of times until we are reasonably sure about accuracy. For test purposes, I have set this value to 2 seconds.

## Roofline analysis

For conducting roofline analysis you can check the [tuned version](/roofline-analysis/stream/stream.c) of the [Stream](https://www.cs.virginia.edu/stream/) benchmark that I wrote which beat the results I got from the auto-vectorized default. After this, I wrote my own memory benchmark [mGBPS](/roofline-analysis/mGBPS/mgbps.c), which was able to record my highest results so far on this machine. Results can be found [here](/results/stream/) and [here](/results/mGBPS/).

## Benchmark results

I'm only posting the graphed versions of the results I obtained here. For more precise information check the [results directory](/results/BLAS/) or better yet, build and run the benchmarks yourselves :)

### Level 1

1. `sscal` and `dscal`

![sscal_benchmark](/assets/sscal.png)

![dscal_benchmark](/assets/dscal.png)

2. `ssdot` and `ddot`

![sdot_benchmark](/assets/sdot.png)

![ddot_benchmark](/assets/ddot.png)

3. `saxpy` and `daxpy`

![saxpy_benchmark](/assets/saxpy.png)

![daxpy_benchmark](/assets/daxpy.png)

### Level 2

TODO: ...

