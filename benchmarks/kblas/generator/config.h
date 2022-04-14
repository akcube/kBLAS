#ifndef __KBLAS_BENCHGEN_CONFIG
#define __KBLAS_BENCHGEN_CONFIG

/**
 * The goal here is to benchmark the functions for varying sizes in memory. Ultimately
 * memory will end up becoming the benchmark for large vector / matrix operations.
 * Here is a reasonable guide to picking config sizes: 
 * + 1-3 options should fit easily in L1 cache
 * + 4-6 options should fit in L2-L3 cache
 * + 7-9 options should overflow cache and start occupying large amounts of system memory
 * 
 * Tip: For the double versions of benchmarks, simply half the memory usage of the float versions
 */

#define INPUT_DIR "../input/"
#define VERIF_DIR "../verification/"

#endif