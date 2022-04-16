#ifndef __BENCHMARK_TOOLS
#define __BENCHMARK_TOOLS

#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>

/**
 * @member 	result - The result of the FLOPS computed by the kernel
 * @member	flop_ct - The total number of FLOPS computed by the kernel
 * @member 	mem_accesses - The total number of memory accesses made by the kernel
 */
typedef struct Result{
	double result;
	uint64_t flop_ct;
	uint64_t mem_accesses;
} Result;

/**
 * @member 	iters - The total number of loop iterations to be done
 * @member 	input - Any input data that is to be sent to the kernel, can be casted to anything
 */
typedef struct KernelArgs{
	uint32_t iters;
	void *input;
} KernelArgs;

void output_hwinfo();
void pretty_print(long double duration, Result ret);
void compressed_pretty_print(long double duration, Result ret);

long double tick_tock(struct timespec *tinfo);

void benchmark(Result (*kernel_func)(KernelArgs args), KernelArgs args, long double duration, char *name, bool parallel);

void mem_flush(const void *p, unsigned int allocation_size);
void fill_cache(const char *p, unsigned int allocation_size);


char *get_filepath(const char *path, const char *fname);

bool fverify_benchmark(float *result, int n, int m, const char *dir, const char *bench);
bool dverify_benchmark(double *result, int n, int m, const char *dir, const char *bench);

float* get_farg(FILE *fptr, int *_n, int *_m);
double* get_darg(FILE *fptr, int *_n, int *_m);
/**
 * When the functions being benchmarked can't be conformed to the required 
 * function signature this is a hacky way to still run the same benchmark on 
 * them. 
 *
 * Arguments:
 * FRUN - The number of FLOPS performed per function call
 * MRUN - The number of memory accesses per function call
 * DUR  - The minimum duration to run the benchmark for
 * NAME - The name of the benchmark
 *
 * Usage:
 * BENCH_START(FRUN, MRUN, DUR, NAME) 
 *     reset_vars(...)
 *     START_RECORD
 *         fun(arg1, arg2. ...) 
 *     END_RECORD
 * BENCH_END
 */
#define BENCH_START(FRUN, MRUN, DUR, NAME) { struct timespec *tinfo = malloc(sizeof(struct timespec)); \
        for(int i=0; i<60; i++) printf("-"); \
        printf("\nBenchmark - %s\n", NAME); \
        Result ret = {0.0, 0, 0}; \
        long double min_duration = DUR; \
        long double runtime = 0, st, en; \
        do { \
            ret.flop_ct += FRUN; \
            ret.mem_accesses += MRUN;

#define START_RECORD st = tick_tock(tinfo);

#define END_RECORD en = tick_tock(tinfo); \

#define BENCH_END runtime += en - st; \
        } while(runtime < min_duration); \
        compressed_pretty_print(runtime, ret); \
        free(tinfo); }

#endif