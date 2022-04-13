#ifndef __BENCHMARK_TOOLS
#define __BENCHMARK_TOOLS

#include <stdint.h>
#include <time.h>
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
 * BENCH_START(FRUN, MRUN, DUR, NAME) fun(arg1, arg2. ...) BENCH_END
 */
#define BENCH_START(FRUN, MRUN, DUR, NAME) { struct timespec *tinfo = malloc(sizeof(struct timespec)); \
        Result ret = {0.0, 0, 0}; \
        long double st = tick_tock(tinfo); \
        long double min_duration = st + DUR, en; \
        for(int i=0; i<60; i++) printf("-"); \
        printf("\nBenchmark - %s\n", NAME); \
        for(int i=0; i<60; i++) printf("-"); \
        do { \
            ret.flop_ct += FRUN; \
            ret.mem_accesses += MRUN;
#define BENCH_END } while((en = tick_tock(tinfo)) < min_duration); \
        en = tick_tock(tinfo); \
        compressed_pretty_print(en - st, ret); }

#endif