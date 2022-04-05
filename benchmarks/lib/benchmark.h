#ifndef __BENCHMARK_TOOLS
#define __BENCHMARK_TOOLS

#include <stdint.h>
#include <time.h>

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

long double tick_tock(struct timespec *tinfo);
Result benchmark(Result (*kernel_func)(KernelArgs args), KernelArgs args, long double duration, char *name);

Result n_benchmark(Result (*kernel)(KernelArgs args), KernelArgs args, int iters, char *name);
#endif