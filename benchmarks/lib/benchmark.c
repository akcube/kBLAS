#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <benchmark.h>

/**
 * Outputs details about the precision of the clocks on the system and other useful hardware spec 
 * information
 */
void output_hwinfo(){
	struct timespec *info = malloc(sizeof(struct timespec));
	
	for(int i=0; i<60; i++) printf("-");
	puts("\nHardware timer resolution information: ");
	for(int i=0; i<60; i++) printf("-");
	
	clock_getres(CLOCK_REALTIME, info);
	printf("\nReal time clock: \t\t%.9Lf sec\n", (long double) info->tv_sec + info->tv_nsec * 1e-9);
	clock_getres(CLOCK_MONOTONIC, info);
	printf("Monotonic clock: \t\t%.9Lf sec\n", (long double) info->tv_sec + info->tv_nsec * 1e-9);
	clock_getres(CLOCK_PROCESS_CPUTIME_ID, info);
	printf("Per-process CPU clock: \t\t%.9Lf sec\n", (long double) info->tv_sec + info->tv_nsec * 1e-9);
	clock_getres(CLOCK_THREAD_CPUTIME_ID, info);
	printf("Per-thread CPU clock: \t\t%.9Lf sec\n\n", (long double) info->tv_sec + info->tv_nsec * 1e-9);
}

/**
 * Returns the current CPU clock time in seconds
 * @param  tinfo - timespec struct pointing to alloc'd memory
 */
long double tick_tock(struct timespec *tinfo){
	clock_gettime(CLOCK_MONOTONIC, tinfo);
	return (long double) tinfo->tv_sec + tinfo->tv_nsec * 1e-9;
}

/**
 * Pretty prints the result returned by a benchmark
 * @param st - The start time of the benchmark
 * @param en - The end time of the benchmark
 * @param ret - The result obtained by the benchmark over all runs
 */
void pretty_print(long double duration, Result ret){
	printf("\nTotal runtime:\t\t\t%Lf\n", duration);
	printf("Result computed:\t\t%lf\n", ret.result);
	printf("Total FLOPS computed:\t\t%lu\n", ret.flop_ct);
	printf("Total memory accesses:\t\t%lu\n", ret.mem_accesses);
	printf("\nGFLOPS:\t\t\t\t%Lf\n", (long double) ret.flop_ct / duration * 1e-9);
	printf("Bandwidth:\t\t\t%Lf GB/s\n\n", (long double) ret.mem_accesses / duration * 1e-9);
}

/**
 * Pretty prints only the GFLOPS & Bandwidth info returned by a benchmark
 * @param st - The start time of the benchmark
 * @param en - The end time of the benchmark
 * @param ret - The result obtained by the benchmark over all runs
 */
void compressed_pretty_print(long double duration, Result ret){
	printf("\nGFLOPS:\t\t\t\t%Lf\n", (long double) ret.flop_ct / duration * 1e-9);
	printf("Bandwidth:\t\t\t%Lf GB/s\n", (long double) ret.mem_accesses / duration * 1e-9);
}

/**
 * Given some kernel which takes in `args` as params, it runs the kernel repeatedly until a minimum of 
 * `duration` seconds has passed and outputs the GFLOPS/sec achieved by the kernel and other bench info.
 * This is first done on a single thread, then run again over all threads. Information for both tests 
 * is included in the final output.
 * @param 	kernel - A kernel which takes an argument of type 'KernelArgs'
 * @param 	args - The arguments to be passed to the kernel
 * @param 	duration - The minimum amount of time to run the kernel for
 * @param 	name - The name / header to display when outputting benchmark details
 * @param 	parallel - Run the benchmark parallelized over all threads
 */
void benchmark(Result (*kernel)(KernelArgs args), KernelArgs args, long double duration, char *name, bool parallel){
	// Single thread run
	{
		struct timespec *tinfo = malloc(sizeof(struct timespec));
		Result ret = {0.0, 0, 0};

		long double st = tick_tock(tinfo);
		long double min_duration = st + duration, en;

		do {
			Result t = kernel(args);
			ret.result += t.result;
			ret.flop_ct += t.flop_ct;
			ret.mem_accesses += t.mem_accesses;
		} while((en = tick_tock(tinfo)) < min_duration);

		for(int i=0; i<60; i++) printf("-");
		printf("\nSingle-thread:\tBenchmark information - %s\n", name);
		for(int i=0; i<60; i++) printf("-");

		pretty_print(en - st, ret);
	}
	// Multi-thread run
	if(parallel)
	{
		struct timespec *tinfo = malloc(sizeof(struct timespec));
		Result ret = {0.0, 0, 0};

		long double st = tick_tock(tinfo);
		long double min_duration = st + duration, en;

		#pragma omp parallel
		do {
			Result t = kernel(args);
			ret.result += t.result;
			ret.flop_ct += t.flop_ct;
			ret.mem_accesses += t.mem_accesses;
		} while((en = tick_tock(tinfo)) < min_duration);

		en = tick_tock(tinfo);		
		for(int i=0; i<60; i++) printf("-");
		printf("\nMulti-thread:\tBenchmark information - %s\n", name);
		for(int i=0; i<60; i++) printf("-");

		pretty_print(en - st, ret);
	}
}
