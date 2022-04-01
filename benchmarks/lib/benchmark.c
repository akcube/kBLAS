#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "benchmark.h"

/**
 *	struct timespec {
		time_t   tv_sec;        // seconds
		long     tv_nsec;       // nanoseconds
 	}; 

 	Useful Clock IDs:
 	- CLOCK_REALTIME :	A settable system-wide clock that measures real wall- clock time.  
 						This clock is affected by discontinuous jumps in the system time, 
 						and by the incremental adjustments performed by adjtime(3) and NTP.

 	- CLOCK_MONOTONIC : A system-wide clock that represents monotonic time some unspecified 
 						point in the past. On Linux, that point corresponds to the number of 
 						seconds that the system has been running since it was booted. This clock 
 						is affected by the incremental adjustments performed by adjtime(3) and NTP.

 	- CLOCK_MONOTONIC_RAW :	Similar to CLOCK_MONOTONIC, but provides access to a raw hardware-based 
 							time that is not subject to NTP adjustments	or the incremental adjustments 
 							performed by adjtime(3).

 	- CLOCK_PROCESS_CPUTIME_ID : This is a clock that measures CPU time consumed by this process
 	- CLOCK_THREAD_CPUTIME_ID : This is a clock that measures CPU time consumed by this thread.
 */

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
 * Given some kernel which takes in `args` as params, it runs the kernel repeatedly until a minimum of 
 * `duration` seconds has passed and outputs the GFLOPS/sec achieved by the kernel and other bench info.
 * This is first done on a single thread, then run again over all threads. Information for both tests 
 * is included in the final output.
 * @param 	kernel - A kernel which takes an argument of type 'KernelArgs'
 * @param 	args - The arguments to be passed to the kernel
 * @param 	duration - The minimum amount of time to run the kernel for
 * @param 	name - The name / header to display when outputting benchmark details
 */
Result benchmark(Result (*kernel)(KernelArgs args), KernelArgs args, long double duration, char *name){
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

		printf("\nTotal runtime:\t\t\t%Lf\n", en - st);
		printf("Result computed:\t\t%lf\n", ret.result);
		printf("Total FLOPS computed:\t\t%lu\n", ret.flop_ct);
		printf("Total memory accesses:\t\t%lu\n", ret.mem_accesses);
		printf("\nGFLOPS:\t\t\t\t%Lf\n\n", (long double) ret.flop_ct/(en - st) * 1e-9);
	}
	// Multi-thread run
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

		for(int i=0; i<60; i++) printf("-");
		printf("\nMulti-thread:\tBenchmark information - %s\n", name);
		for(int i=0; i<60; i++) printf("-");

		printf("\nTotal runtime:\t\t\t%Lf\n", en - st);
		printf("Result computed:\t\t%lf\n", ret.result);
		printf("Total FLOPS computed:\t\t%lu\n", ret.flop_ct);
		printf("Total memory accesses:\t\t%lu\n", ret.mem_accesses);
		printf("\nGFLOPS:\t\t\t\t%Lf\n\n", (long double) ret.flop_ct/(en - st) * 1e-9);
	}
}