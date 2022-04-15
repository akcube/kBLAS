#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <benchmark.h>

/**
 * Returns the current CPU clock time in seconds
 * @param  tinfo - timespec struct pointing to alloc'd memory
 */
long double tick_tock(struct timespec *tinfo){
	clock_gettime(CLOCK_MONOTONIC, tinfo);
	return (long double) tinfo->tv_sec + tinfo->tv_nsec * 1e-9;
}

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

/** Intel recommended method to flush the cache */
void mem_flush(const void *p, unsigned int allocation_size){
    const size_t cache_line = 64;
    const char *cp = (const char *)p;
    size_t i = 0;
    if (p == NULL || allocation_size <= 0)
            return;
    for (i = 0; i < allocation_size; i += cache_line) {
            asm volatile("clflush (%0)\n\t" : : "r"(&cp[i]) : "memory");
    }
    asm volatile("sfence\n\t" : : : "memory"); // Really cool instruction
}

/**
 * Concatenate the directory path with filename to generate full 
 * path string.
 * @param  path  Directory path
 * @param  fname File name
 * @return       Pointer to string containing full path string
 */
char *get_filepath(const char *path, const char *fname){
	int plen = strlen(path);
	int flen = strlen(fname);
	char *file_path = malloc(sizeof(char) * (plen + flen + 1));
	memcpy(file_path, path, sizeof(char) * (plen + 1));
	strcat(file_path, fname);
	return file_path;
}

/**
 * Parse the input file to get one matrix argument and move fptr to
 * the next argument if any. _n and _m will contain the dimensions of 
 * the argument read if _n and _m are not NULL.
 * 
 * Expected input file format:
 * [int n, int m, contiguous allocation of n*m single-precision numbers]
 */
float* get_farg(FILE *fptr, int *_n, int *_m){
	int n, m, read;
	read = fread(&n, sizeof(int), 1, fptr);
	
	if(!read) return NULL;

	fread(&m, sizeof(int), 1, fptr);
	float *data = malloc(sizeof(float) * n * m);
	fread(data, sizeof(float), n * m, fptr);
	if(_n) *_n = n;
	if(_m) *_m = m;
	return data;
}

/**
 * Parse the input file to get one matrix argument and move fptr to
 * the next argument if any. _n and _m will contain the dimensions of 
 * the argument read if _n and _m are not NULL.
 * 
 * Expected input file format:
 * [int n, int m, contiguous allocation of n*m double-precision numbers]
 */
double* get_darg(FILE *fptr, int *_n, int *_m){
	int n, m, read;
	read = fread(&n, sizeof(int), 1, fptr);
	
	if(!read) return NULL;

	fread(&m, sizeof(int), 1, fptr);
	double *data = malloc(sizeof(double) * n * m);
	fread(data, sizeof(double), n * m, fptr);
	if(_n) *_n = n;
	if(_m) *_m = m;
	return data;
}

// TODO: Change to epsilon based comparison

/**
 * Single precision version
 * Given the resultant of some computation and the file containing the data to be 
 * verified against, check if the result matches with check data. 
 * @param  result Data to check
 * @param  n, m   Dimensions of data element
 * @param  dir    Directory containing verification data
 * @param  bench  Name of benchmark
 * @return        True if verified else false
 */
bool fverify_benchmark(float *result, int n, int m, const char *dir, const char *bench){
	char *filepath = get_filepath(dir, bench);
	FILE *fptr = fopen(filepath, "rb");
	free(filepath);

	float *check = malloc(sizeof(float) * n * m);
	int read = fread(check, sizeof(float), n*m, fptr);

	if(read != n * m) return false;
	return (memcmp(result, check, n*m) == 0);
}

/**
 * Double precision version
 * Given the resultant of some computation and the file containing the data to be 
 * verified against, check if the result matches with check data. 
 * @param  result Data to check
 * @param  n, m   Dimensions of data element
 * @param  dir    Directory containing verification data
 * @param  bench  Name of benchmark
 * @return        True if verified else false
 */
bool dverify_benchmark(double *result, int n, int m, const char *dir, const char *bench){
	char *filepath = get_filepath(dir, bench);
	FILE *fptr = fopen(filepath, "rb");
	free(filepath);

	double *check = malloc(sizeof(double) * n * m);
	int read = fread(check, sizeof(double), n*m, fptr);

	if(read != n * m) return false;
	return (memcmp(result, check, n*m) == 0);
}