/**
 * This is a memory benchmark that attempts to squeeze every last bit of bandwidth it can out of 
 * your CPU. This was made targetting CPUs with mavx2 instruction set extensions. It will not 
 * run if your CPU does not support mavx2. 
 * I use the compiler tricks described in this (https://www.youtube.com/watch?v=nXaxk27zwlk&t=2398s)
 * talk by Chandler Carruth to inform the compiler that the reads I make to the read_buf array
 * might change the state (The benchmark) of the program without the knowledge of the compiler. This
 * allows it to escape the compiler's Dead Code Elimination (DCE) pass. 
 * I recommend compiling with -DNUM_THREADS=x where x is the number of memory channels you have 
 * available.
 * It is distributed under the MIT License (LICENSE.txt)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <benchmark.h>
#include <immintrin.h>
#include <omp.h>

#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b)

#define ARR_SIZE 80000000
#define NUM_THREADS 2

// Array to be read from. Aligned to 32-bit boundary to prevent GP when using avx instructions
static int read_buf[ARR_SIZE] __attribute__ ((aligned (32)));

Result read_test(KernelArgs args);

// Functions to "escape" compiler DCE passes ===================================================
static void escape(void *p){
	asm volatile("" : : "g"(p) : "memory");
}

static void clobber(){
	asm volatile("" : : : "memory");
}
// =============================================================================================

int main(void){
	output_hwinfo();
	srand(time(0));

	// Initialize array to random data
	for(int i=0; i<ARR_SIZE; i++) read_buf[i] = rand();	

	// Run benchmark
	KernelArgs args = {ARR_SIZE, (void*) read_buf};
	benchmark(read_test, args, 5, "Streamed read", false);
}

/**
 * The benchmark. 16x loop unrolled avx2 instructions to read data from memory as fast
 * as possible.
 * @param  args - KernelArgs object which contains ptr to array and size of array.
 * @return 		Total number of memory accesses
 */
Result read_test(KernelArgs args){
	int n = args.iters;
	int *read = args.input;

#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

// Set NUM_THREADS to number of available memory channels
#pragma omp parallel for num_threads(NUM_THREADS)
	for(int i=0; i < n-128; i+=128){
		// 16x loop unrolling to help compiler register rename to all effective
		// 16 SIMD registers.
		__m256i r1 = _mm256_stream_load_si256((__m256i*) &read[i]);
		__m256i r2 = _mm256_stream_load_si256((__m256i*) &read[i+8]);
		__m256i r3 = _mm256_stream_load_si256((__m256i*) &read[i+16]);
		__m256i r4 = _mm256_stream_load_si256((__m256i*) &read[i+24]);
		__m256i r5 = _mm256_stream_load_si256((__m256i*) &read[i+32]);
		__m256i r6 = _mm256_stream_load_si256((__m256i*) &read[i+40]);
		__m256i r7 = _mm256_stream_load_si256((__m256i*) &read[i+48]);
		__m256i r8 = _mm256_stream_load_si256((__m256i*) &read[i+56]);
		__m256i r9 = _mm256_stream_load_si256((__m256i*) &read[i+64]);
		__m256i r10 = _mm256_stream_load_si256((__m256i*) &read[i+72]);
		__m256i r11 = _mm256_stream_load_si256((__m256i*) &read[i+80]);
		__m256i r12 = _mm256_stream_load_si256((__m256i*) &read[i+88]);
		__m256i r13 = _mm256_stream_load_si256((__m256i*) &read[i+96]);
		__m256i r14 = _mm256_stream_load_si256((__m256i*) &read[i+104]);
		__m256i r15 = _mm256_stream_load_si256((__m256i*) &read[i+112]);
		__m256i r16 = _mm256_stream_load_si256((__m256i*) &read[i+120]);

		// Escape DCE pass
		escape(&r1);
		escape(&r2);
		escape(&r3);
		escape(&r4);
		escape(&r5);
		escape(&r6);
		escape(&r7);
		escape(&r8);
		escape(&r9);
		escape(&r10);
		escape(&r11);
		escape(&r12);
		escape(&r13);
		escape(&r14);
		escape(&r15);
		escape(&r16);
	}

	// Cleanup
	#pragma omp parallel for num_threads(NUM_THREADS)
	for(int i=max(0, n-128); i<n; i++) {
		int t = read[i];
		escape(&t);
	}

	Result retval = {0, 0, args.iters * sizeof(int)};
	return retval;
}
