#include <blis.h>
#include <cblas.h>
#include <kblas.h>
#include <benchmark.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "helper.h"
#include <immintrin.h>

const char *input_dir = "input/daxpy/";
const char *verif_dir = "verification/daxpy/";

static void escape(void *p){
	asm volatile("" : : "g"(p) : "memory");
}

int main(int argc, char *argv[]){

	int num_files = 0; 
	char** names = get_files(input_dir, &num_files);

	long long min_mem, max_mem;
	arg_parse(argc, argv, &min_mem, &max_mem);

	/**
	 * daxpy Benchmark
	 * Params:		N, Alpha, X*, INCX, Y*, INCY
	 * Operation: 	Y = \alpha * X + Y
	 */
	int N;
	for(int i=0; i<num_files; i++){
		// Get test-data file
		char *filepath = get_filepath(input_dir, names[i]);
		FILE *file = fopen(filepath, "rb");
		char namebuf[1024];

		// Parse args
		double *alpha = get_darg(file, NULL, NULL);
		double *X = get_darg(file, NULL, &N);
		double *Y = get_darg(file, NULL, &N);
		double *ycpy = memdup(Y, N * sizeof(double));

		double memory = ((double) 2*N*sizeof(double));
		if(memory <= min_mem || memory >= max_mem) goto cleanup;
		memory /= (1024 * 1024);

		// Run CBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "CBLAS: daxpy 1 x %d, Memory: %f MB", N, memory);
			BENCH_START(2*N, 2*sizeof(double)*N, 1, namebuf)
				memcpy(Y, ycpy, N * sizeof(double));
				mem_flush(Y, N*sizeof(double));
				fill_cache((const char*) X, N * sizeof(double));
				fill_cache((const char*) Y, N * sizeof(double));
			 		START_RECORD
			 			
			 			cblas_daxpy(N, *alpha, X, 1, Y, 1);
			 			
			 		END_RECORD
			BENCH_END
		}
		// Run KBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "KBLAS: daxpy 1 x %d, Memory: %f MB", N, memory);
			BENCH_START(2*N, 2*sizeof(double)*N, 1, namebuf)
				memcpy(Y, ycpy, N * sizeof(double));
				mem_flush(Y, N*sizeof(double));
				fill_cache((const char*) X, N * sizeof(double));
				fill_cache((const char*) Y, N * sizeof(double));
		 		START_RECORD
		 			
		 			kblas_daxpy(N, *alpha, X, 1, Y, 1);
		 			
		 		END_RECORD
			BENCH_END
			printf("Verified:\t\t\t");
			if(dverify_benchmark(Y, 1, N, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// Run BLIS  benchmark ----------------------------------------------------
		{
			obj_t X_b, alpha_b, Y_b;
			bli_obj_create_with_attached_buffer(BLIS_DOUBLE, 1, N, Y, N, 1, &Y_b);
			bli_obj_create_with_attached_buffer(BLIS_DOUBLE, 1, N, X, N, 1, &X_b);
			bli_obj_create_1x1(BLIS_DOUBLE, &alpha_b);
			bli_setsc(*alpha, 0.0, &alpha_b);
			sprintf(namebuf, "BLIS : daxpy 1 x %d, Memory: %f MB", N, memory);
			BENCH_START(2*N, 2*sizeof(double)*N, 1, namebuf)
				memcpy(Y, ycpy, N * sizeof(double));
				mem_flush(Y, N*sizeof(double));
				fill_cache((const char*) X, N * sizeof(double));
				fill_cache((const char*) Y, N * sizeof(double));
		 		START_RECORD
		 			
		 			bli_scalv(&alpha_b, &X_b);
		 			
		 		END_RECORD
			BENCH_END

			bli_obj_free(&alpha_b);
			printf("Verified:\t\t\t");
			if(dverify_benchmark(Y, 1, N, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// ------------------------------------------------------------------------
		// Cleanup
		cleanup: 
		free(filepath);
		fclose(file);
		free(alpha);
		free(X);
		free(ycpy);
	}

	for(int i=0; i<num_files; i++)
		free(names[i]);
	free(names);
}