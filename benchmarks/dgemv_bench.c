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

const char *input_dir = "input/dgemv/";
const char *verif_dir = "verification/dgemv/";

static void escape(void *p){
	asm volatile("" : : "g"(p) : "memory");
}

int main(int argc, char *argv[]){

	int num_files = 0; 
	char** names = get_files(input_dir, &num_files);

	long long min_mem, max_mem;
	arg_parse(argc, argv, &min_mem, &max_mem);

	/**
	 * dgemv Benchmark
	 * Params:		KBLAS_ORDER, KBLAS_ORDER, M, N, alpha, A, X, beta, Y
	 * Operation: 	Y = \alpha * A * x + \beta * y
	 */
	int M, N;
	for(int i=0; i<num_files; i++){
		// Get test-data file
		char *filepath = get_filepath(input_dir, names[i]);
		FILE *file = fopen(filepath, "rb");
		char namebuf[1024];

		// Parse args
		double *alpha = get_darg(file, NULL, NULL);
		double *A = get_darg(file, &M, &N);
		double *x = get_darg(file, NULL, NULL);
		double *beta = get_darg(file, NULL, NULL);
		double *y = get_darg(file, NULL, NULL);
		double *ycpy = memdup(y, M * sizeof(double));

		double memory = ((double) (M * N + N + M)*sizeof(double));
		if(memory <= min_mem || memory >= max_mem) goto cleanup;
		memory /= (1024 * 1024);

		// Run CBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "CBLAS: dgemv %d x %d, Memory: %f MB", M, N, memory);
			BENCH_START(M*N + M * (N + N - 1) + 2*M, (M * N + N + M) * sizeof(double), 1, namebuf)
				memcpy(y, ycpy, M * sizeof(double));
				mem_flush(ycpy, M*sizeof(double));
				fill_cache((const char*) A, N * M * sizeof(double));
				fill_cache((const char*) x, N * sizeof(double));
				fill_cache((const char*) y, M * sizeof(double));
			 	START_RECORD
			 		
			 		cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, *alpha, A, N, x, 1, *beta, y, 1);
			 		
			 	END_RECORD
			BENCH_END
		}
		// Run KBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "KBLAS: dgemv %d x %d, Memory: %f MB", M, N, memory);
			BENCH_START(M*N + M * (N + N - 1) + 2*M, (M * N + N + M) * sizeof(double), 1, namebuf)
				memcpy(y, ycpy, M * sizeof(double));
				mem_flush(ycpy, M*sizeof(double));
				fill_cache((const char*) A, N * M * sizeof(double));
				fill_cache((const char*) x, N * sizeof(double));
				fill_cache((const char*) y, M * sizeof(double));
		 		START_RECORD
		 			
		 			kblas_dgemv(KblasRowMajor, KblasNoTrans, M, N, *alpha, A, N, x, 1, *beta, y, 1);
		 			
		 		END_RECORD
			BENCH_END
			printf("Verified:\t\t\t");
			if(dverify_benchmark(y, M, 1, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// Run BLIS  benchmark ----------------------------------------------------
		{
			obj_t x_b, alpha_b, y_b, A_b, beta_b;
			bli_obj_create_with_attached_buffer(BLIS_DOUBLE, M, N, A, N, 1, &A_b);
			bli_obj_create_with_attached_buffer(BLIS_DOUBLE, N, 1, x, 1, 1, &x_b);
			bli_obj_create_with_attached_buffer(BLIS_DOUBLE, M, 1, y, 1, 1, &y_b);
			bli_obj_create_1x1(BLIS_DOUBLE, &alpha_b);
			bli_obj_create_1x1(BLIS_DOUBLE, &beta_b);
			bli_setsc(*alpha, 0.0, &alpha_b);
			bli_setsc(*beta, 0.0, &beta_b);
			sprintf(namebuf, "BLIS : dgemv %d x %d, Memory: %f MB", M, N, memory);
			BENCH_START(M*N + M * (N + N - 1) + 2*M, (M * N + N + M) * sizeof(double), 1, namebuf)
				memcpy(y, ycpy, M * sizeof(double));
				mem_flush(ycpy, M*sizeof(double));
				fill_cache((const char*) A, N * M * sizeof(double));
				fill_cache((const char*) x, N * sizeof(double));
				fill_cache((const char*) y, M * sizeof(double));
		 		START_RECORD
		 			
		 			bli_gemv(&alpha_b, &A_b, &x_b, &beta_b, &y_b);
		 			
		 		END_RECORD
			BENCH_END

			bli_obj_free(&alpha_b);
			printf("Verified:\t\t\t");
			if(dverify_benchmark(y, M, 1, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// ------------------------------------------------------------------------
		// Cleanup
		cleanup: 
		free(filepath);
		fclose(file);
		free(alpha);
		free(beta);
		free(x);
		free(y);
		free(A);
		free(ycpy);
	}

	for(int i=0; i<num_files; i++)
		free(names[i]);
	free(names);
}