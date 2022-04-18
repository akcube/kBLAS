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

const char *input_dir = "input/sdot/";
const char *verif_dir = "verification/sdot/";

static void escape(void *p){
	asm volatile("" : : "g"(p) : "memory");
}

int main(int argc, char *argv[]){

	int num_files = 0; 
	char** names = get_files(input_dir, &num_files);

	long long min_mem, max_mem;
	arg_parse(argc, argv, &min_mem, &max_mem);

	/**
	 * SDOT Benchmark
	 * Params:		N, X*, INCX, Y*, INCY
	 * Operation: 	Return t = \sum X[i*incX] * Y[i*incY]
	 */
	int N;
	for(int i=0; i<num_files; i++){
		// Get test-data file
		char *filepath = get_filepath(input_dir, names[i]);
		FILE *file = fopen(filepath, "rb");
		char namebuf[1024];

		// Parse args
		float *X = get_farg(file, NULL, &N);
		float *Y = get_farg(file, NULL, NULL);

		float memory = ((float) N*sizeof(float));
		if(memory <= min_mem || memory >= max_mem) goto cleanup;
		memory /= (1024 * 1024);

		fill_cache((const char*) X, N * sizeof(float));
		fill_cache((const char*) Y, N * sizeof(float));
		float dotres = -1;

		// Run CBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "CBLAS: sdot 1 x %d, Memory: %f MB", N, memory);
			dotres = -1;
			BENCH_START(N, sizeof(float)*N, 1, namebuf)
			 		START_RECORD
			 			
			 			dotres = cblas_sdot(N, X, 1, Y, 1);
			 			
			 		END_RECORD
			BENCH_END
		}
		// Run KBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "KBLAS: sdot 1 x %d, Memory: %f MB", N, memory);
			dotres = -1;
			BENCH_START(N, sizeof(float)*N, 1, namebuf)
		 		START_RECORD
		 			
			 			dotres = kblas_sdot(N, X, 1, Y, 1);
		 			
		 		END_RECORD
			BENCH_END
			printf("Verified:\t\t\t");
			if(fverify_benchmark(&dotres, 1, 1, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// Run BLIS  benchmark ----------------------------------------------------
		{
			obj_t X_b, Y_b, rho_b;
			bli_obj_create_1x1(BLIS_FLOAT, &rho_b);
			bli_setsc(-1, 0.0, &rho_b);
			bli_obj_create_with_attached_buffer(BLIS_FLOAT, 1, N, X, N, 1, &X_b);
			bli_obj_create_with_attached_buffer(BLIS_FLOAT, 1, N, Y, N, 1, &Y_b);
			sprintf(namebuf, "BLIS : sdot 1 x %d, Memory: %f MB", N, memory);
			BENCH_START(N, sizeof(float)*N, 1, namebuf)
		 		START_RECORD
		 			
		 			bli_dotv(&X_b, &Y_b, &rho_b);
		 			
		 		END_RECORD
			BENCH_END

			bli_obj_free(&rho_b);
			printf("Verified:\t\t\t");
			if(fverify_benchmark(rho_b.buffer, 1, 1, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// ------------------------------------------------------------------------
		// Cleanup
		cleanup: 
		free(filepath);
		fclose(file);
		free(X);
		free(Y);
	}

	for(int i=0; i<num_files; i++)
		free(names[i]);
	free(names);
}