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

const char *input_dir = "input/sscal/";
const char *verif_dir = "verification/sscal/";

static void escape(void *p){
	asm volatile("" : : "g"(p) : "memory");
}

int main(int argc, char *argv[]){

	int num_files = 0; 
	char** names = get_files(input_dir, &num_files);

	/**
	 * SSCAL Benchmark
	 * Params:		N, Alpha, X*, INCX
	 * Operation: 	X = \alpha * X
	 */
	int N;
	float *data;
	for(int i=0; i<num_files; i++){
		// Get test-data file
		char *filepath = get_filepath(input_dir, names[i]);
		FILE *file = fopen(filepath, "rb");
		char namebuf[1024];

		// Parse args
		float *alpha = get_farg(file, NULL, NULL);
		float *X = get_farg(file, NULL, &N);
		float *xcpy = memdup(X, N * sizeof(float));

		float memory = ((float) N*sizeof(float))/(1024 * 1024);

		// Run CBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "CBLAS: sscal 1 x %d, Memory: %f MB", N, memory);
			BENCH_START(N, sizeof(float)*N, 1, namebuf);
				memcpy(X, xcpy, N * sizeof(float));
				fill_cache((const char*) X, N * sizeof(float));
			 		START_RECORD
			 			
			 			cblas_sscal(N, *alpha, X, 1);
			 			
			 		END_RECORD
			BENCH_END
		}
		// Run KBLAS benchmark ----------------------------------------------------
		{
			sprintf(namebuf, "KBLAS: sscal 1 x %d, Memory: %f MB", N, memory);
			BENCH_START(N, sizeof(float)*N, 1, namebuf)
				memcpy(X, xcpy, N * sizeof(float));
				fill_cache((const char*) X, N * sizeof(float));
		 		START_RECORD
		 			
		 			kblas_sscal(N, *alpha, X, 1);
		 			
		 		END_RECORD
			BENCH_END
			printf("Verified:\t\t\t");
			if(fverify_benchmark(X, 1, N, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// Run BLIS  benchmark ----------------------------------------------------
		{
			obj_t X_b, alpha_b;
			bli_obj_create(BLIS_FLOAT, 1, N, 0, 0, &X_b);
			bli_obj_create_1x1(BLIS_FLOAT, &alpha_b);
			bli_setsc(*alpha, 0.0, &alpha_b);
			sprintf(namebuf, "BLIS : sscal 1 x %d, Memory: %f MB", N, memory);
			BENCH_START(N, sizeof(float)*N, 1, namebuf)
				bli_randv(&X_b);
		 		START_RECORD
		 			
		 			bli_scalv(&alpha_b, &X_b);
		 			
		 		END_RECORD
			BENCH_END
		}
		// ------------------------------------------------------------------------
		// Cleanup
		free(filepath);
		fclose(file);
		free(alpha);
		free(X);
		free(xcpy);
	}
}