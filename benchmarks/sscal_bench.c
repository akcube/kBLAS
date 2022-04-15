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

int main(void){

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
		
		// Parse args
		float *alpha = get_farg(file, NULL, NULL);
		float *X = get_farg(file, NULL, &N);
		float *xcpy = memdup(X, N * sizeof(float));
		// Run CBLAS benchmark
		{
			mem_flush(X, N * sizeof(float));
			mem_flush(xcpy, N * sizeof(float));
			BENCH_START(N, sizeof(float)*N, 4, "CBLAS SSCAL")
				memcpy(X, xcpy, N * sizeof(float));
		 		START_RECORD
		 			
		 			cblas_sscal(N, *alpha, X, 1);
		 			
		 		END_RECORD
			BENCH_END
			printf("Verified:\t\t\t");
			if(fverify_benchmark(X, 1, N, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}
		// Run KBLAS benchmark
		{
			mem_flush(X, N * sizeof(float));
			mem_flush(xcpy, N * sizeof(float));
			BENCH_START(N, sizeof(float)*N, 4, "CBLAS SSCAL")
				memcpy(X, xcpy, N * sizeof(float));
		 		START_RECORD
		 			
		 			cblas_sscal(N, *alpha, X, 1);
		 			
		 		END_RECORD
			BENCH_END
			printf("Verified:\t\t\t");
			if(fverify_benchmark(X, 1, N, verif_dir, names[i])) puts("Yes");
			else puts("No");
		}

		// Cleanup
		free(filepath);
		fclose(file);
		free(alpha);
		free(X);
		free(xcpy);
	}
}