#include <kblas.h>
#include <stdio.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>
#include <kblas_config.h>

//--------------------------------------------------------------------------------
/**
 * Unrolled loop code in a macro so I can easily generate a version parallelized
 * using OpenMP and one untouched by OpenMP. #pragma omp ... if(f) also has too
 * much overhead. Generating separate copies gives the best performance.
 */

#define VECTOR_DECL register __m256d acc1 = _mm256_setzero_pd(); \
	register __m256d acc2 = _mm256_setzero_pd();		\
	register __m256d acc3 = _mm256_setzero_pd();		\

#define UNROLLED_VECT_COMP {\
	register __m256d a1 = _mm256_load_pd(&X[i]);		\
	register __m256d a2 = _mm256_load_pd(&X[i+4]);		\
	register __m256d a3 = _mm256_load_pd(&X[i+8]);		\
	register __m256d a4 = _mm256_load_pd(&X[i+12]);		\
	register __m256d a5 = _mm256_load_pd(&X[i+16]);		\
	register __m256d a6 = _mm256_load_pd(&X[i+20]);		\
	register __m256d b1 = _mm256_load_pd(&Y[i]);		\
	register __m256d b2 = _mm256_load_pd(&Y[i+4]);		\
	register __m256d b3 = _mm256_load_pd(&Y[i+8]);		\
	register __m256d b4 = _mm256_load_pd(&Y[i+12]);		\
	register __m256d b5 = _mm256_load_pd(&Y[i+16]);		\
	register __m256d b6 = _mm256_load_pd(&Y[i+20]);		\
														\
	acc1 = _mm256_fmadd_pd(a1, b1, acc1);				\
	acc1 = _mm256_fmadd_pd(a2, b2, acc1); 				\
	acc2 = _mm256_fmadd_pd(a3, b3, acc2); 				\
	acc2 = _mm256_fmadd_pd(a4, b4, acc2); 				\
	acc3 = _mm256_fmadd_pd(a5, b5, acc3); 				\
	acc3 = _mm256_fmadd_pd(a6, b6, acc3);}				\

#define ACCUM_COMP acc1 = _mm256_add_pd(acc1, acc2);							\
        acc1 = _mm256_add_pd(acc1, acc3);										\
		acc1 = _mm256_hadd_pd(acc1, acc1);										\
		acc1 = _mm256_add_pd(acc1, _mm256_permute2f128_pd(acc1, acc1, 0x31));	\
        double accumulator = _mm256_cvtsd_f64(acc1);								\

// -------------------------------------------------------------------------------

/**
 * We have 16 SIMD registers. Use loop unrolling to help GCC register rename
 * and use all 16 registers when possible. Use SIMD mult and parallelize over 
 * threads.
 */
// inline __attribute__((always_inline)) 
double kblas_ddot_inc1(const int N, const double *X, const double *Y){

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem = (N * sizeof(double));
	double result = 0;

	if(mem >= 2*L2_SIZE && mem <= L3C_SIZE){
		#pragma omp parallel reduction(+:result) num_threads(4) proc_bind(spread) 
		{
			VECTOR_DECL
			#pragma omp for schedule(static, 128)
			for(int i=0; i<N-24+1; i+=24)
				UNROLLED_VECT_COMP
			ACCUM_COMP
			result += accumulator;
		}
	}
	else if(mem > L3C_SIZE && mem < BIG_MEM){
		#pragma omp parallel reduction(+:result) num_threads(4) proc_bind(spread) 
		{
			VECTOR_DECL
			#pragma omp for
			for(int i=0; i<N-24+1; i+=24)
				UNROLLED_VECT_COMP
			ACCUM_COMP
			result += accumulator;
		}
	}
	else if(mem >= BIG_MEM){
		#pragma omp parallel reduction(+:result) num_threads(2) proc_bind(spread) 
		{
			VECTOR_DECL
			#pragma omp for
			for(int i=0; i<N-24+1; i+=24)
				UNROLLED_VECT_COMP
			ACCUM_COMP
			result += accumulator;
		}
	}
	else {
		VECTOR_DECL
		for(int i=0; i<N-24+1; i+=24)
			UNROLLED_VECT_COMP	
		ACCUM_COMP
		result += accumulator;
	}

	for(int i = ((N-(N % 24)) > 0 ? N-(N % 24) : 0); i<N; i++)
		result += X[i] * Y[i];
	return result;
}

double kblas_ddot(const int N, const double  *X, const int incX, const double  *Y, const int incY){
	omp_set_dynamic(false);
	// Special case if function has stride 1 use SIMD & parallelization
	if(incX == 1 && incY == 1)
		return kblas_ddot_inc1(N, X, Y);
	
	/**
	 * For non-unity stride values attempt to parallelize access over cores and 
	 * make effective use of pre-fetching.
	 */
	int nmax = N * incX;

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem_mb = (N * sizeof(double));

	// TODO: Experiment with masked SIMD operations for stride 2, 4
	double accum = 0;
	for(int i=0, j=0; i<nmax; i+=incX, j+=incY)
		accum += X[i] * Y[j];
	return accum;
}
