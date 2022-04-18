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

#define VECTOR_DECL register __m256 acc1 = _mm256_setzero_ps(); \
	register __m256 acc2 = _mm256_setzero_ps();		\
	register __m256 acc3 = _mm256_setzero_ps();		\

#define UNROLLED_VECT_COMP {\
	register __m256 a1 = _mm256_load_ps(&X[i]);		\
	register __m256 a2 = _mm256_load_ps(&X[i+8]);		\
	register __m256 a3 = _mm256_load_ps(&X[i+16]);		\
	register __m256 a4 = _mm256_load_ps(&X[i+24]);		\
	register __m256 a5 = _mm256_load_ps(&X[i+32]);		\
	register __m256 a6 = _mm256_load_ps(&X[i+40]);		\
	register __m256 b1 = _mm256_load_ps(&Y[i]);		\
	register __m256 b2 = _mm256_load_ps(&Y[i+8]);		\
	register __m256 b3 = _mm256_load_ps(&Y[i+16]);		\
	register __m256 b4 = _mm256_load_ps(&Y[i+24]);		\
	register __m256 b5 = _mm256_load_ps(&Y[i+32]);		\
	register __m256 b6 = _mm256_load_ps(&Y[i+40]);		\
								\
	acc1 = _mm256_fmadd_ps(a1, b1, acc1);			\
        acc1 = _mm256_fmadd_ps(a2, b2, acc1); 			\
        acc2 = _mm256_fmadd_ps(a3, b3, acc2); 			\
        acc2 = _mm256_fmadd_ps(a4, b4, acc2); 			\
        acc3 = _mm256_fmadd_ps(a5, b5, acc3); 			\
        acc3 = _mm256_fmadd_ps(a6, b6, acc3);}			\

#define ACCUM_COMP acc1 = _mm256_add_ps(acc1, acc2);		\
        acc1 = _mm256_add_ps(acc1, acc3);			\
	__m128 x128 = _mm_add_ps(_mm256_extractf128_ps(acc1, 1), _mm256_castps256_ps128(acc1));	\
	__m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));				\
	__m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));				\
        float accumulator = _mm_cvtss_f32(x32);														\

// -------------------------------------------------------------------------------

/**
 * We have 16 SIMD registers. Use loop unrolling to help GCC register rename
 * and use all 16 registers when possible. Use SIMD mult and parallelize over 
 * threads.
 */
// inline __attribute__((always_inline)) 
float kblas_sdot_inc1(const int N, const float *X, const float *Y){

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem = (N * sizeof(float));
	float result = 0;

	if(mem >= 2*L2_SIZE && mem <= L3C_SIZE){
		#pragma omp parallel reduction(+:result) num_threads(4) proc_bind(spread)
		{
			VECTOR_DECL
			#pragma omp for schedule(static, 128) nowait
			for(int i=0; i<N-48+1; i+=48)
				UNROLLED_VECT_COMP
			ACCUM_COMP
			result += accumulator;
		}
	}
	else if(mem > L3C_SIZE && mem < BIG_MEM){
		#pragma omp parallel reduction(+:result) num_threads(4) proc_bind(spread)
		{
			VECTOR_DECL
			#pragma omp for nowait
			for(int i=0; i<N-48+1; i+=48)
				UNROLLED_VECT_COMP
			ACCUM_COMP
			result += accumulator;
		}
	}
	else if(mem >= BIG_MEM){
		#pragma omp parallel reduction(+:result) num_threads(2) proc_bind(spread)
		{
			VECTOR_DECL
			#pragma omp for nowait
			for(int i=0; i<N-48+1; i+=48)
				UNROLLED_VECT_COMP
			ACCUM_COMP
			result += accumulator;
		}
	}
	else {
		VECTOR_DECL
		for(int i=0; i<N-48+1; i+=48)
			UNROLLED_VECT_COMP	
		ACCUM_COMP
		result += accumulator;
	}

	for(int i = ((N-(N % 48)) > 0 ? N-(N % 48) : 0); i<N; i++)
		result += X[i] * Y[i];
	return result;
}

float kblas_sdot(const int N, const float  *X, const int incX, const float  *Y, const int incY){
	omp_set_dynamic(false);
	// Special case if function has stride 1 use SIMD & parallelization
	if(incX == 1 && incY == 1)
		return kblas_sdot_inc1(N, X, Y);
	
	/**
	 * For non-unity stride values attempt to parallelize access over cores and 
	 * make effective use of pre-fetching.
	 */
	int nmax = N * incX;

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem_mb = (N * sizeof(float));

	// TODO: Experiment with masked SIMD operations for stride 2, 4
	float accum = 0;
	for(int i=0, j=0; i<nmax; i+=incX, j+=incY)
		accum += X[i] * Y[j];
	return accum;
}
