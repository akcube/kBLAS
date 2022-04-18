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

#define UNROLLED_VECT_READ_MULT {\
		register __m256 r1 = _mm256_load_ps(&X[i]);		\
		register __m256 r2 = _mm256_load_ps(&X[i+8]);	\
		register __m256 r3 = _mm256_load_ps(&X[i+16]);	\
		register __m256 r4 = _mm256_load_ps(&X[i+24]);	\
		register __m256 r5 = _mm256_load_ps(&X[i+32]);	\
		register __m256 r6 = _mm256_load_ps(&X[i+40]);	\
		register __m256 r7 = _mm256_load_ps(&X[i+48]);	\
		register __m256 r8 = _mm256_load_ps(&X[i+56]);	\
		register __m256 r9 = _mm256_load_ps(&X[i+64]);	\
		register __m256 r10 = _mm256_load_ps(&X[i+72]);	\
		register __m256 r11 = _mm256_load_ps(&X[i+80]);	\
		register __m256 r12 = _mm256_load_ps(&X[i+88]);	\
														\
		r1 = _mm256_mul_ps(r1, sr1);					\
        r2 = _mm256_mul_ps(r2, sr1);					\
        r3 = _mm256_mul_ps(r3, sr1);					\
        r4 = _mm256_mul_ps(r4, sr2);					\
        r5 = _mm256_mul_ps(r5, sr2);					\
        r6 = _mm256_mul_ps(r6, sr2);					\
        r7 = _mm256_mul_ps(r7, sr3);					\
        r8 = _mm256_mul_ps(r8, sr3);					\
        r9 = _mm256_mul_ps(r9, sr3);					\
        r10 = _mm256_mul_ps(r10, sr4);					\
        r11 = _mm256_mul_ps(r11, sr4);					\
        r12 = _mm256_mul_ps(r12, sr4);					\

#define UNROLLED_VECT_STORE								\
		_mm256_store_ps(&X[i], r1);						\
        _mm256_store_ps(&X[i+8], r2);					\
        _mm256_store_ps(&X[i+16], r3);					\
        _mm256_store_ps(&X[i+24], r4);					\
        _mm256_store_ps(&X[i+32], r5);					\
        _mm256_store_ps(&X[i+40], r6);					\
        _mm256_store_ps(&X[i+48], r7);					\
        _mm256_store_ps(&X[i+56], r8);					\
        _mm256_store_ps(&X[i+64], r9);					\
        _mm256_store_ps(&X[i+72], r10);					\
        _mm256_store_ps(&X[i+80], r11);					\
        _mm256_store_ps(&X[i+88], r12);	}				\

#define UNROLLED_VECT_STREAM							\
		_mm256_stream_ps(&X[i], r1);					\
        _mm256_stream_ps(&X[i+8], r2);					\
        _mm256_stream_ps(&X[i+16], r3);					\
        _mm256_stream_ps(&X[i+24], r4);					\
        _mm256_stream_ps(&X[i+32], r5);					\
        _mm256_stream_ps(&X[i+40], r6);					\
        _mm256_stream_ps(&X[i+48], r7);					\
        _mm256_stream_ps(&X[i+56], r8);					\
        _mm256_stream_ps(&X[i+64], r9);					\
        _mm256_stream_ps(&X[i+72], r10);				\
        _mm256_stream_ps(&X[i+80], r11);				\
        _mm256_stream_ps(&X[i+88], r12);	}			\

// -------------------------------------------------------------------------------

/**
 * We have 16 SIMD registers. Use loop unrolling to help GCC register rename
 * and use all 16 registers when possible. Use SIMD mult and parallelize over 
 * threads.
 */
inline __attribute__((always_inline)) 
void kblas_sscal_inc1(const int N, const float alpha, float *X){

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem = (N * sizeof(float));

	register __m256 sr1 = _mm256_broadcast_ss(&alpha);
	register __m256 sr2 = _mm256_broadcast_ss(&alpha);
	register __m256 sr3 = _mm256_broadcast_ss(&alpha);
	register __m256 sr4 = _mm256_broadcast_ss(&alpha);

	if(mem >= 2*L2_SIZE && mem <= L3C_SIZE){
		#pragma omp parallel for num_threads(4) proc_bind(spread) schedule(static, 128)
		for(int i=0; i<N-96+1; i+=96){
			UNROLLED_VECT_READ_MULT
			UNROLLED_VECT_STORE
		}
	}
	else if(mem > L3C_SIZE && mem < BIG_MEM){
		#pragma omp parallel for num_threads(4) proc_bind(spread)
		for(int i=0; i<N-96+1; i+=96){
			UNROLLED_VECT_READ_MULT
			UNROLLED_VECT_STORE
		}
	}
	else if(mem >= BIG_MEM){
		#pragma omp parallel for num_threads(2) proc_bind(spread)
		for(int i=0; i<N-96+1; i+=96){
			UNROLLED_VECT_READ_MULT	
			UNROLLED_VECT_STREAM
		}
	}
	else {
		for(int i=0; i<N-96+1; i+=96){
			UNROLLED_VECT_READ_MULT	
			UNROLLED_VECT_STORE
		}
	}

	for(int i = ((N-(N % 96)) > 0 ? N-(N % 96) : 0); i<N; i++)
		X[i] *= alpha;
}

void kblas_sscal(const int N, const float alpha, float *X, const int incX){
	omp_set_dynamic(false);
	// Special case if function has stride 1 use SIMD & parallelization
	if(incX == 1) {
		kblas_sscal_inc1(N, alpha, X);
		return;
	}
	
	/**
	 * For non-unity stride values attempt to parallelize access over cores and 
	 * make effective use of pre-fetching.
	 */
	int nmax = N * incX;

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem = (N * sizeof(float));

	// TODO: Experiment with masked SIMD operations for stride 2, 4
	if(mem >= 2*L2_SIZE && mem <= L3C_SIZE){
		#pragma omp parallel for num_threads(4) proc_bind(spread) schedule(static, 128)
		for (int i = 0; i < nmax; i+=incX) 
			X[i] *= alpha;
	}
	else if(mem > L3C_SIZE && mem < BIG_MEM){
		#pragma omp parallel for num_threads(4) proc_bind(spread)
		for (int i = 0; i < nmax; i+=incX) 
			X[i] *= alpha;
	}
	else if(mem >= BIG_MEM){
		#pragma omp parallel for num_threads(2) proc_bind(spread)
		for (int i = 0; i < nmax; i+=incX) 
			X[i] *= alpha;
	}
	else {
		for (int i = 0; i < nmax; i+=incX) 
			X[i] *= alpha;
	}
}
