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
		register __m256d r1 = _mm256_load_pd(&X[i]);		\
		register __m256d r2 = _mm256_load_pd(&X[i+4]);		\
		register __m256d r3 = _mm256_load_pd(&X[i+8]);		\
		register __m256d r4 = _mm256_load_pd(&X[i+12]);		\
		register __m256d r5 = _mm256_load_pd(&X[i+16]);		\
		register __m256d r6 = _mm256_load_pd(&X[i+20]);		\
		register __m256d r7 = _mm256_load_pd(&X[i+24]);		\
		register __m256d r8 = _mm256_load_pd(&X[i+28]);		\
		register __m256d r9 = _mm256_load_pd(&X[i+32]);		\
		register __m256d r10 = _mm256_load_pd(&X[i+36]);	\
		register __m256d r11 = _mm256_load_pd(&X[i+40]);	\
		register __m256d r12 = _mm256_load_pd(&X[i+44]);	\
															\
		r1 = _mm256_mul_pd(r1, sr1);						\
        r2 = _mm256_mul_pd(r2, sr1);						\
        r3 = _mm256_mul_pd(r3, sr1);						\
        r4 = _mm256_mul_pd(r4, sr2);						\
        r5 = _mm256_mul_pd(r5, sr2);						\
        r6 = _mm256_mul_pd(r6, sr2);						\
        r7 = _mm256_mul_pd(r7, sr3);						\
        r8 = _mm256_mul_pd(r8, sr3);						\
        r9 = _mm256_mul_pd(r9, sr3);						\
        r10 = _mm256_mul_pd(r10, sr4);						\
        r11 = _mm256_mul_pd(r11, sr4);						\
        r12 = _mm256_mul_pd(r12, sr4);						\

#define UNROLLED_VECT_STORE									\
		_mm256_store_pd(&X[i], r1);							\
        _mm256_store_pd(&X[i+4], r2);						\
        _mm256_store_pd(&X[i+8], r3);						\
        _mm256_store_pd(&X[i+12], r4);						\
        _mm256_store_pd(&X[i+16], r5);						\
        _mm256_store_pd(&X[i+20], r6);						\
        _mm256_store_pd(&X[i+24], r7);						\
        _mm256_store_pd(&X[i+28], r8);						\
        _mm256_store_pd(&X[i+32], r9);						\
        _mm256_store_pd(&X[i+36], r10);						\
        _mm256_store_pd(&X[i+40], r11);						\
        _mm256_store_pd(&X[i+44], r12);	}					\

#define UNROLLED_VECT_STREAM								\
		_mm256_stream_pd(&X[i], r1);						\
        _mm256_stream_pd(&X[i+4], r2);						\
        _mm256_stream_pd(&X[i+8], r3);						\
        _mm256_stream_pd(&X[i+12], r4);						\
        _mm256_stream_pd(&X[i+16], r5);						\
        _mm256_stream_pd(&X[i+20], r6);						\
        _mm256_stream_pd(&X[i+24], r7);						\
        _mm256_stream_pd(&X[i+28], r8);						\
        _mm256_stream_pd(&X[i+32], r9);						\
        _mm256_stream_pd(&X[i+36], r10);					\
        _mm256_stream_pd(&X[i+40], r11);					\
        _mm256_stream_pd(&X[i+44], r12);	}				\

// -------------------------------------------------------------------------------

/**
 * We have 16 SIMD registers. Use loop unrolling to help GCC register rename
 * and use all 16 registers when possible. Use SIMD mult and parallelize over 
 * threads.
 */
inline __attribute__((always_inline)) 
void kblas_dscal_inc1(const int N, const double alpha, double *X){

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem = (N * sizeof(double));

	register __m256d sr1 = _mm256_broadcast_sd(&alpha);
	register __m256d sr2 = _mm256_broadcast_sd(&alpha);
	register __m256d sr3 = _mm256_broadcast_sd(&alpha);
	register __m256d sr4 = _mm256_broadcast_sd(&alpha);

	if(mem >= L2_SIZE * 2 && mem <= L3C_SIZE){
		#pragma omp parallel for num_threads(4) proc_bind(spread) schedule(static, 128)
		for(int i=0; i<N-48+1; i+=48){
			UNROLLED_VECT_READ_MULT
			UNROLLED_VECT_STORE
		}
	}
	else if(mem > L3C_SIZE && mem < BIG_MEM){
		#pragma omp parallel for num_threads(4) proc_bind(spread)
		for(int i=0; i<N-48+1; i+=48){
			UNROLLED_VECT_READ_MULT
			UNROLLED_VECT_STORE
		}
	}
	else if(mem >= BIG_MEM){
		#pragma omp parallel for num_threads(2) proc_bind(spread)
		for(int i=0; i<N-48+1; i+=48){
			UNROLLED_VECT_READ_MULT	
			UNROLLED_VECT_STREAM
		}
	}
	else {
		for(int i=0; i<N-48+1; i+=48){
			UNROLLED_VECT_READ_MULT	
			UNROLLED_VECT_STORE
		}
	}

	for(int i = ((N-(N % 48)) > 0 ? N-(N % 48) : 0); i<N; i++)
		X[i] *= alpha;
}

void kblas_dscal(const int N, const double alpha, double *X, const int incX){
	omp_set_dynamic(false);
	// Special case if function has stride 1 use SIMD & parallelization
	if(incX == 1) {
		kblas_dscal_inc1(N, alpha, X);
		return;
	}
	
	/**
	 * For non-unity stride values attempt to parallelize access over cores and 
	 * make effective use of pre-fetching.
	 */
	int nmax = N * incX;

	// Switch to multi-thread only if vector won't fit in even one-fourth of L2
	long long mem_mb = (N * sizeof(double));
	bool multi_thread = (mem_mb >= L2_SIZE/4);

	// TODO: Experiment with masked SIMD operations for stride 2, 4
	if(multi_thread){
		#pragma omp parallel for num_threads(4) proc_bind(spread)
		for (int i = 0; i < nmax; i+=incX) 
			X[i] *= alpha;
	}
	else{
		for (int i = 0; i < nmax; i+=incX) 
			X[i] *= alpha;
	}
}
