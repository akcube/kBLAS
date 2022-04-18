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
        register __m256d a1 = _mm256_load_pd(&X[i]);     \
        register __m256d a2 = _mm256_load_pd(&X[i+8]);   \
        register __m256d a3 = _mm256_load_pd(&X[i+16]);  \
        register __m256d a4 = _mm256_load_pd(&X[i+24]);  \
        register __m256d a5 = _mm256_load_pd(&X[i+32]);  \
        register __m256d a6 = _mm256_load_pd(&X[i+40]);  \
        register __m256d b1 = _mm256_load_pd(&Y[i]);     \
        register __m256d b2 = _mm256_load_pd(&Y[i+8]);   \
        register __m256d b3 = _mm256_load_pd(&Y[i+16]);  \
        register __m256d b4 = _mm256_load_pd(&Y[i+24]);  \
        register __m256d b5 = _mm256_load_pd(&Y[i+32]);  \
        register __m256d b6 = _mm256_load_pd(&Y[i+40]);  \
                                                        \
        b1 = _mm256_fmadd_pd(a1, sr1, b1);              \
        b2 = _mm256_fmadd_pd(a2, sr1, b2);              \
        b3 = _mm256_fmadd_pd(a3, sr2, b3);              \
        b4 = _mm256_fmadd_pd(a4, sr2, b4);              \
        b5 = _mm256_fmadd_pd(a5, sr3, b5);              \
        b6 = _mm256_fmadd_pd(a6, sr3, b6);              \

#define UNROLLED_VECT_STORE                             \
        _mm256_store_pd(&Y[i], b1);                     \
        _mm256_store_pd(&Y[i+8], b2);                   \
        _mm256_store_pd(&Y[i+16], b3);                  \
        _mm256_store_pd(&Y[i+24], b4);                  \
        _mm256_store_pd(&Y[i+32], b5);                  \
        _mm256_store_pd(&Y[i+40], b6); }                \


#define UNROLLED_VECT_STREAM                            \
        _mm256_stream_pd(&Y[i], b1);                     \
        _mm256_stream_pd(&Y[i+8], b2);                   \
        _mm256_stream_pd(&Y[i+16], b3);                  \
        _mm256_stream_pd(&Y[i+24], b4);                  \
        _mm256_stream_pd(&Y[i+32], b5);                  \
        _mm256_stream_pd(&Y[i+40], b6); }                \

// -------------------------------------------------------------------------------

/**
 * We have 16 SIMD registers. Use loop unrolling to help GCC register rename
 * and use all 16 registers when possible. Use SIMD mult and parallelize over 
 * threads.
 */
inline __attribute__((always_inline)) 
void kblas_daxpy_inc1(const int N, const double alpha, const double *X, double *Y){

    // Switch to multi-thread only if vector won't fit in even one-fourth of L2
    long long mem = (N * sizeof(double));

    register __m256d sr1 = _mm256_broadcast_sd(&alpha);
    register __m256d sr2 = _mm256_broadcast_sd(&alpha);
    register __m256d sr3 = _mm256_broadcast_sd(&alpha);
    register __m256d sr4 = _mm256_broadcast_sd(&alpha);

    if(mem >= 2*L2_SIZE && mem <= L3C_SIZE){
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
        Y[i] = X[i] * alpha + Y[i];
}

void kblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY){
    omp_set_dynamic(false);
    // Special case if function has stride 1 use SIMD & parallelization
    if(incX == 1 && incY == 1) {
        kblas_daxpy_inc1(N, alpha, X, Y);
        return;
    }
    
    /**
     * For non-unity stride values attempt to parallelize access over cores and 
     * make effective use of pre-fetching.
     */

    // Switch to multi-thread only if vector won't fit in even one-fourth of L2
    long long mem = (N * sizeof(double));

    // TODO: Experiment with masked SIMD operations for stride 2, 4
    int nmax = N * incY;
    for(int i=0, j=0; i<nmax; i+=incY, j+=incX){
        Y[i] = alpha * X[j] + Y[i];
    }
}
