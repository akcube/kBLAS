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
        register __m256 a1 = _mm256_load_ps(&X[i]);     \
        register __m256 a2 = _mm256_load_ps(&X[i+8]);   \
        register __m256 a3 = _mm256_load_ps(&X[i+16]);  \
        register __m256 a4 = _mm256_load_ps(&X[i+24]);  \
        register __m256 a5 = _mm256_load_ps(&X[i+32]);  \
        register __m256 a6 = _mm256_load_ps(&X[i+40]);  \
        register __m256 b1 = _mm256_load_ps(&Y[i]);     \
        register __m256 b2 = _mm256_load_ps(&Y[i+8]);   \
        register __m256 b3 = _mm256_load_ps(&Y[i+16]);  \
        register __m256 b4 = _mm256_load_ps(&Y[i+24]);  \
        register __m256 b5 = _mm256_load_ps(&Y[i+32]);  \
        register __m256 b6 = _mm256_load_ps(&Y[i+40]);  \
                                                        \
        b1 = _mm256_fmadd_ps(a1, sr1, b1);              \
        b2 = _mm256_fmadd_ps(a2, sr1, b2);              \
        b3 = _mm256_fmadd_ps(a3, sr2, b3);              \
        b4 = _mm256_fmadd_ps(a4, sr2, b4);              \
        b5 = _mm256_fmadd_ps(a5, sr3, b5);              \
        b6 = _mm256_fmadd_ps(a6, sr3, b6);              \

#define UNROLLED_VECT_STORE                             \
        _mm256_store_ps(&Y[i], b1);                     \
        _mm256_store_ps(&Y[i+8], b2);                   \
        _mm256_store_ps(&Y[i+16], b3);                  \
        _mm256_store_ps(&Y[i+24], b4);                  \
        _mm256_store_ps(&Y[i+32], b5);                  \
        _mm256_store_ps(&Y[i+40], b6); }                \


#define UNROLLED_VECT_STREAM                            \
        _mm256_stream_ps(&Y[i], b1);                     \
        _mm256_stream_ps(&Y[i+8], b2);                   \
        _mm256_stream_ps(&Y[i+16], b3);                  \
        _mm256_stream_ps(&Y[i+24], b4);                  \
        _mm256_stream_ps(&Y[i+32], b5);                  \
        _mm256_stream_ps(&Y[i+40], b6); }                \

// -------------------------------------------------------------------------------

/**
 * We have 16 SIMD registers. Use loop unrolling to help GCC register rename
 * and use all 16 registers when possible. Use SIMD mult and parallelize over 
 * threads.
 */
inline __attribute__((always_inline)) 
void kblas_saxpy_inc1(const int N, const float alpha, const float *X, float *Y){

    // Switch to multi-thread only if vector won't fit in even one-fourth of L2
    long long mem = (N * sizeof(float));

    register __m256 sr1 = _mm256_broadcast_ss(&alpha);
    register __m256 sr2 = _mm256_broadcast_ss(&alpha);
    register __m256 sr3 = _mm256_broadcast_ss(&alpha);
    register __m256 sr4 = _mm256_broadcast_ss(&alpha);

    if(mem >= 2*L2_SIZE && mem <= L3C_SIZE){
        #pragma omp parallel for num_threads(2) proc_bind(spread) schedule(static, 128)
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
        Y[i] = X[i] * alpha + Y[i];
}

void kblas_saxpy(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY){
    omp_set_dynamic(false);
    // Special case if function has stride 1 use SIMD & parallelization
    if(incX == 1 && incY == 1) {
        kblas_saxpy_inc1(N, alpha, X, Y);
        return;
    }
    
    /**
     * For non-unity stride values attempt to parallelize access over cores and 
     * make effective use of pre-fetching.
     */

    // Switch to multi-thread only if vector won't fit in even one-fourth of L2
    long long mem = (N * sizeof(float));

    // TODO: Experiment with masked SIMD operations for stride 2, 4
    int nmax = N * incY;
    for(int i=0, j=0; i<nmax; i+=incY, j+=incX){
        Y[i] = alpha * X[j] + Y[i];
    }
}
