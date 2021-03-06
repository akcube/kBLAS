#include <kblas.h>

void kblas_sgemv(const enum KBLAS_ORDER order,
                 const enum KBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY){

    // Y[i] = \beta * Y[i] + \alpha * (A[i][0] * x[0]+ A[i][1] * x[1] + A[i][2] * x[2] + ...) 
    int lenX, lenY;
    if (TransA == KblasNoTrans) { lenX = N, lenY = M; }
    else { lenX = M, lenY = N; }

    // Standard
    if((order == KblasRowMajor && TransA == KblasNoTrans) || (order == KblasColMajor && TransA == KblasTrans)){
        kblas_sscal(lenY, beta, Y, incY);
        #pragma omp parallel for num_threads(4) proc_bind(spread)
        for(int i=0; i<lenY; i++){
            float sum = 0;
            #pragma omp simd
            for(int j=0; j<lenX; j++)
                sum += A[i*lda + j] * X[j*incX];
            Y[i*incY] += alpha * sum;
        }
    }
    else{
        kblas_sscal(lenY, beta, Y, incY);
        for(int j=0; j<lenX; j++){
            float sum = X[j] * alpha;
            #pragma omp parallel for  num_threads(4) proc_bind(spread)
            for(int i=0; i<lenY; i++)
                Y[i*incY] += sum * A[lda*j + i];
        }
    }
}