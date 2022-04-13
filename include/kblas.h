#ifndef KBLAS_H
#define KBLAS_H

enum KBLAS_ORDER {KblasRowMajor=101, KblasColMajor=102};
enum KBLAS_TRANSPOSE {KblasNoTrans=111, KblasTrans=112, KblasConjTrans=113};

// BLAS Level 1

void kblas_sscal(const int N, const float alpha, float *X, const int incX);
void kblas_dscal(const int N, const double alpha, double *X, const int incX);

float  kblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);

double kblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY);

void kblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);

void kblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY);

// BLAS Level 2 

void kblas_sgemv(const enum KBLAS_ORDER order,
                 const enum KBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);

void kblas_dgemv(const enum KBLAS_ORDER order,
                 const enum KBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);


// BLAS Level 3

void kblas_sgemm(const enum KBLAS_ORDER Order, const enum KBLAS_TRANSPOSE TransA,
                 const enum KBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void kblas_dgemm(const enum KBLAS_ORDER Order, const enum KBLAS_TRANSPOSE TransA,
                 const enum KBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
#endif