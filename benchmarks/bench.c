#include<cblas.h>
#include<kblas.h>
#include<benchmark.h>
#include<stdlib.h>
#include<stdio.h>

#define N 10000000
float arr[N];



int main(void){

	float alpha = (float) rand()/RAND_MAX;

	for(int i=0; i<N; i++) arr[i] = (float) rand()/RAND_MAX;

	BENCH_START(N, sizeof(float)*N, 5, "CBLAS SSCAL")
	 	cblas_sscal(N, alpha, arr, 1);
	BENCH_END
}