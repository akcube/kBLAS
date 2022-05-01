#include <stdio.h>
#include "benchmark.h"
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

Result kernel_tooslow(KernelArgs args){
	double alpha = (double)rand()/RAND_MAX;
	double c = (double)rand()/RAND_MAX;
	double res = 0;

	for(int i=0; i<args.iters; i++)
		res = (res*alpha) + c;

	Result ret;
	ret.result = res;
	ret.flop_ct = (long long) args.iters * 2LL;
	ret.mem_accesses = 0;
	return ret;
}

int main(void){
	output_hwinfo();
    srand(time(0));
	KernelArgs args = {100000000, NULL};
	benchmark(kernel_tooslow, args, 3, "The weirdly slow one");
}