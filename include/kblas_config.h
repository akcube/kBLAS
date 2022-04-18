#ifndef __KBLAS_CONFIG_ARCH
#define __KBLAS_CONFIG_ARCH

#define KB(X) X * 1024
#define MB(X) KB(X) * 1024LL
#define NUM_CORES 8

#define L1D_SIZE KB(32)
#define L2_SIZE KB(512)
#define L3_SIZE MB(16)
#define L3C_SIZE L3_SIZE/NUM_CORES
#define BIG_MEM MB(32)

#endif