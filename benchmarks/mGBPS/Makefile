CC = gcc
IDIR = ../lib
CFLAGS = -O3 -ffast-math -mavx2 -fopenmp -march=znver3

mgbps: mgbps.c
	$(CC) $(CFLAGS) mgbps.c -o mgbps -I$(IDIR) ../lib/benchmark.c

clean:
	rm -f mgbps *.o

mgbps.icc: mgbps.c
	icc -O3 -qopenmp mgbps.c -o mgbps.omp.AVX2.80M.20x.icc -I$(IDIR) ../lib/benchmark.c
