CC = gcc
CFLAGS = -O3 -ffast-math -mavx2 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 -march=znver3 -DTUNED

all: stream_f.exe stream_c.exe

stream_c.exe: stream.c
	$(CC) $(CFLAGS) stream.c -o stream_c.exe

clean:
	rm -f stream_c.exe *.o

# an example of a more complex build line for the Intel icc compiler
stream.icc: stream.c
	icc -O3 -xCORE-AVX2 -qopenmp -g -ffreestanding -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream.omp.AVX2.80M.20x.icc -DTUNED
