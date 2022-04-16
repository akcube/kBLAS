#ifndef __BENCH_HELPER_H
#define __BENCH_HELPER_H

char** get_files(const char *name, int *n);
void *memdup(void *src, size_t bytes);
void arg_parse(int argc, char *argv[], long long *min_mem, long long *max_mem);

#endif