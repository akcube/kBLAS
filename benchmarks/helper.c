#include<stdio.h>
#include<stdlib.h>
#include<dirent.h> 
#include<string.h>
#include<stdbool.h>
#include<unistd.h>
#include<getopt.h>
#include<limits.h>
#include<ctype.h>

#define ISFILE(X) (strcmp(X, ".") != 0 && strcmp(X, "..") != 0)

/**
 * Comparator to compare two integers stored in string form in ascending
 * order. If a string containing a non-integer is passed it is considered 0. 
 */
int cmp_func(const void *a, const void *b){
	long a_i = strtol(*((char**) a), NULL, 10);
	long b_i = strtol(*((char**) b), NULL, 10);
	return a_i > b_i;
}

/**
 * Memory version of strdup, return a pointer to memory containing duplicated
 * data from src upto `bytes` number of bytes.
 */
void *memdup(void *src, size_t bytes){
	void *ret = aligned_alloc(64, bytes);
	if(!ret) return ret;
	memcpy(ret, src, bytes);
	return ret;
}

/**
 * Given the path to the directory, returns a pointer to a sorted array of 
 * char*'s each of which point to the name of a file in the directory
 * @param  path	The directory path
 * @param  n    Pointer to a variable which will be set to the number
 *              of files in the directory
 * @return      Pointer to sorted char* array containing file names
 */
char** get_files(const char *path, int *n){
	DIR *d; 
	d = opendir(path);
	struct dirent *dir; 
  	
  	int file_count = 0;
	for(;(dir = readdir(d)) != NULL; file_count += ISFILE(dir->d_name));
	
	closedir(d);
	d = opendir(path);

	char **names = malloc(sizeof(char*) * file_count);
	for(int i=0; (dir = readdir(d)) != NULL;){
		if(ISFILE(dir->d_name)) {
			int len = strlen(dir->d_name);
			names[i] = malloc(sizeof(char) * (len + 1));
			memcpy(names[i], dir->d_name, sizeof(char) * (len + 1));
			i++;
		}
	}
	closedir(d);
	qsort(names, file_count, sizeof(char*), cmp_func);
	*n = file_count;
	return names;
}

long long byte_parse(char *cptr){
	long long num, mult=1;
	char *eptr;
	num = strtol(cptr, &eptr, 10);
	for(char *c=eptr; *c; c++) *c = tolower(*c);
	
	if(strcmp(eptr, "kb") == 0) mult = 1000;
	else if(strcmp(eptr, "kib") == 0) mult = 1024;
	else if(strcmp(eptr, "mb") == 0) mult = 1000*1000;
	else if(strcmp(eptr, "mib") == 0) mult = 1024*1024;
	else if(strcmp(eptr, "gb") == 0) mult = 1000*1000*1000;
	else if(strcmp(eptr, "gib") == 0) mult = 1024*1024*1024;
	else if(strlen(eptr) != 0) { 
		puts("Incorrect parameters passed");
		exit(1);
	}

	return num*mult;
}

void arg_parse(int argc, char *argv[], long long *min_mem, long long *max_mem){
	long long min = 0, max = LLONG_MAX;
	static struct option long_options[] = {
		{"min", required_argument, 0, 'l'},
		{"max",   required_argument, 0, 'r'},
		{0, 0, 0, 0}
	}; 
	int option_index, c;
	while ((c = getopt_long(argc, argv, "l:r:", long_options, &option_index)) != -1){
		switch (c){
			// Long argument
			case 0:
				if(option_index == 0 && optarg) min = byte_parse(optarg);
				else if(option_index == 1 && optarg) max = byte_parse(optarg);
			break;

			case 'l':
				if(optarg) min = byte_parse(optarg);
			break;

			case 'r':				
				if(optarg) max = byte_parse(optarg);
			break;

			default:
				puts("Usage ./<program_name> --min [number][/KB/MB/GB] --max [number][/KB/MB/GB]");
				exit(1);
        }
    }
    *min_mem = min;
    *max_mem = max;
    printf("min: %lld, max: %lld\n", min, max);
}
