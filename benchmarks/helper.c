#include<stdio.h>
#include<stdlib.h>
#include<dirent.h> 
#include<string.h>
#include<stdbool.h>

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
	void *ret = malloc(bytes);
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
	qsort(names, file_count, sizeof(char*), cmp_func);
	*n = file_count;
	return names;
}


