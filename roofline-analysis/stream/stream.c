 /*
    This is a modified version of the Stream benchmark https://www.cs.virginia.edu/stream/
    The code has been tuned to make efficint use of AVX2 and FMA instructions. The tuned code 
    can be found in functions who's names are prefixed with "tuned_STREAM_"
    Modifications to the original benchmark mainly include forcing all the statically allocated
    arrays to 32-bit boundaries and limiting the number of threads OpenMP spawns to 2. 
    I recommend setting this value to the number of memory channels you have available.
    This can be done by compiling with -DNUMTHREADS=n
    It is distributed under the same license as the original Stream benchmark (LICENSE.txt)
 */

# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <omp.h>
# include <sys/time.h>
# include <immintrin.h>

/*-----------------------------------------------------------------------
 * INSTRUCTIONS:
 *
 *	1) STREAM requires different amounts of memory to run on different
 *           systems, depending on both the system cache size(s) and the
 *           granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *           to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. I don't worry about the difference
 *           between 10^6 and 2^20, so in practice the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.  
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.  
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.  
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M elements.
 */
#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	80000000
#endif

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */
#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

/*  Users are allowed to modify the "OFFSET" variable, which *may* change the
 *         relative alignment of the arrays (though compilers may change the 
 *         effective offset by making the arrays non-contiguous on some systems). 
 *      Use of non-zero values for OFFSET can be especially helpful if the
 *         STREAM_ARRAY_SIZE is set to a value close to a large power of 2.
 *      OFFSET can also be set on the compile line without changing the source
 *         code using, for example, "-DOFFSET=56".
 */
#ifndef OFFSET
#   define OFFSET	0
#endif

/*
 *	3) Compile the code with optimization.  Many compilers generate
 *       unreasonably bad code before the optimizer tightens things up.  
 *     If the results are unreasonably good, on the other hand, the
 *       optimizer might be too smart for me!
 *
 *     For a simple single-core version, try compiling with:
 *            cc -O stream.c -o stream
 *     This is known to work on many, many systems....
 *
 *     To use multiple cores, you need to tell the compiler to obey the OpenMP
 *       directives in the code.  This varies by compiler, but a common example is
 *            gcc -O -fopenmp stream.c -o stream_omp
 *       The environment variable OMP_NUM_THREADS allows runtime control of the 
 *         number of threads/cores used when the resulting "stream_omp" program
 *         is executed.
 *
 *     To run with single-precision variables and arithmetic, simply add
 *         -DSTREAM_TYPE=float
 *     to the compile line.
 *     Note that this changes the minimum array sizes required --- see (1) above.
 *
 *     The preprocessor directive "TUNED" does not do much -- it simply causes the 
 *       code to call separate functions to execute each kernel.  Trivial versions
 *       of these functions are provided, but they are *not* tuned -- they just 
 *       provide predefined interfaces to be replaced with tuned code.
 *
 *
 *	4) Optional: Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include info that will help me understand:
 *		a) the computer hardware configuration (e.g., processor model, memory type)
 *		b) the compiler name/version and compilation flags
 *      c) any run-time information (such as OMP_NUM_THREADS)
 *		d) all of the output from the test case.
 *
 * Thanks!
 *
 *-----------------------------------------------------------------------*/

#define NUM_THREADS 2

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

static STREAM_TYPE	a[STREAM_ARRAY_SIZE+OFFSET] __attribute__ ((aligned (32)));
static STREAM_TYPE  b[STREAM_ARRAY_SIZE+OFFSET] __attribute__ ((aligned (32)));
static STREAM_TYPE  c[STREAM_ARRAY_SIZE+OFFSET] __attribute__ ((aligned (32)));

static double	avgtime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };

extern double mysecond();
extern void checkSTREAMresults();
#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(STREAM_TYPE scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(STREAM_TYPE scalar);
#endif
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif
int
main()
    {
    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t, times[4][NTIMES];

    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    printf("STREAM version $Revision: 5.10 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf(HLINE);
#ifdef N
    printf("*****  WARNING: ******\n");
    printf("      It appears that you set the preprocessor variable N when compiling this code.\n");
    printf("      This version of the code uses the preprocesor variable STREAM_ARRAY_SIZE to control the array size\n");
    printf("      Reverting to default value of STREAM_ARRAY_SIZE=%llu\n",(unsigned long long) STREAM_ARRAY_SIZE);
    printf("*****  WARNING: ******\n");
#endif

    printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE, OFFSET);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n", 
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf(" The *best* time for each kernel (excluding the first iteration)\n"); 
    printf(" will be used to compute the reported bandwidth.\n");

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel 
    {
#pragma omp master
	{
	    k = omp_get_num_threads();
	    printf ("Number of Threads requested = %i\n",k);
        }
    }
#endif

#ifdef _OPENMP
	k = 0;
#pragma omp parallel
#pragma omp atomic 
		k++;
    printf ("Number of Threads counted = %i\n",k);
#endif

    /* Get initial value for system clock. */
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
    for (j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }

    t = mysecond();
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
    for (j = 0; j < STREAM_ARRAY_SIZE; j++)
		a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	times[0][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Copy();
#else
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j];
#endif
	times[0][k] = mysecond() - times[0][k];
	
	times[1][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    b[j] = scalar*c[j];
#endif
	times[1][k] = mysecond() - times[1][k];
	
	times[2][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j]+b[j];
#endif
	times[2][k] = mysecond() - times[2][k];
	
	times[3][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    a[j] = b[j]+scalar*c[j];
#endif
	times[3][k] = mysecond() - times[3][k];
	}

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}
    
    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);

		printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults();
    printf(HLINE);

    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }



/* A clock_gettime routine to give access to the CPU
   realtime timer on most UNIX-like systems.  */

#include <time.h>

double mysecond()
{
        struct timespec tinfo;        
        clock_gettime(CLOCK_REALTIME, &tinfo);
        return (double) tinfo.tv_sec + tinfo.tv_nsec * 1e-9;
}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
void checkSTREAMresults ()
{
	STREAM_TYPE aj,bj,cj,scalar;
	STREAM_TYPE aSumErr,bSumErr,cSumErr;
	STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
	double epsilon;
	ssize_t	j;
	int	k,ierr,err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }

    /* accumulate deltas between observed and expected results */
	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
	for (j=0; j<STREAM_ARRAY_SIZE; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
		// if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	// MCCALPIN
	}
	aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

	if (sizeof(STREAM_TYPE) == 4) {
		epsilon = 1.e-6;
	}
	else if (sizeof(STREAM_TYPE) == 8) {
		epsilon = 1.e-13;
	}
	else {
		printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(STREAM_TYPE));
		epsilon = 1.e-6;
	}

	err = 0;
	if (abs(aAvgErr/aj) > epsilon) {
		err++;
		printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(a[j]/aj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,aj,a[j],abs((aj-a[j])/aAvgErr));
				}
#endif
			}
		}
		printf("     For array a[], %d errors were found.\n",ierr);
	}
	if (abs(bAvgErr/bj) > epsilon) {
		err++;
		printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(b[j]/bj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,bj,b[j],abs((bj-b[j])/bAvgErr));
				}
#endif
			}
		}
		printf("     For array b[], %d errors were found.\n",ierr);
	}
	if (abs(cAvgErr/cj) > epsilon) {
		err++;
		printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(c[j]/cj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,cj,c[j],abs((cj-c[j])/cAvgErr));
				}
#endif
			}
		}
		printf("     For array c[], %d errors were found.\n",ierr);
	}
	if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

#ifdef TUNED
/* stubs for "tuned" versions of the kernels */
void tuned_STREAM_Copy()
{
    ssize_t j;
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
    for (j=0; j<STREAM_ARRAY_SIZE-64; j+=64){
        __m256d r1 = _mm256_load_pd(&a[j]);
        __m256d r2 = _mm256_load_pd(&a[j+4]);
        __m256d r3 = _mm256_load_pd(&a[j+8]);
        __m256d r4 = _mm256_load_pd(&a[j+12]);
        __m256d r5 = _mm256_load_pd(&a[j+16]);
        __m256d r6 = _mm256_load_pd(&a[j+20]);
        __m256d r7 = _mm256_load_pd(&a[j+24]);
        __m256d r8 = _mm256_load_pd(&a[j+28]);
        __m256d r9 = _mm256_load_pd(&a[j+32]);
        __m256d r10 = _mm256_load_pd(&a[j+36]);
        __m256d r11 = _mm256_load_pd(&a[j+40]);
        __m256d r12 = _mm256_load_pd(&a[j+44]);
        __m256d r13 = _mm256_load_pd(&a[j+48]);
        __m256d r14 = _mm256_load_pd(&a[j+52]);
        __m256d r15 = _mm256_load_pd(&a[j+56]);
        __m256d r16 = _mm256_load_pd(&a[j+60]);
        _mm256_stream_pd(&c[j], r1);
        _mm256_stream_pd(&c[j+4], r2);
        _mm256_stream_pd(&c[j+8], r3);
        _mm256_stream_pd(&c[j+12], r4);
        _mm256_stream_pd(&c[j+16], r5);
        _mm256_stream_pd(&c[j+20], r6);
        _mm256_stream_pd(&c[j+24], r7);
        _mm256_stream_pd(&c[j+28], r8);
        _mm256_stream_pd(&c[j+32], r9);
        _mm256_stream_pd(&c[j+36], r10);
        _mm256_stream_pd(&c[j+40], r11);
        _mm256_stream_pd(&c[j+44], r12);
        _mm256_stream_pd(&c[j+48], r13);
        _mm256_stream_pd(&c[j+52], r14);
        _mm256_stream_pd(&c[j+56], r15);
        _mm256_stream_pd(&c[j+60], r16);
    }
#pragma omp parallel for num_threads(NUM_THREADS)
    for(j=MAX(0, STREAM_ARRAY_SIZE-64); j < STREAM_ARRAY_SIZE; j++)
        c[j] = a[j];
}

void tuned_STREAM_Scale(STREAM_TYPE scalar)
{
	ssize_t j;
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

    __m256d sr = _mm256_broadcast_sd(&scalar);
#pragma omp parallel for num_threads(NUM_THREADS)
	for (j=0; j<STREAM_ARRAY_SIZE-48; j+=48){
        __m256d r1 = _mm256_load_pd(&c[j]);
        __m256d r2 = _mm256_load_pd(&c[j+4]);
        __m256d r3 = _mm256_load_pd(&c[j+8]);
        __m256d r4 = _mm256_load_pd(&c[j+12]);
        __m256d r5 = _mm256_load_pd(&c[j+16]);
        __m256d r6 = _mm256_load_pd(&c[j+20]);
        __m256d r7 = _mm256_load_pd(&c[j+24]);
        __m256d r8 = _mm256_load_pd(&c[j+28]);
        __m256d r9 = _mm256_load_pd(&c[j+32]);
        __m256d r10 = _mm256_load_pd(&c[j+36]);
        __m256d r11 = _mm256_load_pd(&c[j+40]);
        __m256d r12 = _mm256_load_pd(&c[j+44]);
        r1 = _mm256_mul_pd(r1, sr);
        r2 = _mm256_mul_pd(r2, sr);
        r3 = _mm256_mul_pd(r3, sr);
        r4 = _mm256_mul_pd(r4, sr);
        r5 = _mm256_mul_pd(r5, sr);
        r6 = _mm256_mul_pd(r6, sr);
        r7 = _mm256_mul_pd(r7, sr);
        r8 = _mm256_mul_pd(r8, sr);
        r9 = _mm256_mul_pd(r9, sr);
        r10 = _mm256_mul_pd(r10, sr);
        r11 = _mm256_mul_pd(r11, sr);
        r12 = _mm256_mul_pd(r12, sr);
        _mm256_stream_pd(&b[j], r1);
        _mm256_stream_pd(&b[j+4], r2);
        _mm256_stream_pd(&b[j+8], r3);
        _mm256_stream_pd(&b[j+12], r4);
        _mm256_stream_pd(&b[j+16], r5);
        _mm256_stream_pd(&b[j+20], r6);
        _mm256_stream_pd(&b[j+24], r7);
        _mm256_stream_pd(&b[j+28], r8);
        _mm256_stream_pd(&b[j+32], r9);
        _mm256_stream_pd(&b[j+36], r10);
        _mm256_stream_pd(&b[j+40], r11);
        _mm256_stream_pd(&b[j+44], r12);
    }
#pragma omp parallel for num_threads(NUM_THREADS)
    for(j=MAX(0, STREAM_ARRAY_SIZE-48); j < STREAM_ARRAY_SIZE; j++)
        b[j] = c[j]*scalar;
}

void tuned_STREAM_Add()
{
	ssize_t j;
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 

#pragma omp parallel for num_threads(NUM_THREADS)
    for (j=0; j<STREAM_ARRAY_SIZE-32; j+=32){
        __m256d a1 = _mm256_load_pd(&a[j]);
        __m256d a2 = _mm256_load_pd(&a[j+4]);
        __m256d a3 = _mm256_load_pd(&a[j+8]);
        __m256d a4 = _mm256_load_pd(&a[j+12]);
        __m256d a5 = _mm256_load_pd(&a[j+16]);
        __m256d a6 = _mm256_load_pd(&a[j+20]);
        __m256d a7 = _mm256_load_pd(&a[j+24]);
        __m256d a8 = _mm256_load_pd(&a[j+28]);
        __m256d b1 = _mm256_load_pd(&b[j]);
        __m256d b2 = _mm256_load_pd(&b[j+4]);
        __m256d b3 = _mm256_load_pd(&b[j+8]);
        __m256d b4 = _mm256_load_pd(&b[j+12]);
        __m256d b5 = _mm256_load_pd(&b[j+16]);
        __m256d b6 = _mm256_load_pd(&b[j+20]);
        __m256d b7 = _mm256_load_pd(&b[j+24]);
        __m256d b8 = _mm256_load_pd(&b[j+28]);
        a1 = _mm256_add_pd(a1, b1);
        a2 = _mm256_add_pd(a2, b2);
        a3 = _mm256_add_pd(a3, b3);
        a4 = _mm256_add_pd(a4, b4);
        a5 = _mm256_add_pd(a5, b5);
        a6 = _mm256_add_pd(a6, b6);
        a7 = _mm256_add_pd(a7, b7);
        a8 = _mm256_add_pd(a8, b8);
        _mm256_stream_pd(&c[j], a1);
        _mm256_stream_pd(&c[j+4], a2);
        _mm256_stream_pd(&c[j+8], a3);
        _mm256_stream_pd(&c[j+12], a4);
        _mm256_stream_pd(&c[j+16], a5);
        _mm256_stream_pd(&c[j+20], a6);
        _mm256_stream_pd(&c[j+24], a7);
        _mm256_stream_pd(&c[j+28], a8);
    }
#pragma omp parallel for num_threads(NUM_THREADS)
    for(j=MAX(0, STREAM_ARRAY_SIZE-32); j < STREAM_ARRAY_SIZE; j++)
        c[j] = a[j]+b[j];
}

void tuned_STREAM_Triad(STREAM_TYPE scalar)
{
	ssize_t j;
#ifdef _OPENMP
omp_set_dynamic(0);
#endif 
    
    __m256d sr = _mm256_broadcast_sd(&scalar);
#pragma omp parallel for num_threads(NUM_THREADS)
    for (j=0; j<STREAM_ARRAY_SIZE-32; j+=32){
        __m256d b1 = _mm256_load_pd(&b[j]);
        __m256d b2 = _mm256_load_pd(&b[j+4]);
        __m256d b3 = _mm256_load_pd(&b[j+8]);
        __m256d b4 = _mm256_load_pd(&b[j+12]);
        __m256d b5 = _mm256_load_pd(&b[j+16]);
        __m256d b6 = _mm256_load_pd(&b[j+20]);
        __m256d b7 = _mm256_load_pd(&b[j+24]);
        __m256d b8 = _mm256_load_pd(&b[j+28]);
        __m256d c1 = _mm256_load_pd(&c[j]);
        __m256d c2 = _mm256_load_pd(&c[j+4]);
        __m256d c3 = _mm256_load_pd(&c[j+8]);
        __m256d c4 = _mm256_load_pd(&c[j+12]);
        __m256d c5 = _mm256_load_pd(&c[j+16]);
        __m256d c6 = _mm256_load_pd(&c[j+20]);
        __m256d c7 = _mm256_load_pd(&c[j+24]);
        __m256d c8 = _mm256_load_pd(&c[j+28]);
        b1 = _mm256_fmadd_pd(c1, sr, b1);
        b2 = _mm256_fmadd_pd(c2, sr, b2);
        b3 = _mm256_fmadd_pd(c3, sr, b3);
        b4 = _mm256_fmadd_pd(c4, sr, b4);
        b5 = _mm256_fmadd_pd(c5, sr, b5);
        b6 = _mm256_fmadd_pd(c6, sr, b6);
        b7 = _mm256_fmadd_pd(c7, sr, b7);
        b8 = _mm256_fmadd_pd(c8, sr, b8);
        _mm256_stream_pd(&a[j], b1);
        _mm256_stream_pd(&a[j+4], b2);
        _mm256_stream_pd(&a[j+8], b3);
        _mm256_stream_pd(&a[j+12], b4);
        _mm256_stream_pd(&a[j+16], b5);
        _mm256_stream_pd(&a[j+20], b6);
        _mm256_stream_pd(&a[j+24], b7);
        _mm256_stream_pd(&a[j+28], b8);
    }
#pragma omp parallel for num_threads(NUM_THREADS)
    for(j=MAX(0, STREAM_ARRAY_SIZE-32); j < STREAM_ARRAY_SIZE; j++)
        a[j] = b[j] + scalar*c[j];
}
/* end of stubs for the "tuned" versions of the kernels */
#endif
