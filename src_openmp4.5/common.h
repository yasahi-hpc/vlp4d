#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#if defined(_OPENMP)
#include <omp.h>
#else
#define omp_get_max_threads() 0
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_set_dynamic(a) 0
#define omp_set_lock(a) 0
#define omp_unset_lock(a) 0
#define omp_init_lock(a) 0
#define omp_lock_t int
#endif
