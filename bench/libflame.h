#ifndef FAMLIES_BENCH_LIBFLAME_H
#define FAMLIES_BENCH_LIBFLAME_H

/* The baseline's calls. A type is 0 to 3 for float, double, and their
   complex types; an operand is a buffer and its row and column strides. */

#ifdef __cplusplus
extern "C" {
#endif

void bench_libflame_init(void);
void bench_libflame_finalize(void);

/* FLA_Chol with its default control tree. Returns whether it succeeded. */
int bench_libflame_chol(int type, int lower, long n, void* a, long rs, long cs);

#ifdef __cplusplus
}
#endif

#endif
