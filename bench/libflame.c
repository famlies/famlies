/* libflame behind a narrow C interface, so that FLAME.h, a C header with
   macros of its own, stays out of the C++ harness. */

#include "libflame.h"

#include <FLAME.h>

static FLA_Datatype datatype(int type)
{
    switch (type)
    {
        case 0: return FLA_FLOAT;
        case 1: return FLA_DOUBLE;
        case 2: return FLA_COMPLEX;
        default: return FLA_DOUBLE_COMPLEX;
    }
}

void bench_libflame_init(void) { FLA_Init(); }

void bench_libflame_finalize(void) { FLA_Finalize(); }

int bench_libflame_chol(int type, int lower, long n, void* a, long rs, long cs)
{
    FLA_Obj A;
    FLA_Obj_create_without_buffer(datatype(type), n, n, &A);
    FLA_Obj_attach_buffer(a, rs, cs, &A);
    const FLA_Error e =
        FLA_Chol(lower ? FLA_LOWER_TRIANGULAR : FLA_UPPER_TRIANGULAR, A);
    FLA_Obj_free_without_buffer(&A);
    return e == FLA_SUCCESS;
}
