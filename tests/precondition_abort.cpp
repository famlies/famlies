// precondition_abort <op>.<case>: makes a call that breaks a precondition.
// Aborting exits 0, and getting past the call returns 1.

#include <famlies/blas/gemm.hpp>
#include <famlies/blas/herk.hpp>
#include <famlies/blas/trsm.hpp>
#include <famlies/core/sweep.hpp>
#include <famlies/dec/chol.hpp>
#include <famlies/leaf/blis.hpp>

#include <marray/marray.hpp>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string_view>
#include <utility>

extern "C" void exit_on_abort(int)
{ std::_Exit(0); }

namespace
{

void run_sweep(famlies::len_type n, famlies::len_type bs)
{
    for ([[maybe_unused]] auto step : famlies::sweep(n, bs, famlies::forward))
    {
    }
}

using matrix = MArray::marray<double, 2>;

void run_trsm(matrix a, matrix b)
{
    famlies::trsm(famlies::left_side,
                  famlies::lower_triangle,
                  famlies::explicit_diagonal,
                  1,
                  a,
                  b);
}

void run_herk(matrix a, matrix c)
{ famlies::herk(famlies::lower_triangle, 1, a, 0, c); }

void run_gemm(matrix a, matrix b, matrix c)
{ famlies::gemm(1, a, b, 0, c); }

template <class Tree> void run_chol(matrix a, const Tree& tree)
{ static_cast<void>(famlies::chol(a, famlies::lower_triangle, tree)); }

matrix identity(famlies::len_type n)
{
    matrix a({n, n}, 0.0);
    for (famlies::len_type i = 0; i < n; ++i) a[i][i] = 1;
    return a;
}

const std::pair<std::string_view, std::function<void()>> cases[] = {
    {   "sweep.negative_length",[] { run_sweep(-1, 4); }                                },
    {   "sweep.zero_block_size",                          [] { run_sweep(4, 0); }},
    {   "blis.negative_threads",
     []
     { famlies::detail::blis_runtime("blis", famlies::blis{.threads = -1}); }    },
    {   "blis.threads_and_ways",
     []
     {
     famlies::detail::blis_runtime("blis",
     famlies::blis{.threads = 2, .jc = 2});
     }                                                                           },
    {         "trsm.not_square", [] { run_trsm(matrix({3, 2}), matrix({3, 3})); }},
    {            "trsm.lengths",    [] { run_trsm(identity(3), matrix({2, 2})); }},
    {         "herk.not_square", [] { run_herk(matrix({2, 2}), matrix({2, 3})); }},
    {            "herk.lengths", [] { run_herk(matrix({3, 2}), matrix({2, 2})); }},
    {            "gemm.lengths",
     [] { run_gemm(matrix({2, 3}), matrix({2, 2}), matrix({2, 2})); }            },
    {         "chol.not_square",
     [] { run_chol(matrix({3, 2}), famlies::chol_default); }                     },
    {"chol.negative_block_size",
     []
     {
     run_chol(
     identity(3),
     famlies::chol_blocked{.variant = famlies::chol_right_looking{}, .nb = -1});
     }                                                                           },
    {     "chol.recursive_cutoff",
     [] { run_chol(identity(3), famlies::chol_recursive{.cutoff = 0}); }           },
};

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fputs("usage: precondition_abort <op>.<case>\n", stderr);
        return 2;
    }
    std::signal(SIGABRT, exit_on_abort);
    for (const auto& [name, run] : cases)
        if (name == argv[1])
        {
            run();
            return 1;
        }
    std::fprintf(stderr, "precondition_abort: no case %s\n", argv[1]);
    return 2;
}
