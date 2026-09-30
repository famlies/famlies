// What a failed check prints. That it aborts is precondition_abort's to
// test.

#include <famlies/core/conj.hpp>
#include <famlies/core/precondition.hpp>

#include <catch2/catch_test_macros.hpp>
#include <marray/marray.hpp>

#include <complex>
#include <string>

using famlies::detail::precondition_message;

TEST_CASE("a failed precondition names the op, condition and arguments",
          "[precondition]")
{
    using T = std::complex<double>;
    MArray::marray<T, 2> a({3, 4}, T(0));
    MArray::marray_view<T, 2> A = a;
    const auto R = MArray::range(0, 2);
    const famlies::len_type n = 5;

    const std::string message = precondition_message(
        "chol",
        "A.length(0) == A.length(1)",
        "A, adjoint(A[R][R]), A[1][R], A[0][0], n, f(1, 2)",
        A,
        famlies::adjoint(A[R][R]),
        A[1][R],
        A[0][0],
        n,
        7);
    CHECK(message
          == R"(famlies::chol: precondition failed: A.length(0) == A.length(1)
  A: 3 x 4
  adjoint(A[R][R]): 2 x 2
  A[1][R]: 2
  A[0][0]: element
  n = 5
  f(1, 2) = 7
)");

    CHECK(precondition_message("sweep", "n >= 0", "")
          == "famlies::sweep: precondition failed: n >= 0\n");
}
