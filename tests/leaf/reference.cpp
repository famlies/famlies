#include "support/types.hpp"

#include <famlies/blas/gemm.hpp>
#include <famlies/blas/herk.hpp>
#include <famlies/blas/trsm.hpp>
#include <famlies/core/control.hpp>
#include <famlies/leaf/blis.hpp>
#include <famlies/leaf/reference.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>

using namespace famlies::test;

namespace
{

template <class T, class Leaf>
constexpr bool has_default_block_size =
    requires(const Leaf& leaf) { famlies::default_block_size<T>(leaf); };

} // namespace

TEMPLATE_LIST_TEST_CASE("the reference leaf has a default block size",
                        "[reference]",
                        element_types)
{ CHECK(famlies::default_block_size<TestType>(famlies::reference{}) > 0); }

TEST_CASE("leaves are leaves", "[reference][blis]")
{
    STATIC_CHECK(famlies::leaf<famlies::reference>);
    STATIC_CHECK(famlies::leaf<famlies::blis>);
    STATIC_CHECK(famlies::leaf<const famlies::blis&>);
    STATIC_CHECK(!famlies::leaf<int>);

    // The reference leaf takes any field type; the blis leaf, BLIS's four.
    STATIC_CHECK(has_default_block_size<long double, famlies::reference>);
    STATIC_CHECK(!has_default_block_size<long double, famlies::blis>);
    CHECK(famlies::default_block_size<long double>(famlies::reference{}) > 0);
}

// Hand-computed cases, exact in binary floating point, since the reference
// leaf is the oracle the other tests trust.

namespace
{

using Z = std::complex<double>;
constexpr Z i1{0, 1};

MArray::marray<Z, 2>
matrix(std::initializer_list<std::initializer_list<Z>> rows)
{
    const auto m = famlies::len_type(rows.size());
    const auto n = famlies::len_type(rows.begin()->size());
    MArray::marray<Z, 2> a({m, n}, Z(0), MArray::ROW_MAJOR);
    famlies::len_type i = 0;
    for (const auto& row : rows)
    {
        famlies::len_type j = 0;
        for (const Z& x : row) a[i][j++] = x;
        ++i;
    }
    return a;
}

MArray::marray<Z, 1> vector(std::initializer_list<Z> xs)
{
    MArray::marray<Z, 1> v({famlies::len_type(xs.size())});
    famlies::len_type i = 0;
    for (const Z& x : xs) v[i++] = x;
    return v;
}

const Z not_a_number{std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN()};

} // namespace

TEST_CASE("the reference leaf's trsm, by hand", "[reference][trsm]")
{
    using namespace famlies;
    const reference ref;
    // L = [2 0; 1+i 4] with a NaN where it mustn't be read, and X = [1; i].
    auto l = matrix({
        {2, not_a_number},
        {Z(1, 1), 4}
    });
    auto b = matrix({{2}, {Z(1, 5)}});
    trsm(left_side, lower_triangle, explicit_diagonal, 1, l, b, ref);
    CHECK(b[0][0] == Z(1));
    CHECK(b[1][0] == i1);

    // conj(L)·X, and X·Lᴴ = [1 i]·[2 1-i; 0 4] = [2, 1+3i], read through the
    // upper triangle of Lᴴ.
    b = matrix({{2}, {Z(1, 3)}});
    trsm(left_side,
         lower_triangle,
         explicit_diagonal,
         1,
         conjugated(l),
         b,
         ref);
    CHECK(b[1][0] == i1);
    auto row = matrix({
        {2, Z(1, 3)}
    });
    trsm(right_side,
         upper_triangle,
         explicit_diagonal,
         1,
         adjoint(l),
         row,
         ref);
    CHECK(row[0][0] == Z(1));
    CHECK(row[0][1] == i1);

    // A unit diagonal isn't read, and alpha scales B first: 2·[1; 1+2i].
    auto u = matrix({
        {not_a_number, not_a_number},
        {Z(1, 1), not_a_number}
    });
    auto x = vector({Z(1), Z(1, 2)});
    trsm(left_side, lower_triangle, implicit_unit_diagonal, 2, u, x, ref);
    CHECK(x[0] == Z(2));
    CHECK(x[1] == Z(0, 2));

    // An element A divides: 2·4i / conj(2i) = -4.
    Z a = Z(0, 2);
    auto y = vector({Z(0, 4)});
    trsm(left_side,
         lower_triangle,
         explicit_diagonal,
         2,
         conjugated(a),
         y,
         ref);
    CHECK(y[0] == Z(-4));
}

TEST_CASE("the reference leaf's herk, by hand", "[reference][herk]")
{
    using namespace famlies;
    const reference ref;
    // A·Aᴴ for A = [1+i 2; 0 i]: [6 -2i; 2i 1].
    const auto a = matrix({
        {Z(1, 1), 2},
        {0, i1}
    });
    auto c = matrix({
        {not_a_number, not_a_number},
        {not_a_number, not_a_number}
    });
    herk(lower_triangle, 1, a, 0, c, ref);
    CHECK(c[0][0] == Z(6));
    CHECK(c[1][0] == Z(0, 2));
    CHECK(c[1][1] == Z(1));
    CHECK(std::isnan(c[0][1].real()));

    // beta·C is added, and the diagonal's imaginary part cleared.
    c = matrix({
        {Z(1, 7), 0},
        {not_a_number, 0}
    });
    herk(upper_triangle, 1, a, 1, c, ref);
    CHECK(c[0][0] == Z(7));
    CHECK(c[0][1] == Z(0, -2));
    CHECK(c[1][1] == Z(1));

    // A vector A: a column with a matrix C, a row with an element.
    const auto v = vector({3, Z(0, 4)});
    c = matrix({
        {not_a_number, not_a_number},
        {not_a_number, not_a_number}
    });
    herk(lower_triangle, 1, v, 0, c, ref);
    CHECK(c[1][0] == Z(0, 12));
    CHECK(c[1][1] == Z(16));
    Z e = Z(1, 1);
    herk(lower_triangle, 2, v, 1, e, ref);
    CHECK(e == Z(51));
    e = not_a_number;
    herk(lower_triangle, 1, conjugated(v), 0, e, ref);
    CHECK(e == Z(25));
}

TEST_CASE("the reference leaf's gemm, by hand", "[reference][gemm]")
{
    using namespace famlies;
    const reference ref;
    const auto a = matrix({
        {1, i1},
        {2,  0}
    });
    const auto b = matrix({
        {i1, 1},
        { 1, 0}
    });
    auto c = matrix({
        {not_a_number, not_a_number},
        {not_a_number, not_a_number}
    });
    gemm(1, a, b, 0, c, ref);
    CHECK(c[0][0] == Z(0, 2));
    CHECK(c[0][1] == Z(1));
    CHECK(c[1][0] == Z(0, 2));
    CHECK(c[1][1] == Z(2));
    gemm(1, adjoint(a), b, 1, c, ref);
    CHECK(c[0][0] == Z(2, 3));
    CHECK(c[1][1] == Z(2, -1));

    // A matrix times a vector is a column; a vector times a matrix a row.
    const auto x = vector({i1, 1});
    auto y = vector({not_a_number, not_a_number});
    gemm(1, a, conjugated(x), 0, y, ref);
    CHECK(y[0] == Z(0));
    CHECK(y[1] == Z(0, -2));
    gemm(2, x, a, 1, y, ref);
    CHECK(y[0] == Z(4, 2));
    CHECK(y[1] == Z(-2, -2));

    // Two vectors give an element.
    Z e = not_a_number;
    gemm(1, x, x, 0, e, ref);
    CHECK(e == Z(0));
    gemm(1, conjugated(x), x, 1, e, ref);
    CHECK(e == Z(2));
}
