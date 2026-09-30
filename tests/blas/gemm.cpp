#include "support/pass.hpp"
#include "support/random.hpp"
#include "support/sentinel.hpp"
#include "support/storage.hpp"
#include "support/types.hpp"
#include "support/wide.hpp"

#include <famlies/blas/gemm.hpp>
#include <famlies/leaf/reference.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <complex>
#include <utility>
#include <vector>

using famlies::blis;
using famlies::len_type;
using famlies::reference;
using namespace famlies::test;

namespace
{

enum class beta_kind
{
    zero, // C starts as the sentinel, and mustn't be read
    one,
    other,
};

template <class T> T beta_value(beta_kind b)
{
    if (b == beta_kind::zero)
        return T(0);
    if (b == beta_kind::one)
        return T(1);
    if constexpr (famlies::is_complex_v<T>)
        return T(0.5, -0.25);
    else
        return T(0.5);
}

template <class T> T alpha_value()
{
    if constexpr (famlies::is_complex_v<T>)
        return T(-0.75, 0.5);
    else
        return T(-0.75);
}

template <class F> void for_each_leaf(F&& f)
{
    f(blis{});
    f(reference{});
}

// f(the operand of rank N that a test passes from matrix x: x itself, its
// one column, or its one element).
template <int N, class T, class F>
void operand_of(const test_matrix<T>& x, pass p, F&& f)
{
    if constexpr (N == 2)
        passed(p, x.view(), f);
    else if constexpr (N == 1)
        passed(p, x.view()[MArray::slice::all][0], f);
    else
        passed(p, x.view()[0][0], f);
}

// Operand x in the wide type as a matrix: a vector as a row or a column.
template <class X> auto wide_matrix_of(const X& x, bool row)
{
    auto w = widen(x);
    if constexpr (famlies::rank_v<X> == 2)
        return w;
    else
    {
        using W = typename decltype(w)::value_type;
        const len_type len = w.length(0);
        MArray::marray<W, 2> out({row ? 1 : len, row ? len : 1},
                                 W(0),
                                 MArray::COLUMN_MAJOR);
        for (len_type i = 0; i < len; ++i) (row ? out[0][i] : out[i][0]) = w[i];
        return out;
    }
}

// x's elements in column order.
template <class X> auto flatten(const X& x)
{
    std::vector<std::remove_cvref_t<decltype(x[0][0])>> out;
    for (len_type j = 0; j < x.length(1); ++j)
        for (len_type i = 0; i < x.length(0); ++i) out.push_back(x[i][j]);
    return out;
}

// C := alpha·op(A)·op(B) + beta·C with A of rank NA, B of rank NB and C of
// rank NC: an m x k by k x n product, where a vector A is a row of length k
// and a vector B a column of length k.
template <class T, int NA, int NB, int NC, class Leaf>
void check(Leaf leaf,
           pass pa,
           pass pb,
           storage s,
           len_type m,
           len_type k,
           len_type n,
           beta_kind bk)
{
    using W = wide_t<T>;
    auto rng = make_rng();
    // Vectors are stored as columns, and elements as 1 x 1 matrices. A
    // transposed matrix is stored transposed.
    const auto stored = [](int rank, pass p, len_type rows, len_type cols)
    {
        if (rank == 2)
            return transposes(p) ? std::pair{cols, rows}
                                 : std::pair{rows, cols};
        return std::pair{rank == 1 ? rows * cols : 1, len_type(1)};
    };
    const auto [am, an] = stored(NA, pa, NA == 2 ? m : 1, k);
    const auto [bm, bn] = stored(NB, pb, k, NB == 2 ? n : 1);
    const auto [cm, cn] =
        stored(NC, pass::as_is, NA == 2 ? m : 1, NB == 2 ? n : 1);
    test_matrix<T> a(am, an, s), b(bm, bn, s), c(cm, cn, s);
    fill_random(a.view(), rng);
    fill_random(b.view(), rng);
    if (bk != beta_kind::zero)
        fill_random(c.view(), rng);
    const auto c0 = flatten(widen(c.view()));
    const T alpha = alpha_value<T>();
    const T beta = beta_value<T>(bk);

    // The product and |op(A)|·|op(B)| in the wide type, m x n.
    MArray::marray<W, 2> product;
    MArray::marray<long double, 2> product_bound;
    operand_of<NA>(
        a,
        pa,
        [&](auto&& A)
        {
            operand_of<NB>(
                b,
                pb,
                [&](auto&& B)
                {
                    if constexpr (NC == 2)
                        famlies::gemm(alpha, A, B, beta, c.view(), leaf);
                    else if constexpr (NC == 1)
                        famlies::gemm(alpha,
                                      A,
                                      B,
                                      beta,
                                      c.view()[MArray::slice::all][0],
                                      leaf);
                    else
                        famlies::gemm(alpha, A, B, beta, c.view()[0][0], leaf);
                    const auto aw = wide_matrix_of(A, true);
                    const auto bw = wide_matrix_of(B, false);
                    product.reset({aw.length(0), bw.length(1)},
                                  W(0),
                                  MArray::COLUMN_MAJOR);
                    product_bound.reset({aw.length(0), bw.length(1)},
                                        0.0L,
                                        MArray::COLUMN_MAJOR);
                    famlies::gemm(W(1), aw, bw, W(0), product, reference{});
                    const auto aw_bound = magnitude(aw);
                    const auto bw_bound = magnitude(bw);
                    famlies::gemm(1.0L,
                                  aw_bound,
                                  bw_bound,
                                  0.0L,
                                  product_bound,
                                  reference{});
                });
        });

    CHECK(a.gaps_intact());
    CHECK(b.gaps_intact());
    CHECK(c.gaps_intact());
    // C and the product hold the same elements in column order.
    const auto result = flatten(c.view());
    const auto p = flatten(product);
    const auto pb_ = flatten(product_bound);
    REQUIRE(result.size() == p.size());
    const long double tol = tolerance<T>(k);
    for (std::size_t e = 0; e < p.size(); ++e)
    {
        const bool reads_c = bk != beta_kind::zero;
        const W expected = W(alpha) * p[e] + (reads_c ? W(beta) * c0[e] : W(0));
        const long double bound =
            magnitude(alpha)
            * pb_[e]
            + (reads_c ? magnitude(beta) * magnitude(c0[e]) : 0);
        CAPTURE(e);
        CHECK(within(result[e], expected, tol, bound));
    }
}

} // namespace

TEMPLATE_LIST_TEST_CASE("gemm with three matrices", "[gemm]", element_types)
{
    using T = TestType;
    const pass pa = GENERATE(from_range(passes));
    const pass pb = GENERATE(from_range(passes));
    const storage s = GENERATE(from_range(storages));
    const len_type m = GENERATE(0, 2, 33);
    const len_type k = GENERATE(0, 1, 17);
    const len_type n = GENERATE(1, 17);
    const beta_kind bk =
        GENERATE(beta_kind::zero, beta_kind::one, beta_kind::other);
    CAPTURE(to_string(pa), to_string(pb), to_string(s), m, k, n, int(bk));
    for_each_leaf([&](auto leaf)
                  { check<T, 2, 2, 2>(leaf, pa, pb, s, m, k, n, bk); });
}

TEMPLATE_LIST_TEST_CASE("gemm with a matrix and a vector",
                        "[gemm]",
                        element_types)
{
    using T = TestType;
    const bool matrix_first = GENERATE(true, false);
    const pass pm = GENERATE(from_range(passes));
    const pass pv = GENERATE(from_range(vector_passes));
    const storage s = GENERATE(from_range(storages));
    const len_type m = GENERATE(0, 1, 2, 33);
    const len_type k = GENERATE(0, 1, 17);
    const beta_kind bk =
        GENERATE(beta_kind::zero, beta_kind::one, beta_kind::other);
    CAPTURE(matrix_first,
            to_string(pm),
            to_string(pv),
            to_string(s),
            m,
            k,
            int(bk));
    for_each_leaf(
        [&](auto leaf)
        {
            if (matrix_first)
                check<T, 2, 1, 1>(leaf, pm, pv, s, m, k, 1, bk);
            else
                check<T, 1, 2, 1>(leaf, pv, pm, s, 1, k, m, bk);
        });
}

TEMPLATE_LIST_TEST_CASE("gemm with two vectors and an element C",
                        "[gemm]",
                        element_types)
{
    using T = TestType;
    const pass pa = GENERATE(from_range(vector_passes));
    const pass pb = GENERATE(from_range(vector_passes));
    const storage s = GENERATE(from_range(storages));
    const len_type k = GENERATE(0, 1, 2, 17, 65);
    const beta_kind bk =
        GENERATE(beta_kind::zero, beta_kind::one, beta_kind::other);
    CAPTURE(to_string(pa), to_string(pb), to_string(s), k, int(bk));
    for_each_leaf([&](auto leaf)
                  { check<T, 1, 1, 0>(leaf, pa, pb, s, 1, k, 1, bk); });
}

TEMPLATE_LIST_TEST_CASE("gemm's blis leaf with thread counts",
                        "[gemm][blis]",
                        element_types)
{
    using T = TestType;
    const blis leaf = GENERATE(blis{.threads = 4}, blis{.jc = 2, .ic = 2});
    CAPTURE(leaf.threads);
    check<T, 2, 2, 2>(leaf,
                      pass::as_is,
                      pass::adjoint,
                      storage::column_major,
                      300,
                      300,
                      300,
                      beta_kind::other);
}

TEST_CASE("gemm's select chooses a leaf from the operands", "[gemm]")
{
    using T = double;
    std::vector<len_type> seen;
    const famlies::select tree{
        .when =
            [&](const auto& A, const auto& B, const auto& C)
        {
            seen.push_back(A.length(0));
            seen.push_back(B.length(0));
            seen.push_back(C.length(0));
            return false;
        },
        .then = blis{},
        .otherwise = reference{}};
    test_matrix<T> a(2, 3, storage::general), b(3, 1, storage::general),
        c(2, 1, storage::general);
    auto rng = make_rng();
    fill_random(a.view(), rng);
    fill_random(b.view(), rng);
    const auto B = b.view()[MArray::slice::all][0];
    const auto C = c.view()[MArray::slice::all][0];
    famlies::gemm(1, a.view(), B, 0, C, tree);
    CHECK(seen == std::vector<len_type>{2, 3, 2});
    const auto A = a.view();
    CHECK(std::abs(C[1] - (A[1][0] * B[0] + A[1][1] * B[1] + A[1][2] * B[2]))
          < 1e-14);
}

TEST_CASE("gemm's call forms", "[gemm]")
{
    using T = std::complex<float>;
    auto rng = make_rng();
    MArray::marray<T, 2> a({3, 2}, T(0), MArray::COLUMN_MAJOR);
    MArray::marray<T, 2> b({2, 4}, T(0), MArray::COLUMN_MAJOR);
    fill_random(a.view(), rng);
    fill_random(b.view(), rng);
    const T expected = T(2) * (a[2][0] * b[0][3] + a[2][1] * b[1][3]) + T(1);
    const auto run = [&](auto call)
    {
        MArray::marray<T, 2> c({3, 4}, T(1), MArray::COLUMN_MAJOR);
        call(c);
        CHECK(std::abs(c[2][3] - expected) < 1e-5f);
    };
    // Scalars convert from ints, and owning arrays pass as lvalues.
    run([&](auto& c) { famlies::gemm(2, a, b, 1, c); });
    run([&](auto& c) { famlies::gemm(2, a, b, 1, c, reference{}); });
    run([&](auto& c) { famlies::gemm<blis{.threads = 2}>(2, a, b, 1, c); });
}

namespace
{

template <class A, class B, class C, class Ctl>
constexpr bool gemm_callable = requires(A&& a, B&& b, C&& c, const Ctl& ctl) {
    famlies::gemm(1,
                  std::forward<A>(a),
                  std::forward<B>(b),
                  1,
                  std::forward<C>(c),
                  ctl);
};

// A select's `when`: one that takes gemm's operands, and ones that take
// ints, too few operands, or return nothing.
constexpr auto always = [](const auto&, const auto&, const auto&)
{ return true; };
constexpr auto on_ints = [](int, int, int) { return true; };
constexpr auto on_two = [](const auto&, const auto&) { return true; };
constexpr auto returns_void = [](const auto&, const auto&, const auto&) {};

} // namespace

TEST_CASE("gemm rejects what it can't run", "[gemm]")
{
    using Z = std::complex<double>;
    using M = MArray::marray_view<Z, 2>;
    using V = MArray::marray_view<Z, 1>;
    using CM = decltype(famlies::conjugated(std::declval<M>()));
    using CV = decltype(famlies::conjugated(std::declval<V>()));
    STATIC_CHECK(gemm_callable<CM, M, M, blis>);
    STATIC_CHECK(gemm_callable<M, CV, V, blis>);
    STATIC_CHECK(gemm_callable<CV, M, V, blis>);
    STATIC_CHECK(gemm_callable<V, CV, Z&, blis>);
    STATIC_CHECK(!gemm_callable<M, M, CM, blis>);
    STATIC_CHECK(!gemm_callable<V, V, const Z&, blis>);
    STATIC_CHECK(!gemm_callable<V, V, M, blis>);
    STATIC_CHECK(!gemm_callable<V, V, V, blis>);
    STATIC_CHECK(!gemm_callable<M, M, V, blis>);
    STATIC_CHECK(!gemm_callable<M, V, M, blis>);
    STATIC_CHECK(!gemm_callable<M, M, MArray::marray_view<float, 2>, blis>);
    using L = MArray::marray_view<long double, 2>;
    STATIC_CHECK(!gemm_callable<L, L, L, blis>);
    STATIC_CHECK(gemm_callable<L, L, L, reference>);
    // A select whose `when` doesn't take the operands, or doesn't give a
    // bool, at the root or nested.
    using famlies::select;
    STATIC_CHECK(
        gemm_callable<M, M, M, select<decltype(always), blis, reference>>);
    STATIC_CHECK(
        !gemm_callable<M, M, M, select<decltype(on_ints), blis, reference>>);
    STATIC_CHECK(
        !gemm_callable<M, M, M, select<decltype(on_two), blis, reference>>);
    STATIC_CHECK(
        !gemm_callable<M,
                       M,
                       M,
                       select<decltype(returns_void), blis, reference>>);
    STATIC_CHECK(
        !gemm_callable<M,
                       M,
                       M,
                       select<decltype(always),
                              select<decltype(on_ints), blis, reference>,
                              blis>>);
}
