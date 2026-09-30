// trsm is checked by its residual: a triangular solve is backward stable,
// so op(A)·X − alpha·B is small relative to |op(A)|·|X| + |alpha|·|B|
// whatever A's conditioning, where comparing X with the reference leaf's X
// would measure it.

#include "support/pass.hpp"
#include "support/random.hpp"
#include "support/sentinel.hpp"
#include "support/storage.hpp"
#include "support/types.hpp"
#include "support/wide.hpp"

#include <famlies/blas/gemm.hpp>
#include <famlies/blas/trsm.hpp>
#include <famlies/leaf/reference.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <complex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

using famlies::blis;
using famlies::len_type;
using famlies::reference;
using namespace famlies::test;

namespace
{

template <class T> T alpha_value(bool one)
{
    if (one)
        return T(1);
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

// op(A) as a wide matrix: its triangle, the other one zero, and a unit
// diagonal if the diagonal is implicit.
template <class A> auto wide_triangle(const A& a, bool lower, bool unit)
{
    auto w = widen(a);
    using W = typename decltype(w)::value_type;
    for (len_type i = 0; i < w.length(0); ++i)
        for (len_type j = 0; j < w.length(1); ++j)
            if (i == j ? unit : (lower ? i < j : i > j))
                w[i][j] = W(i == j ? 1 : 0);
    return w;
}

// Whether left ? M·X : X·M is alpha·B0 within tolerance<T>(k) of
// |M|·|X| + |alpha|·|B0|, per element, all as wide matrices.
template <class T, class W>
bool solves(bool left,
            const MArray::marray<W, 2>& m,
            const MArray::marray<W, 2>& x,
            T alpha,
            const MArray::marray<W, 2>& b0)
{
    MArray::marray<W, 2> r = b0;
    MArray::marray<long double, 2> bound = magnitude(b0);
    const auto mm = magnitude(m);
    const auto mx = magnitude(x);
    const long double ma = magnitude(alpha);
    if (left)
    {
        famlies::gemm(W(1), m, x, -W(alpha), r, reference{});
        famlies::gemm(1.0L, mm, mx, ma, bound, reference{});
    }
    else
    {
        famlies::gemm(W(1), x, m, -W(alpha), r, reference{});
        famlies::gemm(1.0L, mx, mm, ma, bound, reference{});
    }
    const long double tol = tolerance<T>(m.length(0));
    bool ok = true;
    for (len_type i = 0; i < r.length(0); ++i)
        for (len_type j = 0; j < r.length(1); ++j)
            if (!(std::abs(r[i][j]) <= tol * bound[i][j]))
                ok = false;
    return ok;
}

template <class W>
MArray::marray<W, 2> as_matrix(const MArray::marray<W, 1>& v, bool column)
{
    const len_type n = v.length(0);
    MArray::marray<W, 2> m({column ? n : 1, column ? 1 : n},
                           W(0),
                           MArray::COLUMN_MAJOR);
    for (len_type i = 0; i < n; ++i) (column ? m[i][0] : m[0][i]) = v[i];
    return m;
}

// trsm with an m x m A, stored in storage s and passed as p, whose triangle
// as passed is `lower`, and B a matrix (m x n on the left, n x m on the
// right) or a vector of length m.
template <class T, int RankB, class Leaf>
void check_matrix_a(Leaf leaf,
                    bool left,
                    bool lower,
                    bool unit,
                    pass p,
                    storage s,
                    len_type m,
                    len_type n,
                    bool alpha_one)
{
    auto rng = make_rng();
    const bool stored_lower = lower != transposes(p);
    test_matrix<T> a(m, m, s);
    fill_triangular(a.view(),
                    stored_lower ? strictly_lower : strictly_upper,
                    rng);
    if (unit)
        fill_sentinel(a.view(), on_diagonal);
    const auto a_before = snapshot(a.view());

    const len_type rows = RankB == 1 || left ? m : n;
    const len_type cols = RankB == 1 ? 1 : left ? n : m;
    test_matrix<T> b(rows, cols, s);
    const auto B = [&]
    {
        if constexpr (RankB == 1)
            return b.view()[MArray::slice::all][0];
        else
            return b.view();
    }();
    fill_random(B, rng);
    const auto b0 = widen(B);
    const T alpha = alpha_value<T>(alpha_one);

    std::optional<MArray::marray<wide_t<T>, 2>> op_a;
    with(left,
         famlies::left_side,
         famlies::right_side,
         [&](auto side)
         {
             with(lower,
                  famlies::lower_triangle,
                  famlies::upper_triangle,
                  [&](auto tri)
                  {
                      with(unit,
                           famlies::implicit_unit_diagonal,
                           famlies::explicit_diagonal,
                           [&](auto diag)
                           {
                               passed(p,
                                      a.view(),
                                      [&](auto A)
                                      {
                                          famlies::trsm(side,
                                                        tri,
                                                        diag,
                                                        alpha,
                                                        A,
                                                        B,
                                                        leaf);
                                          op_a.emplace(
                                              wide_triangle(A, lower, unit));
                                      });
                           });
                  });
         });

    CHECK(unchanged(a.view(), a_before));
    CHECK(a.gaps_intact());
    CHECK(b.gaps_intact());
    if constexpr (RankB == 1)
        CHECK(solves(left,
                     *op_a,
                     as_matrix(widen(B), left),
                     alpha,
                     as_matrix(b0, left)));
    else
        CHECK(solves(left, *op_a, widen(B), alpha, b0));
}

} // namespace

TEMPLATE_LIST_TEST_CASE("trsm with a matrix B", "[trsm]", element_types)
{
    using T = TestType;
    const bool left = GENERATE(true, false);
    const bool lower = GENERATE(true, false);
    const bool unit = GENERATE(false, true);
    const pass p = GENERATE(from_range(passes));
    const storage s = GENERATE(from_range(storages));
    const len_type m = GENERATE(0, 1, 2, 17, 65);
    const len_type n = GENERATE(0, 1, 33);
    const bool alpha_one = GENERATE(true, false);
    CAPTURE(left, lower, unit, to_string(p), to_string(s), m, n, alpha_one);
    for_each_leaf(
        [&](auto leaf)
        {
            check_matrix_a<T,
                           2>(leaf, left, lower, unit, p, s, m, n, alpha_one);
        });
}

TEMPLATE_LIST_TEST_CASE("trsm with a vector B", "[trsm]", element_types)
{
    using T = TestType;
    const bool left = GENERATE(true, false);
    const bool lower = GENERATE(true, false);
    const bool unit = GENERATE(false, true);
    const pass p = GENERATE(from_range(passes));
    const storage s = GENERATE(from_range(storages));
    const len_type m = GENERATE(0, 1, 2, 17, 65);
    const bool alpha_one = GENERATE(true, false);
    CAPTURE(left, lower, unit, to_string(p), to_string(s), m, alpha_one);
    for_each_leaf(
        [&](auto leaf)
        {
            check_matrix_a<T,
                           1>(leaf, left, lower, unit, p, s, m, 0, alpha_one);
        });
}

TEMPLATE_LIST_TEST_CASE("trsm with an element A divides a vector B",
                        "[trsm]",
                        element_types)
{
    using T = TestType;
    const bool left = GENERATE(true, false);
    const bool unit = GENERATE(false, true);
    const pass p = GENERATE(from_range(vector_passes));
    const storage s = GENERATE(from_range(storages));
    const len_type n = GENERATE(0, 1, 2, 17, 65);
    const bool alpha_one = GENERATE(true, false);
    CAPTURE(left, unit, to_string(p), to_string(s), n, alpha_one);

    for_each_leaf(
        [&](auto leaf)
        {
            auto rng = make_rng();
            test_matrix<T> a(1, 1, s);
            a.view()[0][0] = T(2) + random_value<T>(rng);
            if (unit)
                a.view()[0][0] = sentinel<T>();
            const auto a_before = snapshot(a.view());
            test_vector<T> b(n, s);
            fill_random(b.view(), rng);
            const auto b0 = widen(b.view());
            const T alpha = alpha_value<T>(alpha_one);

            using W = wide_t<T>;
            W op_a(1);
            with(left,
                 famlies::left_side,
                 famlies::right_side,
                 [&](auto side)
                 {
                     with(unit,
                          famlies::implicit_unit_diagonal,
                          famlies::explicit_diagonal,
                          [&](auto diag)
                          {
                              passed(p,
                                     a.view()[0][0],
                                     [&](auto&& A)
                                     {
                                         famlies::trsm(side,
                                                       famlies::lower_triangle,
                                                       diag,
                                                       alpha,
                                                       A,
                                                       b.view(),
                                                       leaf);
                                         if (!unit)
                                             op_a = widen(A);
                                     });
                          });
                 });

            CHECK(unchanged(a.view(), a_before));
            CHECK(b.gaps_intact());
            const auto x = widen(b.view());
            const long double tol = tolerance<T>(1);
            for (len_type i = 0; i < n; ++i)
                CHECK(std::abs(op_a * x[i] - W(alpha) * b0[i])
                      <= tol
                      * (magnitude(op_a)
                         * magnitude(x[i])
                         + magnitude(alpha)
                         * magnitude(b0[i])));
        });
}

TEMPLATE_LIST_TEST_CASE("trsm's blis leaf with thread counts",
                        "[trsm][blis]",
                        element_types)
{
    using T = TestType;
    const bool left = GENERATE(true, false);
    const blis leaf = GENERATE(blis{.threads = 4}, blis{.jc = 2, .ic = 2});
    CAPTURE(left, leaf.threads);
    check_matrix_a<T, 2>(leaf,
                         left,
                         true,
                         false,
                         pass::adjoint,
                         storage::column_major,
                         300,
                         300,
                         false);
}

TEST_CASE("trsm's select chooses a leaf from the operands", "[trsm]")
{
    using T = std::complex<double>;
    std::vector<len_type> seen;
    const famlies::select tree{.when =
                                   [&](const auto& A, const auto& B)
                               {
                                   seen.push_back(A.length(0));
                                   seen.push_back(B.length(1));
                                   return B.length(1) > 2;
                               },
                               .then = reference{},
                               .otherwise = blis{}};
    for (const len_type n : {2, 5})
    {
        auto rng = make_rng();
        test_matrix<T> a(3, 3, storage::general);
        fill_triangular(a.view(), strictly_lower, rng);
        test_matrix<T> b(3, n, storage::general);
        fill_random(b.view(), rng);
        const auto b0 = widen(b.view());
        famlies::trsm(famlies::left_side,
                      famlies::lower_triangle,
                      famlies::explicit_diagonal,
                      T(1),
                      a.view(),
                      b.view(),
                      tree);
        CHECK(solves(true,
                     wide_triangle(a.view(), true, false),
                     widen(b.view()),
                     T(1),
                     b0));
    }
    CHECK(seen == std::vector<len_type>{3, 2, 3, 5});
}

TEST_CASE("trsm's call forms", "[trsm]")
{
    using T = double;
    auto rng = make_rng();
    MArray::marray<T, 2> a({4, 4}, T(0), MArray::COLUMN_MAJOR);
    fill_triangular(a.view(), strictly_upper, rng);
    MArray::marray<T, 2> b({4, 3}, T(0), MArray::COLUMN_MAJOR);
    fill_random(b.view(), rng);
    const auto b0 = widen(b);
    const auto run = [&](auto call)
    {
        MArray::marray<T, 2> x = b;
        call(x);
        CHECK(solves(true, wide_triangle(a, false, false), widen(x), T(2), b0));
    };
    using famlies::explicit_diagonal;
    using famlies::left_side;
    using famlies::upper_triangle;
    // alpha converts from an int, and owning arrays pass as lvalues.
    run(
        [&](auto& x)
        {
            famlies::trsm(left_side,
                          upper_triangle,
                          explicit_diagonal,
                          2,
                          a,
                          x);
        });
    run(
        [&](auto& x)
        {
            famlies::trsm(left_side,
                          upper_triangle,
                          explicit_diagonal,
                          2,
                          a,
                          x,
                          reference{});
        });
    run(
        [&](auto& x)
        {
            famlies::trsm<blis{.threads = 2}>(left_side,
                                              upper_triangle,
                                              explicit_diagonal,
                                              2,
                                              a,
                                              x);
        });
}

namespace
{

template <class A, class B, class Ctl>
constexpr bool trsm_callable = requires(A&& a, B&& b, const Ctl& ctl) {
    famlies::trsm(famlies::left_side,
                  famlies::lower_triangle,
                  famlies::explicit_diagonal,
                  1,
                  std::forward<A>(a),
                  std::forward<B>(b),
                  ctl);
};

// A select's `when`: one that takes trsm's operands, and ones that take
// ints, too few operands, or return nothing.
constexpr auto always = [](const auto&, const auto&) { return true; };
constexpr auto on_ints = [](int, int) { return true; };
constexpr auto on_one = [](const auto&) { return true; };
constexpr auto returns_void = [](const auto&, const auto&) {};

} // namespace

TEST_CASE("trsm rejects what it can't run", "[trsm]")
{
    using Z = std::complex<double>;
    using M = MArray::marray_view<Z, 2>;
    using V = MArray::marray_view<Z, 1>;
    using CM = decltype(famlies::conjugated(std::declval<M>()));
    STATIC_CHECK(trsm_callable<M, M, blis>);
    STATIC_CHECK(trsm_callable<CM, V, blis>);
    STATIC_CHECK(trsm_callable<Z&, V, blis>);
    // A conjugated or const output.
    STATIC_CHECK(!trsm_callable<M, CM, blis>);
    STATIC_CHECK(!trsm_callable<M, MArray::marray_view<const Z, 2>, blis>);
    // Ranks trsm doesn't have, and mixed element types.
    STATIC_CHECK(!trsm_callable<V, M, blis>);
    STATIC_CHECK(!trsm_callable<Z&, M, blis>);
    STATIC_CHECK(!trsm_callable<M, Z&, blis>);
    STATIC_CHECK(!trsm_callable<M, MArray::marray_view<double, 2>, blis>);
    // A type the blis leaf doesn't take, which the reference leaf does,
    // anywhere in the tree.
    using L = MArray::marray_view<long double, 2>;
    STATIC_CHECK(!trsm_callable<L, L, blis>);
    STATIC_CHECK(trsm_callable<L, L, reference>);
    STATIC_CHECK(
        !trsm_callable<L,
                       L,
                       famlies::select<decltype(always), reference, blis>>);
    // A tree that isn't a leaf tree.
    STATIC_CHECK(!trsm_callable<M, M, int>);
    // A select whose `when` doesn't take the operands, or doesn't give a
    // bool, at the root or nested.
    using famlies::select;
    STATIC_CHECK(
        trsm_callable<M, M, select<decltype(always), blis, reference>>);
    STATIC_CHECK(
        !trsm_callable<M, M, select<decltype(on_ints), blis, reference>>);
    STATIC_CHECK(
        !trsm_callable<M, M, select<decltype(on_one), blis, reference>>);
    STATIC_CHECK(
        !trsm_callable<M, M, select<decltype(returns_void), blis, reference>>);
    STATIC_CHECK(
        !trsm_callable<M,
                       M,
                       select<decltype(always),
                              select<decltype(on_ints), blis, reference>,
                              blis>>);
}
