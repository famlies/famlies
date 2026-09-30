#include "support/pass.hpp"
#include "support/random.hpp"
#include "support/sentinel.hpp"
#include "support/storage.hpp"
#include "support/types.hpp"
#include "support/wide.hpp"

#include <famlies/blas/herk.hpp>
#include <famlies/leaf/reference.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <complex>
#include <optional>
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

template <class R> R beta_value(beta_kind b)
{ return b == beta_kind::zero ? R(0) : b == beta_kind::one ? R(1) : R(0.5); }

template <class F> void for_each_leaf(F&& f)
{
    f(blis{});
    f(reference{});
}

// herk with C m x m, stored in storage s, writing triangle `lower`, and A
// passed as p: an m x k matrix, or a vector of length m (RankA = 1).
template <class T, int RankA, class Leaf>
void check_matrix_c(Leaf leaf,
                    bool lower,
                    pass p,
                    storage s,
                    len_type m,
                    len_type k,
                    beta_kind bk)
{
    using R = famlies::real_t<T>;
    using W = wide_t<T>;
    auto rng = make_rng();
    const auto in_triangle = [&](len_type i, len_type j)
    { return lower ? i >= j : i <= j; };
    const auto other_triangle = [&](len_type i, len_type j)
    { return !in_triangle(i, j); };

    test_matrix<T> a(transposes(p) ? k : m, transposes(p) ? m : k, s);
    fill_random(a.view(), rng);
    test_matrix<T> c(m, m, s);
    if (bk != beta_kind::zero)
    {
        fill_random(c.view(), rng);
        fill_sentinel(c.view(), other_triangle);
    }
    const auto c0 = widen(c.view());
    const R alpha = R(-0.75);
    const R beta = beta_value<R>(bk);

    std::optional<MArray::marray<W, 2>> expected;
    std::optional<MArray::marray<long double, 2>> bound;
    with(lower,
         famlies::lower_triangle,
         famlies::upper_triangle,
         [&](auto tri)
         {
             const auto run = [&](auto&& A)
             {
                 famlies::herk(tri, alpha, A, beta, c.view(), leaf);
                 // What C should be, and |alpha|·|A|·|A|ᴴ + |beta|·|C|, which
                 // bounds its rounding.
                 const auto aw = widen(A);
                 expected.emplace(c0);
                 famlies::herk(tri, alpha, aw, beta, *expected, reference{});
                 bound.emplace(magnitude(c0));
                 const auto aw_bound = magnitude(aw);
                 famlies::herk(tri,
                               std::abs(alpha),
                               aw_bound,
                               std::abs(beta),
                               *bound,
                               reference{});
             };
             if constexpr (RankA == 1)
                 passed(p, a.view()[MArray::slice::all][0], run);
             else
                 passed(p, a.view(), run);
         });

    CHECK(a.gaps_intact());
    CHECK(c.gaps_intact());
    CHECK(sentinel_intact(c.view(), other_triangle));
    const long double tol = tolerance<T>(k);
    for (len_type i = 0; i < m; ++i)
        for (len_type j = 0; j < m; ++j)
            if (in_triangle(i, j))
            {
                CAPTURE(i, j);
                CHECK(within(c.view()[i][j],
                             (*expected)[i][j],
                             tol,
                             (*bound)[i][j]));
                if constexpr (famlies::is_complex_v<T>)
                    if (i == j)
                        CHECK(c.view()[i][j].imag() == 0);
            }
}

} // namespace

TEMPLATE_LIST_TEST_CASE("herk with a matrix A", "[herk]", element_types)
{
    using T = TestType;
    const bool lower = GENERATE(true, false);
    const pass p = GENERATE(from_range(passes));
    const storage s = GENERATE(from_range(storages));
    const len_type m = GENERATE(0, 1, 2, 17, 65);
    const len_type k = GENERATE(0, 1, 33);
    const beta_kind bk =
        GENERATE(beta_kind::zero, beta_kind::one, beta_kind::other);
    CAPTURE(lower, to_string(p), to_string(s), m, k, int(bk));
    for_each_leaf([&](auto leaf)
                  { check_matrix_c<T, 2>(leaf, lower, p, s, m, k, bk); });
}

TEMPLATE_LIST_TEST_CASE("herk with a vector A and a matrix C",
                        "[herk]",
                        element_types)
{
    using T = TestType;
    const bool lower = GENERATE(true, false);
    const pass p = GENERATE(from_range(vector_passes));
    const storage s = GENERATE(from_range(storages));
    const len_type m = GENERATE(0, 1, 2, 17, 65);
    const beta_kind bk =
        GENERATE(beta_kind::zero, beta_kind::one, beta_kind::other);
    CAPTURE(lower, to_string(p), to_string(s), m, int(bk));
    for_each_leaf([&](auto leaf)
                  { check_matrix_c<T, 1>(leaf, lower, p, s, m, 1, bk); });
}

TEMPLATE_LIST_TEST_CASE("herk with a vector A and an element C",
                        "[herk]",
                        element_types)
{
    using T = TestType;
    using R = famlies::real_t<T>;
    using W = wide_t<T>;
    const bool lower = GENERATE(true, false);
    const pass p = GENERATE(from_range(vector_passes));
    const storage s = GENERATE(from_range(storages));
    const len_type k = GENERATE(0, 1, 2, 17, 65);
    const beta_kind bk =
        GENERATE(beta_kind::zero, beta_kind::one, beta_kind::other);
    CAPTURE(lower, to_string(p), to_string(s), k, int(bk));

    for_each_leaf(
        [&](auto leaf)
        {
            auto rng = make_rng();
            test_vector<T> a(k, s);
            fill_random(a.view(), rng);
            test_matrix<T> c(1, 1, s);
            if (bk != beta_kind::zero)
                c.view()[0][0] = random_value<T>(rng);
            const T c0 = c.view()[0][0];
            const R alpha = R(-0.75);
            const R beta = beta_value<R>(bk);
            with(lower,
                 famlies::lower_triangle,
                 famlies::upper_triangle,
                 [&](auto tri)
                 {
                     passed(p,
                            a.view(),
                            [&](auto A)
                            {
                                famlies::herk(tri,
                                              alpha,
                                              A,
                                              beta,
                                              c.view()[0][0],
                                              leaf);
                            });
                 });

            // alpha·Σ|a|² + beta·Re(c).
            long double sum = 0;
            for (len_type i = 0; i < k; ++i)
                sum += magnitude(a.view()[i]) * magnitude(a.view()[i]);
            const W expected = bk == beta_kind::zero
                                 ? W(alpha * sum)
                                 : W(alpha * sum + beta * std::real(W(c0)));
            const long double bound =
                std::abs(alpha)
                * sum
                + (bk == beta_kind::zero ? 0 : beta * magnitude(c0));
            const T result = c.view()[0][0];
            CHECK(within(result, expected, tolerance<T>(k), bound));
            if constexpr (famlies::is_complex_v<T>)
                CHECK(result.imag() == 0);
            CHECK(a.gaps_intact());
            CHECK(c.gaps_intact());
        });
}

TEMPLATE_LIST_TEST_CASE("herk's blis leaf with thread counts",
                        "[herk][blis]",
                        element_types)
{
    using T = TestType;
    const blis leaf = GENERATE(blis{.threads = 4}, blis{.jc = 2, .ic = 2});
    CAPTURE(leaf.threads);
    check_matrix_c<T, 2>(leaf,
                         true,
                         pass::as_is,
                         storage::column_major,
                         300,
                         300,
                         beta_kind::other);
}

TEST_CASE("herk's select chooses a leaf from the operands", "[herk]")
{
    using T = std::complex<float>;
    std::vector<len_type> seen;
    const famlies::select tree{.when =
                                   [&](const auto& A, const auto& C)
                               {
                                   seen.push_back(A.length(1));
                                   seen.push_back(C.length(0));
                                   return true;
                               },
                               .then = reference{},
                               .otherwise = blis{}};
    auto rng = make_rng();
    test_matrix<T> a(3, 2, storage::row_major);
    fill_random(a.view(), rng);
    test_matrix<T> c(3, 3, storage::row_major);
    famlies::herk(famlies::lower_triangle, 1, a.view(), 0, c.view(), tree);
    CHECK(seen == std::vector<len_type>{2, 3});
    CHECK(sentinel_intact(c.view(), strictly_upper));
    CHECK(std::abs(c.view()[2][0]
                   - (a.view()[2][0]
                      * std::conj(a.view()[0][0])
                      + a.view()[2][1]
                      * std::conj(a.view()[0][1])))
          < 1e-5f);
}

TEST_CASE("herk's call forms", "[herk]")
{
    using T = std::complex<double>;
    auto rng = make_rng();
    MArray::marray<T, 2> a({4, 3}, T(0), MArray::COLUMN_MAJOR);
    fill_random(a.view(), rng);
    const auto run = [&](auto call)
    {
        MArray::marray<T, 2> c({4, 4}, T(1), MArray::COLUMN_MAJOR);
        call(c);
        CHECK(std::abs(c[3][1]
                       - (T(1)
                          + T(2)
                          * (a[3][0]
                             * std::conj(a[1][0])
                             + a[3][1]
                             * std::conj(a[1][1])
                             + a[3][2]
                             * std::conj(a[1][2]))))
              < 1e-14);
    };
    // alpha and beta convert from ints, and owning arrays pass as lvalues.
    using famlies::lower_triangle;
    run([&](auto& c) { famlies::herk(lower_triangle, 2, a, 1, c); });
    run([&](auto& c)
        { famlies::herk(lower_triangle, 2, a, 1, c, reference{}); });
    run([&](auto& c)
        { famlies::herk<blis{.jc = 2}>(lower_triangle, 2, a, 1, c); });
}

namespace
{

template <class A, class C, class Ctl>
constexpr bool herk_callable = requires(A&& a, C&& c, const Ctl& ctl) {
    famlies::herk(famlies::lower_triangle,
                  1,
                  std::forward<A>(a),
                  1,
                  std::forward<C>(c),
                  ctl);
};

// A select's `when`: one that takes herk's operands, and ones that take
// ints, too few operands, or return nothing.
constexpr auto always = [](const auto&, const auto&) { return true; };
constexpr auto on_ints = [](int, int) { return true; };
constexpr auto on_one = [](const auto&) { return true; };
constexpr auto returns_void = [](const auto&, const auto&) {};

} // namespace

TEST_CASE("herk rejects what it can't run", "[herk]")
{
    using Z = std::complex<double>;
    using M = MArray::marray_view<Z, 2>;
    using V = MArray::marray_view<Z, 1>;
    using CM = decltype(famlies::conjugated(std::declval<M>()));
    using CV = decltype(famlies::conjugated(std::declval<V>()));
    STATIC_CHECK(herk_callable<M, M, blis>);
    STATIC_CHECK(herk_callable<CM, M, blis>);
    STATIC_CHECK(herk_callable<CV, M, blis>);
    STATIC_CHECK(herk_callable<V, Z&, blis>);
    STATIC_CHECK(!herk_callable<M, CM, blis>);
    STATIC_CHECK(!herk_callable<M, MArray::marray_view<const Z, 2>, blis>);
    STATIC_CHECK(!herk_callable<V, const Z&, blis>);
    STATIC_CHECK(!herk_callable<M, Z&, blis>);
    STATIC_CHECK(!herk_callable<M, V, blis>);
    STATIC_CHECK(!herk_callable<Z&, Z&, blis>);
    STATIC_CHECK(!herk_callable<MArray::marray_view<float, 2>,
                                MArray::marray_view<double, 2>,
                                blis>);
    using L = MArray::marray_view<std::complex<long double>, 2>;
    STATIC_CHECK(!herk_callable<L, L, blis>);
    STATIC_CHECK(herk_callable<L, L, reference>);
    // A select whose `when` doesn't take the operands, or doesn't give a
    // bool, at the root or nested.
    using famlies::select;
    STATIC_CHECK(
        herk_callable<M, M, select<decltype(always), blis, reference>>);
    STATIC_CHECK(
        !herk_callable<M, M, select<decltype(on_ints), blis, reference>>);
    STATIC_CHECK(
        !herk_callable<M, M, select<decltype(on_one), blis, reference>>);
    STATIC_CHECK(
        !herk_callable<M, M, select<decltype(returns_void), blis, reference>>);
    STATIC_CHECK(
        !herk_callable<M,
                       M,
                       select<decltype(always),
                              select<decltype(on_ints), blis, reference>,
                              blis>>);
}
