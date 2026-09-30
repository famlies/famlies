#include "support/random.hpp"
#include "support/sentinel.hpp"
#include "support/storage.hpp"
#include "support/types.hpp"
#include "support/wide.hpp"

#include <famlies/dec/chol.hpp>
#include <famlies/leaf/reference.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <expected>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using famlies::blis;
using famlies::chol_blocked;
using famlies::chol_bordered;
using famlies::chol_left_looking;
using famlies::chol_recursive;
using famlies::chol_right_looking;
using famlies::chol_unblocked;
using famlies::len_type;
using famlies::not_positive_definite;
using famlies::reference;
using famlies::select;
using namespace famlies::test;

namespace
{

constexpr auto small = [](const auto& A) { return A.length(0) <= 16; };

// Trees covering every variant: unblocked, blocked at small nb (several
// blocks), recursive, nested combinations both ways, select at the root and
// between leaves, the reference leaf throughout, and the default tree.
const auto trees = std::tuple{
    chol_unblocked{.variant = chol_bordered{}},
    chol_unblocked{.variant = chol_left_looking{}},
    chol_unblocked{.variant = chol_right_looking{}},
    chol_blocked{.variant = chol_bordered{}, .nb = 4},
    chol_blocked{.variant = chol_left_looking{}, .nb = 4},
    chol_blocked{.variant = chol_right_looking{}, .nb = 4},
    chol_recursive{.cutoff = 4},
    chol_blocked{
                   .variant = chol_bordered{},
                   .nb = 16,
                   .chol = chol_blocked{
                       .variant = chol_left_looking{},
                       .nb = 5,
                       .chol =
                           chol_unblocked{.variant = chol_right_looking{}}}},
    chol_blocked{
                   .variant = chol_left_looking{},
                   .nb = 24,
                   .chol = chol_recursive{
                       .cutoff = 3,
                       .chol = chol_unblocked{.variant = chol_bordered{}}}},
    chol_recursive{
                   .cutoff = 8,
                   .chol = chol_blocked{
                       .variant = chol_right_looking{},
                       .nb = 3,
                       .chol = chol_unblocked{.variant = chol_bordered{}}}},
    select{
                   .when = small,
                   .then = chol_unblocked{.variant = chol_left_looking{}},
                   .otherwise =
            chol_blocked{
                .variant = chol_bordered{},
                .nb = 20,
                .chol = select{
                    .when = small,
                    .then = chol_unblocked{.variant = chol_right_looking{}},
                    .otherwise = chol_blocked{.variant = chol_left_looking{},
                                              .nb = 6}}}},
    chol_blocked{.variant = chol_left_looking{},
                   .nb = 5,
                   .chol = chol_unblocked{.variant = chol_left_looking{},
                                        .chol = select{.when = [](const auto&)
                                                       { return true; },
                                                       .then = reference{},
                                                       .otherwise = blis{}},
                                        .gemm = reference{}},
                   .trsm = select{.when = [](const auto&, const auto& B)
                                { return B.length(0) < 10; },
                                .then = reference{},
                                .otherwise = blis{}},
                   .herk = select{.when = [](const auto& A, const auto&)
                                { return A.length(1) > 10; },
                                .then = reference{},
                                .otherwise = blis{}}},
    chol_blocked{.variant = chol_right_looking{},
                   .nb = 7,
                   .chol = chol_unblocked{.variant = chol_bordered{},
                                        .chol = reference{},
                                        .trsm = reference{},
                                        .herk = reference{},
                                        .gemm = reference{}},
                   .trsm = reference{},
                   .herk = reference{},
                   .gemm = reference{}},
    famlies::chol_default,
};

constexpr std::size_t tree_count = std::tuple_size_v<decltype(trees)>;

const char* const tree_names[] = {
    "unblocked bordered",
    "unblocked left-looking",
    "unblocked right-looking",
    "blocked bordered nb=4",
    "blocked left-looking nb=4",
    "blocked right-looking nb=4",
    "recursive cutoff=4",
    "blocked bordered over blocked left-looking",
    "blocked left-looking over recursive",
    "recursive over blocked right-looking",
    "select at the root and below",
    "select between leaves",
    "reference leaves",
    "default",
};
static_assert(std::size(tree_names) == tree_count);

// f(the tree at index i).
template <class F> void with_tree(std::size_t i, F&& f)
{
    [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        ((I == i ? f(std::get<I>(trees)) : void()), ...);
    }(std::make_index_sequence<tree_count>{});
}

template <class F>
std::expected<void, not_positive_definite> with_triangle(bool lower, F&& f)
{ return lower ? f(famlies::lower_triangle) : f(famlies::upper_triangle); }

// Copies the triangle of the Hermitian h into a, leaving the other one as
// it is.
template <class T>
void copy_triangle(const MArray::marray<T, 2>& h,
                   MArray::marray_view<T, 2> a,
                   bool lower)
{
    for (len_type i = 0; i < h.length(0); ++i)
        for (len_type j = 0; j < h.length(1); ++j)
            if (lower ? i >= j : i <= j)
                a[i][j] = h[i][j];
}

// ‖H − L·Lᴴ‖_F / (n·‖H‖_F·ε), for the factor chol left in a's triangle.
template <class T>
long double
residual(const MArray::marray<T, 2>& h, MArray::marray_view<T, 2> a, bool lower)
{
    using W = wide_t<T>;
    const len_type n = h.length(0);
    if (n == 0)
        return 0;
    MArray::marray<W, 2> l({n, n}, W(0), MArray::COLUMN_MAJOR);
    for (len_type i = 0; i < n; ++i)
        for (len_type j = 0; j <= i; ++j)
            l[i][j] =
                W(lower ? a[i][j] : famlies::detail::conjugate(T(a[j][i])));
    auto e = widen(h);
    famlies::herk(famlies::lower_triangle, -1, l, 1, e, reference{});
    // Over the lower triangle, counting what's below the diagonal twice.
    long double e2 = 0, h2 = 0;
    for (len_type i = 0; i < n; ++i)
        for (len_type j = 0; j <= i; ++j)
        {
            const long double w = i == j ? 1 : 2;
            e2 += w * magnitude(e[i][j]) * magnitude(e[i][j]);
            h2 += w * magnitude(h[i][j]) * magnitude(h[i][j]);
        }
    return std::sqrt(e2)
         / (static_cast<long double>(n) * std::sqrt(h2) * eps<T>);
}

template <class T, class Tree>
void check_factors(const Tree& tree, bool lower, storage s, len_type n)
{
    auto rng = make_rng();
    const auto h = random_hpd<T>(n, rng);
    test_matrix<T> a(n, n, s);
    copy_triangle(h, a.view(), lower);

    const auto r = with_triangle(
        lower,
        [&](auto tri) { return famlies::chol(a.view(), tri, tree); });

    REQUIRE(r.has_value());
    CHECK(sentinel_intact(a.view(), lower ? strictly_upper : strictly_lower));
    CHECK(a.gaps_intact());
    if constexpr (famlies::is_complex_v<T>)
        for (len_type i = 0; i < n; ++i) CHECK(a.view()[i][i].imag() == 0);
    CHECK(residual(h, a.view(), lower) < 30);
}

} // namespace

TEMPLATE_LIST_TEST_CASE("chol factors a Hermitian positive definite matrix",
                        "[chol]",
                        element_types)
{
    using T = TestType;
    const std::size_t t = GENERATE(range(std::size_t(0), tree_count));
    const bool lower = GENERATE(true, false);
    const storage s = GENERATE(from_range(storages));
    const len_type n = GENERATE(0, 1, 2, 3, 7, 16, 17, 65, 130);
    CAPTURE(tree_names[t], lower, to_string(s), n);
    with_tree(t,
              [&](const auto& tree) { check_factors<T>(tree, lower, s, n); });
}

TEMPLATE_LIST_TEST_CASE("chol at n = 300", "[chol]", element_types)
{
    using T = TestType;
    // Every variant blocked, with the default tree and recursion.
    const std::size_t t = GENERATE(as<std::size_t>{}, 3, 4, 5, 6, 13);
    const bool lower = GENERATE(true, false);
    const storage s = GENERATE(from_range(storages));
    CAPTURE(tree_names[t], lower, to_string(s));
    with_tree(t,
              [&](const auto& tree) { check_factors<T>(tree, lower, s, 300); });
}

TEMPLATE_LIST_TEST_CASE("chol's blis leaves with thread counts",
                        "[chol][blis]",
                        element_types)
{
    using T = TestType;
    const bool lower = GENERATE(true, false);
    CAPTURE(lower);
    const chol_blocked tree{
        .variant = chol_right_looking{},
        .nb = 64,
        .trsm = blis{.threads = 4},
        .herk = blis{.jc = 2, .ic = 2}
    };
    check_factors<T>(tree, lower, storage::column_major, 300);
}

TEST_CASE("chol runs the reference leaf on any field type", "[chol]")
{
    const chol_blocked tree{
        .variant = chol_left_looking{},
        .nb = 5,
        .chol = chol_unblocked{.variant = chol_right_looking{},
                               .chol = reference{},
                               .trsm = reference{},
                               .herk = reference{},
                               .gemm = reference{}},
        .trsm = reference{},
        .herk = reference{},
        .gemm = reference{}
    };
    const bool lower = GENERATE(true, false);
    const len_type n = GENERATE(0, 1, 17, 40);
    CAPTURE(lower, n);
    const auto check = [&]<class T>(T)
    {
        auto rng = make_rng();
        const auto h = random_hpd<T>(n, rng);
        MArray::marray<T, 2> a({n, n}, T(0), MArray::ROW_MAJOR);
        copy_triangle(h, a.view(), lower);
        const auto r = with_triangle(lower,
                                     [&](auto tri)
                                     { return famlies::chol(a, tri, tree); });
        REQUIRE(r.has_value());
        CHECK(residual(h, a.view(), lower) < 30);
    };
    check(0.0L);
    check(std::complex<long double>());
}

TEMPLATE_LIST_TEST_CASE(
    "chol reports the first leading minor that isn't positive definite",
    "[chol]",
    element_types)
{
    using T = TestType;
    const std::size_t t = GENERATE(range(std::size_t(0), tree_count));
    const bool lower = GENERATE(true, false);
    // Positive definite through order 17 and not at 18, and at the ends.
    const len_type order = GENERATE(1, 6, 18, 40);
    const bool nan = GENERATE(false, true);
    CAPTURE(tree_names[t], lower, order, nan);

    with_tree(
        t,
        [&](const auto& tree)
        {
            auto rng = make_rng();
            auto h = random_hpd<T>(40, rng);
            h[order - 1][order - 1] =
                T(nan ? std::numeric_limits<famlies::real_t<T>>::quiet_NaN()
                      : -1);
            test_matrix<T> a(40, 40, storage::general);
            copy_triangle(h, a.view(), lower);
            const auto r = with_triangle(
                lower,
                [&](auto tri) { return famlies::chol(a.view(), tri, tree); });
            REQUIRE_FALSE(r.has_value());
            CHECK(r.error() == not_positive_definite{order});
            CHECK(sentinel_intact(a.view(),
                                  lower ? strictly_upper : strictly_lower));
            CHECK(a.gaps_intact());
        });
}

TEST_CASE("chol's nb = 0 is the default block size of herk's leaf", "[chol]")
{
    using T = double;
    // The diagonal blocks, as a select on the chol child sees them.
    std::vector<len_type> blocks;
    const select record{.when =
                            [&](const auto& A)
                        {
                            blocks.push_back(A.length(0));
                            return true;
                        },
                        .then = chol_unblocked{.variant = chol_right_looking{}},
                        .otherwise =
                            chol_unblocked{.variant = chol_right_looking{}}};
    const auto run = [&](const auto& herk)
    {
        blocks.clear();
        auto rng = make_rng();
        auto h = random_hpd<T>(70, rng);
        const auto r = famlies::chol(
            h,
            famlies::lower_triangle,
            chol_blocked{.variant = chol_right_looking{},
                         .chol = record,
                         .herk = herk});
        CHECK(r.has_value());
        return blocks;
    };
    CHECK(run(reference{}) == std::vector<len_type>{32, 32, 6});
    // A select's is its `then`'s.
    const auto always = [](const auto&, const auto&) { return true; };
    CHECK(run(select{.when = always, .then = reference{}, .otherwise = blis{}})
          == std::vector<len_type>{32, 32, 6});
    const len_type kc = famlies::default_block_size<T>(blis{});
    CHECK(run(blis{}).front() == std::min<len_type>(kc, 70));
}

TEST_CASE("chol's call forms", "[chol]")
{
    using T = std::complex<double>;
    auto rng = make_rng();
    const auto h = random_hpd<T>(20, rng);
    const auto run = [&](bool lower, auto call)
    {
        MArray::marray<T, 2> a({20, 20}, T(0), MArray::COLUMN_MAJOR);
        copy_triangle(h, a.view(), lower);
        const std::expected<void, not_positive_definite> r = call(a);
        REQUIRE(r.has_value());
        CHECK(residual(h, a.view(), lower) < 30);
    };
    using famlies::lower_triangle;
    using famlies::upper_triangle;
    run(true, [](auto& a) { return famlies::chol(a, lower_triangle); });
    run(false,
        [](auto& a)
        {
            return famlies::chol(a.view(),
                                 upper_triangle,
                                 chol_blocked{.variant = chol_left_looking{},
                                              .nb = 8});
        });
    run(false,
        [](auto& a)
        {
            return famlies::chol<
                chol_blocked{.variant = chol_bordered{}, .nb = 8}>(
                a,
                upper_triangle);
        });
    run(true,
        [](auto& a)
        { return famlies::chol<famlies::chol_default>(a, lower_triangle); });
}

TEST_CASE("chol's select on a matrix sees a view", "[chol]")
{
    using T = std::complex<double>;
    // Only a view binds: an owning array doesn't convert to one's reference.
    int calls = 0;
    const auto small_view = [&](MArray::marray_view<T, 2>& A)
    {
        ++calls;
        return A.length(0) <= 8;
    };
    const chol_unblocked bordered{.variant = chol_bordered{}};
    const select blocks{.when = small_view,
                        .then = bordered,
                        .otherwise = bordered};
    const select tree{
        .when = small_view,
        .then = chol_unblocked{.variant = chol_right_looking{}},
        .otherwise = chol_blocked{.variant = chol_left_looking{},
                               .nb = 8,
                               .chol = blocks}
    };
    auto rng = make_rng();
    const auto h = random_hpd<T>(20, rng);
    for (const bool lower : {true, false})
    {
        CAPTURE(lower);
        MArray::marray<T, 2> a({20, 20}, T(0), MArray::COLUMN_MAJOR);
        copy_triangle(h, a.view(), lower);
        calls = 0;
        const auto r = with_triangle(lower,
                                     [&](auto tri)
                                     { return famlies::chol(a, tri, tree); });
        REQUIRE(r.has_value());
        // The root, then the diagonal blocks: 8, 8 and 4.
        CHECK(calls == 4);
        CHECK(residual(h, a.view(), lower) < 30);
    }
}

TEMPLATE_LIST_TEST_CASE("chol on an element is a square root",
                        "[chol]",
                        element_types)
{
    using T = TestType;
    using R = famlies::real_t<T>;
    using famlies::lower_triangle;
    using famlies::upper_triangle;

    T x = 4;
    CHECK(famlies::chol(x, lower_triangle).has_value());
    CHECK(x == T(2));
    x = 9;
    CHECK(famlies::chol(x, upper_triangle, reference{}).has_value());
    CHECK(x == T(3));
    x = 16;
    CHECK(famlies::chol<blis{}>(x, lower_triangle).has_value());
    CHECK(x == T(4));
    if constexpr (famlies::is_complex_v<T>)
    {
        // Only the real part is read, and the imaginary part is cleared.
        x = T(4, 3);
        CHECK(famlies::chol(x, lower_triangle).has_value());
        CHECK(x == T(2));
    }

    for (const R bad : {R(0), R(-1), std::numeric_limits<R>::quiet_NaN()})
    {
        CAPTURE(bad);
        x = bad;
        CHECK(famlies::chol(x, lower_triangle)
              == std::unexpected(not_positive_definite{1}));
        x = bad;
        CHECK(famlies::chol(x, lower_triangle, reference{})
              == std::unexpected(not_positive_definite{1}));
    }
}

namespace
{

template <class A, class Ctl>
constexpr bool chol_callable = requires(A&& a, const Ctl& ctl) {
    famlies::chol(std::forward<A>(a), famlies::lower_triangle, ctl);
};

template <class A>
constexpr bool chol_callable_by_default = requires(A&& a) {
    famlies::chol(std::forward<A>(a), famlies::lower_triangle);
};

// A tag chol doesn't have as a variant.
struct not_a_chol_variant
{
};

// A select's `when`: one that takes any operands, and one that takes only
// ints, however many.
constexpr auto takes_anything = [](const auto&...) { return true; };
constexpr auto takes_ints = [](std::same_as<int> auto...) { return true; };

} // namespace

TEST_CASE("chol rejects what it can't run", "[chol]")
{
    using Z = std::complex<double>;
    using M = MArray::marray_view<Z, 2>;
    STATIC_CHECK(chol_callable_by_default<M>);
    STATIC_CHECK(chol_callable_by_default<MArray::marray<Z, 2>&>);
    STATIC_CHECK(chol_callable_by_default<Z&>);
    STATIC_CHECK(chol_callable<M, chol_unblocked<chol_right_looking>>);

    // A variant chol doesn't have, anywhere in the tree.
    STATIC_CHECK(!famlies::chol_control<chol_unblocked<not_a_chol_variant>>);
    STATIC_CHECK(!chol_callable<M, chol_unblocked<not_a_chol_variant>>);
    STATIC_CHECK(!chol_callable<M, chol_blocked<not_a_chol_variant>>);
    STATIC_CHECK(
        !chol_callable<
            M,
            chol_blocked<chol_bordered, chol_unblocked<not_a_chol_variant>>>);
    STATIC_CHECK(
        !chol_callable<M, chol_recursive<chol_blocked<not_a_chol_variant>>>);
    using when = decltype(small);
    STATIC_CHECK(!chol_callable<M,
                                select<when,
                                       chol_unblocked<chol_bordered>,
                                       chol_unblocked<not_a_chol_variant>>>);
    // A node where a leaf goes, and a leaf where a node goes.
    STATIC_CHECK(
        !chol_callable<
            M,
            chol_unblocked<chol_bordered, chol_unblocked<chol_bordered>>>);
    STATIC_CHECK(!chol_callable<M, chol_blocked<chol_bordered, blis>>);
    STATIC_CHECK(!chol_callable<M, chol_unblocked<chol_bordered, blis, int>>);
    STATIC_CHECK(!chol_callable<M, blis>);
    STATIC_CHECK(!chol_callable<Z&, chol_unblocked<chol_bordered>>);

    // A select whose `when` doesn't take what it's given: at the root, on
    // an element, choosing a diagonal block's tree or a diagonal element's
    // leaf, and choosing a BLAS-level op's leaf.
    using anything = decltype(takes_anything);
    using ints = decltype(takes_ints);
    using unblocked = chol_unblocked<chol_left_looking>;
    STATIC_CHECK(chol_callable<M, select<anything, unblocked, unblocked>>);
    STATIC_CHECK(!chol_callable<M, select<ints, unblocked, unblocked>>);
    STATIC_CHECK(chol_callable<Z&, select<anything, blis, reference>>);
    STATIC_CHECK(!chol_callable<Z&, select<ints, blis, reference>>);
    STATIC_CHECK(
        !chol_callable<Z&,
                       select<anything, select<ints, blis, reference>, blis>>);
    STATIC_CHECK(
        chol_callable<M,
                      chol_blocked<chol_left_looking,
                                   select<anything, unblocked, unblocked>>>);
    STATIC_CHECK(
        !chol_callable<M,
                       chol_blocked<chol_left_looking,
                                    select<ints, unblocked, unblocked>>>);
    STATIC_CHECK(
        !chol_callable<M, chol_recursive<select<ints, unblocked, unblocked>>>);
    STATIC_CHECK(
        chol_callable<M,
                      chol_unblocked<chol_left_looking,
                                     select<anything, blis, reference>>>);
    STATIC_CHECK(
        !chol_callable<
            M,
            chol_unblocked<chol_left_looking, select<ints, blis, reference>>>);
    using leaves = select<anything, blis, reference>;
    using bad_leaves = select<ints, blis, reference>;
    STATIC_CHECK(
        chol_callable<
            M,
            chol_unblocked<chol_left_looking, blis, leaves, leaves, leaves>>);
    STATIC_CHECK(!chol_callable<M,
                                chol_unblocked<chol_left_looking,
                                               blis,
                                               bad_leaves,
                                               leaves,
                                               leaves>>);
    STATIC_CHECK(!chol_callable<M,
                                chol_unblocked<chol_left_looking,
                                               blis,
                                               leaves,
                                               bad_leaves,
                                               leaves>>);
    STATIC_CHECK(!chol_callable<M,
                                chol_blocked<chol_left_looking,
                                             unblocked,
                                             leaves,
                                             leaves,
                                             bad_leaves>>);

    // A conjugated, const, or wrongly ranked operand.
    STATIC_CHECK(!chol_callable_by_default<decltype(famlies::conjugated(
                     std::declval<M>()))>);
    STATIC_CHECK(!chol_callable_by_default<MArray::marray_view<const Z, 2>>);
    STATIC_CHECK(!chol_callable_by_default<const Z&>);
    STATIC_CHECK(!chol_callable_by_default<MArray::marray_view<Z, 1>>);
    STATIC_CHECK(!chol_callable_by_default<MArray::marray<Z, 2>>);

    // A type the blis leaf doesn't take, anywhere in the tree.
    using L = MArray::marray_view<long double, 2>;
    using all_reference = chol_blocked<chol_left_looking,
                                       chol_unblocked<chol_left_looking,
                                                      reference,
                                                      reference,
                                                      reference,
                                                      reference>,
                                       reference,
                                       reference,
                                       reference>;
    STATIC_CHECK(!chol_callable_by_default<L>);
    STATIC_CHECK(chol_callable<L, all_reference>);
    STATIC_CHECK(!chol_callable<L,
                                chol_blocked<chol_left_looking,
                                             chol_unblocked<chol_left_looking,
                                                            reference,
                                                            reference,
                                                            reference,
                                                            reference>,
                                             reference,
                                             blis,
                                             reference>>);
    STATIC_CHECK(!chol_callable<L,
                                chol_blocked<chol_left_looking,
                                             chol_unblocked<chol_left_looking,
                                                            blis,
                                                            reference,
                                                            reference,
                                                            reference>,
                                             reference,
                                             reference,
                                             reference>>);
}
