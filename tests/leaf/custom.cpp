// Leaves defined outside famlies, after the op headers: their ops are found
// by argument-dependent lookup in their own namespace. `counting` counts its
// calls and runs them on the reference leaf; `no_gemm` has every op but
// gemm, and `no_ops` none.

#include "support/random.hpp"

#include <famlies/dec/chol.hpp>
#include <famlies/leaf/reference.hpp>

#include <catch2/catch_test_macros.hpp>

#include <utility>

namespace custom
{

using famlies::len_type;

struct counts
{
    int trsm = 0;
    int herk = 0;
    int gemm = 0;
    int chol = 0;
    int block_size = 0;
};

inline counts calls;

struct counting
{
};

template <class T> len_type default_block_size(const counting&)
{
    ++calls.block_size;
    return 5;
}

template <class S, class Tri, class D, class T, class A, class B>
void leaf_trsm(const counting&, S s, Tri t, D d, T alpha, A&& a, B&& b)
{
    ++calls.trsm;
    famlies::leaf_trsm(famlies::reference{},
                       s,
                       t,
                       d,
                       alpha,
                       std::forward<A>(a),
                       std::forward<B>(b));
}

template <class Tri, class R, class A, class C>
void leaf_herk(const counting&, Tri t, R alpha, A&& a, R beta, C&& c)
{
    ++calls.herk;
    famlies::leaf_herk(famlies::reference{},
                       t,
                       alpha,
                       std::forward<A>(a),
                       beta,
                       std::forward<C>(c));
}

template <class T, class A, class B, class C>
void leaf_gemm(const counting&, T alpha, A&& a, B&& b, T beta, C&& c)
{
    ++calls.gemm;
    famlies::leaf_gemm(famlies::reference{},
                       alpha,
                       std::forward<A>(a),
                       std::forward<B>(b),
                       beta,
                       std::forward<C>(c));
}

template <class A> auto leaf_chol(const counting&, A&& a)
{
    ++calls.chol;
    return famlies::leaf_chol(famlies::reference{}, std::forward<A>(a));
}

/// Every op but gemm, on the reference leaf.
struct no_gemm
{
};

template <class T> len_type default_block_size(const no_gemm&)
{ return 5; }

template <class... Args> void leaf_trsm(const no_gemm&, Args&&... args)
{ famlies::leaf_trsm(famlies::reference{}, std::forward<Args>(args)...); }

template <class... Args> void leaf_herk(const no_gemm&, Args&&... args)
{ famlies::leaf_herk(famlies::reference{}, std::forward<Args>(args)...); }

template <class A> auto leaf_chol(const no_gemm&, A&& a)
{ return famlies::leaf_chol(famlies::reference{}, std::forward<A>(a)); }

/// No ops.
struct no_ops
{
};

} // namespace custom

template <>
inline constexpr bool famlies::enable_leaf<custom::counting> = true;

template <class A>
inline constexpr bool famlies::leaf_takes<custom::counting, A> =
    famlies::leaf_takes<famlies::reference, A>;

template <> inline constexpr bool famlies::enable_leaf<custom::no_gemm> = true;

template <class A>
inline constexpr bool famlies::leaf_takes<custom::no_gemm, A> = true;

template <> inline constexpr bool famlies::enable_leaf<custom::no_ops> = true;

template <class A>
inline constexpr bool famlies::leaf_takes<custom::no_ops, A> = true;

namespace
{

using M = MArray::marray_view<double, 2>;

template <class A, class Ctl>
constexpr bool chol_callable = requires(A a, const Ctl& ctl) {
    famlies::chol(a, famlies::lower_triangle, ctl);
};

template <class Ctl>
constexpr bool herk_callable = requires(M a, M c, const Ctl& ctl) {
    famlies::herk(famlies::lower_triangle, 1.0, a, 1.0, c, ctl);
};

template <class Ctl>
constexpr bool gemm_callable = requires(M a, M b, M c, const Ctl& ctl) {
    famlies::gemm(1.0, a, b, 1.0, c, ctl);
};

template <class Ctl>
constexpr bool trsm_callable = requires(M a, M b, const Ctl& ctl) {
    famlies::trsm(famlies::left_side,
                  famlies::lower_triangle,
                  famlies::explicit_diagonal,
                  1.0,
                  a,
                  b,
                  ctl);
};

} // namespace

TEST_CASE("a leaf outside famlies runs every op of a tree", "[leaf]")
{
    using famlies::chol_blocked;
    using famlies::chol_left_looking;
    using famlies::chol_right_looking;
    using famlies::chol_unblocked;
    using custom::counting;
    using famlies::len_type;

    STATIC_CHECK(famlies::leaf<counting>);
    STATIC_CHECK(famlies::chol_control<chol_blocked<chol_right_looking,
                                                    chol_unblocked<
                                                        chol_left_looking,
                                                        counting,
                                                        counting,
                                                        counting,
                                                        counting>,
                                                    counting,
                                                    counting,
                                                    counting>>);

    // nb = 0 takes the custom leaf's block size, 5: blocks of 5, 5 and 2.
    const chol_blocked tree{
        .variant = chol_right_looking{},
        .chol = chol_unblocked{.variant = chol_left_looking{},
                               .chol = counting{},
                               .trsm = counting{},
                               .herk = counting{},
                               .gemm = counting{}},
        .trsm = counting{},
        .herk = counting{},
        .gemm = counting{}
    };
    const chol_blocked on_reference{
        .variant = chol_right_looking{},
        .nb = 5,
        .chol = chol_unblocked{.variant = chol_left_looking{},
                               .chol = famlies::reference{},
                               .trsm = famlies::reference{},
                               .herk = famlies::reference{},
                               .gemm = famlies::reference{}},
        .trsm = famlies::reference{},
        .herk = famlies::reference{},
        .gemm = famlies::reference{}
    };

    auto rng = famlies::test::make_rng();
    auto a = famlies::test::random_hpd<double>(12, rng);
    auto b = a;
    custom::calls = {};
    REQUIRE(famlies::chol(a, famlies::lower_triangle, tree).has_value());
    REQUIRE(
        famlies::chol(b, famlies::lower_triangle, on_reference).has_value());

    // The same updates in the same order as on the reference leaf.
    for (len_type j = 0; j < 12; ++j)
        for (len_type i = j; i < 12; ++i) CHECK(a[i][j] == b[i][j]);

    CHECK(custom::calls.block_size == 1);
    CHECK(custom::calls.chol == 12);
    // Per block, trsm and herk once blocked, and per element once each
    // unblocked, with gemm only there.
    CHECK(custom::calls.trsm == 3 + 12);
    CHECK(custom::calls.herk == 3 + 12);
    CHECK(custom::calls.gemm == 12);
}

TEST_CASE("chol on an element runs a leaf outside famlies", "[leaf]")
{
    custom::calls = {};
    double x = 9;
    CHECK(famlies::chol(x, famlies::lower_triangle, custom::counting{})
              .has_value());
    CHECK(x == 3);
    CHECK(custom::calls.chol == 1);
}

TEST_CASE("a leaf without an op the tree calls is rejected at the call site",
          "[leaf]")
{
    using custom::no_gemm;
    using custom::no_ops;
    using famlies::blis;
    using famlies::chol_blocked;
    using famlies::chol_bordered;
    using famlies::chol_left_looking;
    using famlies::chol_recursive;
    using famlies::chol_right_looking;
    using famlies::chol_unblocked;
    using famlies::select;
    using when = decltype([](const auto&...) { return true; });

    // BLAS-level ops.
    STATIC_CHECK(herk_callable<no_gemm>);
    STATIC_CHECK(trsm_callable<no_gemm>);
    STATIC_CHECK(!gemm_callable<no_gemm>);
    STATIC_CHECK(!herk_callable<no_ops>);
    STATIC_CHECK(!trsm_callable<no_ops>);
    STATIC_CHECK(gemm_callable<select<when, blis, blis>>);
    STATIC_CHECK(!gemm_callable<select<when, blis, no_gemm>>);

    // chol on an element.
    STATIC_CHECK(chol_callable<double&, no_gemm>);
    STATIC_CHECK(!chol_callable<double&, no_ops>);

    // chol on a matrix: only left-looking calls gemm.
    using unblocked_bordered =
        chol_unblocked<chol_bordered, no_gemm, no_gemm, no_gemm, no_gemm>;
    using unblocked_left_looking =
        chol_unblocked<chol_left_looking, no_gemm, no_gemm, no_gemm, no_gemm>;
    using unblocked_right_looking =
        chol_unblocked<chol_right_looking, no_gemm, no_gemm, no_gemm, no_gemm>;
    STATIC_CHECK(chol_callable<M, unblocked_bordered>);
    STATIC_CHECK(chol_callable<M, unblocked_right_looking>);
    STATIC_CHECK(!chol_callable<M, unblocked_left_looking>);
    STATIC_CHECK(chol_callable<M,
                               chol_blocked<chol_right_looking,
                                            unblocked_right_looking,
                                            no_gemm,
                                            no_gemm,
                                            no_gemm>>);
    STATIC_CHECK(!chol_callable<M,
                                chol_blocked<chol_left_looking,
                                             unblocked_right_looking,
                                             blis,
                                             blis,
                                             no_gemm>>);
    STATIC_CHECK(
        chol_callable<M,
                      chol_recursive<unblocked_bordered, no_gemm, no_gemm>>);

    // Anywhere in the tree: below a blocked node, and through a select.
    STATIC_CHECK(
        !chol_callable<M, chol_blocked<chol_right_looking,
                                       unblocked_left_looking>>);
    STATIC_CHECK(!chol_callable<
                 M,
                 chol_recursive<chol_unblocked<chol_left_looking,
                                               blis,
                                               blis,
                                               blis,
                                               select<when, blis, no_gemm>>>>);
    // A leaf without chol on an element, under an unblocked node.
    STATIC_CHECK(!chol_callable<M, chol_unblocked<chol_right_looking, no_ops>>);

    // What's accepted runs.
    auto rng = famlies::test::make_rng();
    auto a = famlies::test::random_hpd<double>(12, rng);
    CHECK(famlies::chol(a,
                        famlies::lower_triangle,
                        chol_blocked{.variant = chol_right_looking{},
                                     .chol = unblocked_right_looking{},
                                     .trsm = no_gemm{},
                                     .herk = no_gemm{},
                                     .gemm = no_gemm{}})
              .has_value());
}
