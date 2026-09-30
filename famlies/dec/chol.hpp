#ifndef FAMLIES_DEC_CHOL_HPP
#define FAMLIES_DEC_CHOL_HPP

#include <famlies/blas/gemm.hpp>
#include <famlies/blas/herk.hpp>
#include <famlies/blas/trsm.hpp>
#include <famlies/core/conj.hpp>
#include <famlies/core/control.hpp>
#include <famlies/core/error.hpp>
#include <famlies/core/operand.hpp>
#include <famlies/core/precondition.hpp>
#include <famlies/core/sweep.hpp>
#include <famlies/core/tags.hpp>
#include <famlies/leaf/blis.hpp>

#include <marray/marray.hpp>

#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

// Cholesky factorization.

namespace famlies
{

// chol's variants, named for the lower factor (upper runs them on A's
// transpose).

/// Each step solves the row or panel left of the diagonal block against the
/// finished factor above it, then updates and factors the block.
struct chol_bordered
{
};

/// Lazy: each step brings the current column or panel up to date from the
/// finished part to its left, then factors it. It writes only the panel.
struct chol_left_looking
{
};

/// Eager: each step factors the diagonal block, solves the panel below it,
/// and updates the whole trailing matrix with it.
struct chol_right_looking
{
};

/// Unblocked variant V: sweeps one index at a time, running `chol` (a leaf)
/// on each diagonal element.
template <class V,
          class Chol = blis,
          class Trsm = blis,
          class Herk = blis,
          class Gemm = blis>
struct chol_unblocked
{
    [[no_unique_address]] V variant;
    Chol chol{};
    Trsm trsm{};
    Herk herk{};
    Gemm gemm{};
};

/// Blocked variant V: sweeps nb indices at a time, running `chol` on each
/// diagonal block. nb = 0 is the default block size of herk's leaf.
template <class V,
          class Chol = chol_unblocked<V>,
          class Trsm = blis,
          class Herk = blis,
          class Gemm = blis>
struct chol_blocked
{
    [[no_unique_address]] V variant;
    len_type nb = 0;
    Chol chol{};
    Trsm trsm{};
    Herk herk{};
    Gemm gemm{};
};

/// Recursive halving: an operand of order at most `cutoff` (at least 1) goes
/// to `chol`.
template <class Chol = chol_unblocked<chol_left_looking>,
          class Trsm = blis,
          class Herk = blis>
struct chol_recursive
{
    len_type cutoff = 32;
    Chol chol{};
    Trsm trsm{};
    Herk herk{};
};

namespace detail
{

template <class V>
inline constexpr bool is_chol_variant = std::same_as<V, chol_bordered>
                                     || std::same_as<V, chol_left_looking>
                                     || std::same_as<V, chol_right_looking>;

// A tree that runs chol on a matrix.
template <class N> inline constexpr bool is_chol_tree = false;

template <class V, class Chol, class Trsm, class Herk, class Gemm>
inline constexpr bool is_chol_tree<chol_unblocked<V, Chol, Trsm, Herk, Gemm>> =
    is_chol_variant<V>
    && leaf_tree<Chol>
    && trsm_control<Trsm>
    && herk_control<Herk>
    && gemm_control<Gemm>;

template <class V, class Chol, class Trsm, class Herk, class Gemm>
inline constexpr bool is_chol_tree<chol_blocked<V, Chol, Trsm, Herk, Gemm>> =
    is_chol_variant<V>
    && is_chol_tree<Chol>
    && trsm_control<Trsm>
    && herk_control<Herk>
    && gemm_control<Gemm>;

template <class Chol, class Trsm, class Herk>
inline constexpr bool is_chol_tree<chol_recursive<Chol, Trsm, Herk>> =
    is_chol_tree<Chol> && trsm_control<Trsm> && herk_control<Herk>;

template <class When, class Then, class Otherwise>
inline constexpr bool is_chol_tree<select<When, Then, Otherwise>> =
    is_chol_tree<Then> && is_chol_tree<Otherwise>;

// Whether every leaf in leaf tree N has leaf_chol for the element A.
template <class N, class A>
inline constexpr bool leaf_tree_has_chol =
    requires(const N& leaf, A&& a) { leaf_chol(leaf, std::forward<A>(a)); };

template <class When, class Then, class Otherwise, class A>
inline constexpr bool leaf_tree_has_chol<select<When, Then, Otherwise>, A> =
    std::predicate<const When&, A&>
    && leaf_tree_has_chol<Then, A&>
    && leaf_tree_has_chol<Otherwise, A&>;

// The element of V an unblocked node's `chol` runs on.
template <class V>
using chol_element_t = decltype(std::declval<V&>()[len_type()][len_type()]);

// A step's updates before and after chol on its diagonal block, lower
// triangle. One text serves unblocked (bs = one: A11 an element, A10 and A21
// vectors) and blocked nodes, and chol_recursive makes right-looking's
// updates after, on ranges with R0 empty. Each group's requires-clause
// repeats its calls, which is what chol_steps_run checks.
template <class V,
          class Ctl,
          class I0,
          class I1,
          class I2,
          class T = element_t<V>,
          class R = real_t<T>>
void chol_before(chol_bordered, V A, const Ctl& ctl, I0 R0, I1 R1, I2)
    requires requires {
        trsm(right_side,
             upper_triangle,
             explicit_diagonal,
             T(1),
             adjoint(A[R0][R0]),
             A[R1][R0],
             ctl.trsm);
        herk(lower_triangle, R(-1), A[R1][R0], R(1), A[R1][R1], ctl.herk);
    }
{
    // A10 := A10·tril(A00)⁻ᴴ
    trsm(right_side,
         upper_triangle,
         explicit_diagonal,
         T(1),
         adjoint(A[R0][R0]),
         A[R1][R0],
         ctl.trsm);
    // A11 := A11 − A10·A10ᴴ
    herk(lower_triangle, R(-1), A[R1][R0], R(1), A[R1][R1], ctl.herk);
}

template <class V, class Ctl, class I0, class I1, class I2>
void chol_after(chol_bordered, V, const Ctl&, I0, I1, I2)
{
}

template <class V,
          class Ctl,
          class I0,
          class I1,
          class I2,
          class T = element_t<V>,
          class R = real_t<T>>
void chol_before(chol_left_looking, V A, const Ctl& ctl, I0 R0, I1 R1, I2 R2)
    requires requires {
        herk(lower_triangle, R(-1), A[R1][R0], R(1), A[R1][R1], ctl.herk);
        gemm(T(-1), A[R2][R0], adjoint(A[R1][R0]), T(1), A[R2][R1], ctl.gemm);
    }
{
    // A11 := A11 − A10·A10ᴴ
    herk(lower_triangle, R(-1), A[R1][R0], R(1), A[R1][R1], ctl.herk);
    // A21 := A21 − A20·A10ᴴ
    gemm(T(-1), A[R2][R0], adjoint(A[R1][R0]), T(1), A[R2][R1], ctl.gemm);
}

template <class V,
          class Ctl,
          class I0,
          class I1,
          class I2,
          class T = element_t<V>>
void chol_after(chol_left_looking, V A, const Ctl& ctl, I0, I1 R1, I2 R2)
    requires requires {
        trsm(right_side,
             upper_triangle,
             explicit_diagonal,
             T(1),
             adjoint(A[R1][R1]),
             A[R2][R1],
             ctl.trsm);
    }
{
    // A21 := A21·tril(A11)⁻ᴴ
    trsm(right_side,
         upper_triangle,
         explicit_diagonal,
         T(1),
         adjoint(A[R1][R1]),
         A[R2][R1],
         ctl.trsm);
}

template <class V, class Ctl, class I0, class I1, class I2>
void chol_before(chol_right_looking, V, const Ctl&, I0, I1, I2)
{
}

template <class V,
          class Ctl,
          class I0,
          class I1,
          class I2,
          class T = element_t<V>,
          class R = real_t<T>>
void chol_after(chol_right_looking, V A, const Ctl& ctl, I0, I1 R1, I2 R2)
    requires requires {
        trsm(right_side,
             upper_triangle,
             explicit_diagonal,
             T(1),
             adjoint(A[R1][R1]),
             A[R2][R1],
             ctl.trsm);
        herk(lower_triangle, R(-1), A[R2][R1], R(1), A[R2][R2], ctl.herk);
    }
{
    // A21 := A21·tril(A11)⁻ᴴ
    trsm(right_side,
         upper_triangle,
         explicit_diagonal,
         T(1),
         adjoint(A[R1][R1]),
         A[R2][R1],
         ctl.trsm);
    // A22 := A22 − A21·A21ᴴ
    herk(lower_triangle, R(-1), A[R2][R1], R(1), A[R2][R2], ctl.herk);
}

// Whether node Ctl, of variant K, can make its step's updates on V, with R1
// an I: an index in unblocked nodes, a range in blocked ones.
template <class K, class Ctl, class V, class I>
inline constexpr bool chol_steps_run =
    requires(K k, V A, const Ctl& ctl, MArray::range_t<len_type> r, I i) {
        chol_before(k, A, ctl, r, i, r);
        chol_after(k, A, ctl, r, i, r);
    };

// Whether chol tree N runs on V, the operand chol runs on: every leaf takes
// V, and so what indexing V gives, and has the ops its node calls it with,
// and every select's `when` takes a V lvalue.
template <class N, class V>
inline constexpr bool chol_tree_runs = leaf_tree_takes<N, V>;

template <class K, class Chol, class Trsm, class Herk, class Gemm, class V>
inline constexpr bool
    chol_tree_runs<chol_unblocked<K, Chol, Trsm, Herk, Gemm>, V> =
        leaf_tree_takes<Chol, V>
        && leaf_tree_takes<Trsm, V>
        && leaf_tree_takes<Herk, V>
        && leaf_tree_takes<Gemm, V>
        && leaf_tree_has_chol<Chol, chol_element_t<V>>
        && chol_steps_run<K,
                          chol_unblocked<K, Chol, Trsm, Herk, Gemm>,
                          V,
                          len_type>;

template <class K, class Chol, class Trsm, class Herk, class Gemm, class V>
inline constexpr bool
    chol_tree_runs<chol_blocked<K, Chol, Trsm, Herk, Gemm>, V> =
        chol_tree_runs<Chol, V>
        && leaf_tree_takes<Trsm, V>
        && leaf_tree_takes<Herk, V>
        && leaf_tree_takes<Gemm, V>
        && chol_steps_run<K,
                          chol_blocked<K, Chol, Trsm, Herk, Gemm>,
                          V,
                          MArray::range_t<len_type>>;

template <class Chol, class Trsm, class Herk, class V>
inline constexpr bool chol_tree_runs<chol_recursive<Chol, Trsm, Herk>, V> =
    chol_tree_runs<Chol, V>
    && leaf_tree_takes<Trsm, V>
    && leaf_tree_takes<Herk, V>
    && chol_steps_run<chol_right_looking,
                      chol_recursive<Chol, Trsm, Herk>,
                      V,
                      MArray::range_t<len_type>>;

template <class When, class Then, class Otherwise, class V>
inline constexpr bool chol_tree_runs<select<When, Then, Otherwise>, V> =
    std::predicate<const When&, V&>
    && chol_tree_runs<Then, V>
    && chol_tree_runs<Otherwise, V>;

} // namespace detail

/// A tree that runs chol: on a matrix, chol nodes, each with one of chol's
/// variants, down to leaves; on an element, a leaf. Selects choose
/// between either, and a select's `when` gets a matrix as a view.
template <class N>
concept chol_control =
    detail::is_chol_tree<std::remove_cvref_t<N>> || leaf_tree<N>;

namespace detail
{

// Ctl runs chol on A: a matrix with a chol tree, or an element with a leaf
// tree, every leaf taking A and having the ops it's called with.
template <class Ctl, class A>
concept chol_runs_on_matrix =
    (rank_v<A> == 2)
    && is_chol_tree<std::remove_cvref_t<Ctl>>
    && chol_tree_runs<std::remove_cvref_t<Ctl>,
                      decltype(as_view(std::declval<A>()))>;

template <class Ctl, class A>
concept chol_runs_on_element = (rank_v<A> == 0)
                            && leaf_tree<Ctl>
                            && leaf_tree_takes<std::remove_cvref_t<Ctl>, A>
                            && leaf_tree_has_chol<std::remove_cvref_t<Ctl>, A>;

template <class Ctl, class A>
concept chol_runs =
    output_operand<A>
    && (chol_runs_on_matrix<Ctl, A> || chol_runs_on_element<Ctl, A>);

} // namespace detail

/// The default tree: blocked right-looking at the default block size,
/// blocked right-looking at nb = 32 on its diagonal blocks, and unblocked
/// left-looking on theirs.
inline constexpr chol_blocked chol_default{
    .variant = chol_right_looking{},
    .chol =
        chol_blocked{.variant = chol_right_looking{},
                                  .nb = 32,
                                  .chol = chol_unblocked{.variant = chol_left_looking{}}}
};

using chol_default_t = std::remove_const_t<decltype(chol_default)>;

/// Factors the Hermitian positive definite A, read from its triangle `Tri`,
/// in place: A = L·Lᴴ with L lower (`lower_triangle`), or A = Uᴴ·U with U
/// upper, overwriting that triangle. The other one is neither read nor
/// written. Fails with `not_positive_definite` at the first leading minor
/// that isn't, leaving A partially factored.
template <operand A, triangle Tri, chol_control Ctl = chol_default_t>
requires(rank_v<A> == 2) && detail::chol_runs<Ctl, A>
std::expected<void, not_positive_definite>
chol(A&& a, Tri t, const Ctl& ctl = chol_default);

/// chol on an element, run by a leaf: its square root, unless its real part
/// isn't positive.
template <operand A, triangle Tri, chol_control Ctl = blis>
requires(rank_v<A> == 0) && detail::chol_runs<Ctl, A>
std::expected<void, not_positive_definite>
chol(A&& a, Tri t, const Ctl& ctl = {})
{
    if constexpr (is_select_v<Ctl>)
        return ctl.when(a) ? chol(a, t, ctl.then) : chol(a, t, ctl.otherwise);
    else
        return leaf_chol(ctl, std::forward<A>(a));
}

/// chol with its tree as a template argument.
template <auto Ctl, operand A, triangle Tri>
requires detail::chol_runs<decltype(Ctl), A>
std::expected<void, not_positive_definite> chol(A&& a, Tri t)
{ return chol(std::forward<A>(a), t, Ctl); }

namespace detail
{

template <class N> inline constexpr bool is_chol_blocked = false;

template <class V, class Chol, class Trsm, class Herk, class Gemm>
inline constexpr bool is_chol_blocked<chol_blocked<V, Chol, Trsm, Herk, Gemm>> =
    true;

template <class N> inline constexpr bool is_chol_recursive = false;

template <class Chol, class Trsm, class Herk>
inline constexpr bool is_chol_recursive<chol_recursive<Chol, Trsm, Herk>> =
    true;

// A variant: at each step, the updates before, chol on the diagonal block,
// and the updates after.
template <class V, class Ctl, class B>
std::expected<void, not_positive_definite>
chol_variant(V A, const Ctl& ctl, B bs)
{
    for (auto [R0, R1, R2] : sweep(A.length(0), bs, forward))
    {
        chol_before(ctl.variant, A, ctl, R0, R1, R2);
        // A11 := chol(A11)
        if (auto r = chol(A[R1][R1], lower_triangle, ctl.chol); !r)
            return offset(r, R1);
        chol_after(ctl.variant, A, ctl, R0, R1, R2);
    }
    return {};
}

// A = L·Lᴴ on a node of a chol tree.
template <class V, class Ctl>
std::expected<void, not_positive_definite> chol_lower(V A, const Ctl& ctl)
{
    using T = element_t<V>;

    if constexpr (is_chol_recursive<Ctl>)
    {
        FAMLIES_EXPECTS("chol", ctl.cutoff >= 1, ctl.cutoff);
        const len_type n = A.length(0);
        if (n <= ctl.cutoff)
            return chol(A, lower_triangle, ctl.chol);

        // A11 := chol(A11); A21 := A21·tril(A11)⁻ᴴ;
        // A22 := chol(A22 − A21·A21ᴴ)
        const auto R0 = MArray::range(len_type(0), len_type(0));
        const auto R1 = MArray::range(len_type(0), n / 2);
        const auto R2 = MArray::range(n / 2, n);
        if (auto r = chol(A[R1][R1], lower_triangle, ctl); !r)
            return r;
        chol_after(chol_right_looking{}, A, ctl, R0, R1, R2);
        return offset(chol(A[R2][R2], lower_triangle, ctl), R2);
    }
    else if constexpr (is_chol_blocked<Ctl>)
    {
        FAMLIES_EXPECTS("chol", ctl.nb >= 0, ctl.nb);
        return chol_variant(A,
                            ctl,
                            ctl.nb > 0 ? ctl.nb
                                       : default_block_size<T>(ctl.herk));
    }
    else
        return chol_variant(A, ctl, one);
}

} // namespace detail

template <operand A, triangle Tri, chol_control Ctl>
requires(rank_v<A> == 2) && detail::chol_runs<Ctl, A>
std::expected<void, not_positive_definite> chol(A&& a, Tri t, const Ctl& ctl)
{
    FAMLIES_EXPECTS("chol", a.length(0) == a.length(1), a);
    // A select's `when` sees a view, at the root as below it.
    auto v = detail::as_view(std::forward<A>(a));
    if constexpr (is_select_v<Ctl>)
        return ctl.when(v) ? chol(v, t, ctl.then) : chol(v, t, ctl.otherwise);
    else
    {
        // Upper: A = Uᴴ·U is Aᵀ = Uᵀ·(Uᵀ)ᴴ, so Uᵀ is the lower factor of
        // Aᵀ, whose lower triangle is A's upper one.
        if constexpr (std::same_as<Tri, upper_triangle_t>)
            return detail::chol_lower(v.T(), ctl);
        else
            return detail::chol_lower(v, ctl);
    }
}

} // namespace famlies

#endif // FAMLIES_DEC_CHOL_HPP
