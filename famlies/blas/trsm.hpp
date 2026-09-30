#ifndef FAMLIES_BLAS_TRSM_HPP
#define FAMLIES_BLAS_TRSM_HPP

#include <famlies/core/control.hpp>
#include <famlies/core/operand.hpp>
#include <famlies/core/precondition.hpp>
#include <famlies/core/tags.hpp>
#include <famlies/leaf/blis.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace famlies
{

/// What runs trsm: a leaf, or a select of them.
template <class N> concept trsm_control = leaf_tree<N>;

namespace detail
{

template <class A, class B>
concept trsm_ranks = (rank_v<A> == 2 && (rank_v<B> == 2 || rank_v<B> == 1))
                  || (rank_v<A> == 0 && rank_v<B> == 1);

// Whether every leaf in leaf tree N has leaf_trsm for these operands.
template <class N, class S, class Tri, class D, class A, class B>
inline constexpr bool leaf_tree_has_trsm =
    requires(const N& leaf, S s, Tri t, D d, element_t<B> x, A&& a, B&& b) {
        leaf_trsm(leaf, s, t, d, x, std::forward<A>(a), std::forward<B>(b));
    };

template <class When,
          class Then,
          class Otherwise,
          class S,
          class Tri,
          class D,
          class A,
          class B>
inline constexpr bool
    leaf_tree_has_trsm<select<When, Then, Otherwise>, S, Tri, D, A, B> =
        std::predicate<const When&, A&, B&>
        && leaf_tree_has_trsm<Then, S, Tri, D, A&, B&>
        && leaf_tree_has_trsm<Otherwise, S, Tri, D, A&, B&>;

template <class Ctl, class S, class Tri, class D, class A, class B>
concept trsm_runs =
    trsm_control<Ctl>
    && trsm_ranks<A, B>
    && std::same_as<element_t<A>, element_t<B>>
    && leaf_tree_takes<std::remove_cvref_t<Ctl>, A, B>
    && leaf_tree_has_trsm<std::remove_cvref_t<Ctl>, S, Tri, D, A, B>;

} // namespace detail

/// B := alpha·op(A)⁻¹·B (left side) or B := alpha·B·op(A)⁻¹ (right side),
/// where op(A) is A as passed (conjugated, transposed, or both) and only its
/// triangle `Tri`, the triangle of A as passed, is read. With
/// `implicit_unit_diagonal`, A's diagonal isn't read either.
///
/// Forms: A a matrix and B a matrix or a vector; A an element and B a vector,
/// which divides B by op(A). A vector B is a column on the left and a row on
/// the right.
template <side S,
          triangle Tri,
          diagonal D,
          operand A,
          output_operand B,
          class Ctl = blis>
requires detail::trsm_runs<Ctl, S, Tri, D, A, B> void
trsm(S s, Tri t, D d, element_t<B> alpha, A&& a, B&& b, const Ctl& ctl = {})
{
    if constexpr (rank_v<A> == 2)
    {
        FAMLIES_EXPECTS("trsm", a.length(0) == a.length(1), a, b);
        constexpr int d_b =
            rank_v<B> == 2 && std::same_as<S, right_side_t> ? 1 : 0;
        FAMLIES_EXPECTS("trsm", a.length(0) == b.length(d_b), a, b);
    }
    if constexpr (is_select_v<Ctl>)
    {
        if (ctl.when(a, b))
            trsm(s, t, d, alpha, a, b, ctl.then);
        else
            trsm(s, t, d, alpha, a, b, ctl.otherwise);
    }
    else
        leaf_trsm(ctl, s, t, d, alpha, std::forward<A>(a), std::forward<B>(b));
}

/// trsm with its tree as a template argument.
template <auto Ctl,
          side S,
          triangle Tri,
          diagonal D,
          operand A,
          output_operand B>
requires detail::trsm_runs<decltype(Ctl), S, Tri, D, A, B>
void trsm(S s, Tri t, D d, element_t<B> alpha, A&& a, B&& b)
{ trsm(s, t, d, alpha, std::forward<A>(a), std::forward<B>(b), Ctl); }

} // namespace famlies

#endif // FAMLIES_BLAS_TRSM_HPP
