#ifndef FAMLIES_BLAS_HERK_HPP
#define FAMLIES_BLAS_HERK_HPP

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

/// What runs herk: a leaf, or a select of them.
template <class N> concept herk_control = leaf_tree<N>;

namespace detail
{

template <class A, class C>
concept herk_ranks = (rank_v<C> == 2 && (rank_v<A> == 2 || rank_v<A> == 1))
                  || (rank_v<C> == 0 && rank_v<A> == 1);

// Whether every leaf in leaf tree N has leaf_herk for these operands.
template <class N, class Tri, class A, class C>
inline constexpr bool leaf_tree_has_herk =
    requires(const N& leaf, Tri t, real_t<element_t<C>> x, A&& a, C&& c) {
        leaf_herk(leaf, t, x, std::forward<A>(a), x, std::forward<C>(c));
    };

template <class When, class Then, class Otherwise, class Tri, class A, class C>
inline constexpr bool
    leaf_tree_has_herk<select<When, Then, Otherwise>, Tri, A, C> =
        std::predicate<const When&, A&, C&>
        && leaf_tree_has_herk<Then, Tri, A&, C&>
        && leaf_tree_has_herk<Otherwise, Tri, A&, C&>;

template <class Ctl, class Tri, class A, class C>
concept herk_runs = herk_control<Ctl>
                 && herk_ranks<A, C>
                 && std::same_as<element_t<A>, element_t<C>>
                 && leaf_tree_takes<std::remove_cvref_t<Ctl>, A, C>
                 && leaf_tree_has_herk<std::remove_cvref_t<Ctl>, Tri, A, C>;

} // namespace detail

/// C := alpha·A·Aᴴ + beta·C, writing only C's triangle `Tri` and leaving its
/// diagonal real. With beta = 0, C isn't read. A vector A is a column with a
/// matrix C, and a row with an element C.
template <triangle Tri, operand A, output_operand C, class Ctl = blis>
requires detail::herk_runs<Ctl, Tri, A, C> void herk(Tri t,
                                                real_t<element_t<C>> alpha,
                                                A&& a,
                                                real_t<element_t<C>> beta,
                                                C&& c,
                                                const Ctl& ctl = {})
{
    if constexpr (rank_v<C> == 2)
    {
        FAMLIES_EXPECTS("herk", c.length(0) == c.length(1), a, c);
        FAMLIES_EXPECTS("herk", a.length(0) == c.length(0), a, c);
    }
    if constexpr (is_select_v<Ctl>)
    {
        if (ctl.when(a, c))
            herk(t, alpha, a, beta, c, ctl.then);
        else
            herk(t, alpha, a, beta, c, ctl.otherwise);
    }
    else
        leaf_herk(ctl,
                  t,
                  alpha,
                  std::forward<A>(a),
                  beta,
                  std::forward<C>(c));
}

/// herk with its tree as a template argument.
template <auto Ctl, triangle Tri, operand A, output_operand C>
requires detail::herk_runs<decltype(Ctl), Tri, A, C> void
herk(Tri t, real_t<element_t<C>> alpha, A&& a, real_t<element_t<C>> beta, C&& c)
{ herk(t, alpha, std::forward<A>(a), beta, std::forward<C>(c), Ctl); }

} // namespace famlies

#endif // FAMLIES_BLAS_HERK_HPP
