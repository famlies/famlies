#ifndef FAMLIES_BLAS_GEMM_HPP
#define FAMLIES_BLAS_GEMM_HPP

#include <famlies/core/control.hpp>
#include <famlies/core/operand.hpp>
#include <famlies/core/precondition.hpp>
#include <famlies/leaf/blis.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace famlies
{

/// What runs gemm: a leaf, or a select of them.
template <class N> concept gemm_control = leaf_tree<N>;

namespace detail
{

template <class A, class B, class C>
concept gemm_ranks = (rank_v<A> == 2 && rank_v<B> == 2 && rank_v<C> == 2)
                  || (rank_v<A> == 2 && rank_v<B> == 1 && rank_v<C> == 1)
                  || (rank_v<A> == 1 && rank_v<B> == 2 && rank_v<C> == 1)
                  || (rank_v<A> == 1 && rank_v<B> == 1 && rank_v<C> == 0);

// Whether every leaf in leaf tree N has leaf_gemm for these operands.
template <class N, class A, class B, class C>
inline constexpr bool leaf_tree_has_gemm =
    requires(const N& leaf, element_t<C> x, A&& a, B&& b, C&& c) {
        leaf_gemm(leaf,
                  x,
                  std::forward<A>(a),
                  std::forward<B>(b),
                  x,
                  std::forward<C>(c));
    };

template <class When, class Then, class Otherwise, class A, class B, class C>
inline constexpr bool
    leaf_tree_has_gemm<select<When, Then, Otherwise>, A, B, C> =
        std::predicate<const When&, A&, B&, C&>
        && leaf_tree_has_gemm<Then, A&, B&, C&>
        && leaf_tree_has_gemm<Otherwise, A&, B&, C&>;

template <class Ctl, class A, class B, class C>
concept gemm_runs = gemm_control<Ctl>
                 && gemm_ranks<A, B, C>
                 && std::same_as<element_t<A>, element_t<C>>
                 && std::same_as<element_t<B>, element_t<C>>
                 && leaf_tree_takes<std::remove_cvref_t<Ctl>, A, B, C>
                 && leaf_tree_has_gemm<std::remove_cvref_t<Ctl>, A, B, C>;

} // namespace detail

/// C := alpha·A·B + beta·C. With beta = 0, C isn't read. A vector A is a row
/// and a vector B a column: a matrix times a vector is a column, a vector
/// times a matrix a row, and two vectors give an element.
template <operand A, operand B, output_operand C, class Ctl = blis>
requires detail::gemm_runs<Ctl, A, B, C> void gemm(element_t<C> alpha,
                                                   A&& a,
                                                   B&& b,
                                                   element_t<C> beta,
                                                   C&& c,
                                                   const Ctl& ctl = {})
{
    if constexpr (rank_v<C> == 2)
    {
        FAMLIES_EXPECTS("gemm", a.length(1) == b.length(0), a, b, c);
        FAMLIES_EXPECTS(
            "gemm",
            c.length(0) == a.length(0) && c.length(1) == b.length(1),
            a,
            b,
            c);
    }
    else if constexpr (rank_v<C> == 1 && rank_v<A> == 2)
    {
        FAMLIES_EXPECTS("gemm", a.length(1) == b.length(0), a, b, c);
        FAMLIES_EXPECTS("gemm", c.length(0) == a.length(0), a, b, c);
    }
    else if constexpr (rank_v<C> == 1)
    {
        FAMLIES_EXPECTS("gemm", a.length(0) == b.length(0), a, b, c);
        FAMLIES_EXPECTS("gemm", c.length(0) == b.length(1), a, b, c);
    }
    else
        FAMLIES_EXPECTS("gemm", a.length(0) == b.length(0), a, b, c);

    if constexpr (is_select_v<Ctl>)
    {
        if (ctl.when(a, b, c))
            gemm(alpha, a, b, beta, c, ctl.then);
        else
            gemm(alpha, a, b, beta, c, ctl.otherwise);
    }
    else
        leaf_gemm(ctl,
                  alpha,
                  std::forward<A>(a),
                  std::forward<B>(b),
                  beta,
                  std::forward<C>(c));
}

/// gemm with its tree as a template argument.
template <auto Ctl, operand A, operand B, output_operand C>
requires detail::gemm_runs<decltype(Ctl), A, B, C>
void gemm(element_t<C> alpha, A&& a, B&& b, element_t<C> beta, C&& c)
{
    gemm(alpha,
         std::forward<A>(a),
         std::forward<B>(b),
         beta,
         std::forward<C>(c),
         Ctl);
}

} // namespace famlies

#endif // FAMLIES_BLAS_GEMM_HPP
