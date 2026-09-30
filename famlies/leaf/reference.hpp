#ifndef FAMLIES_LEAF_REFERENCE_HPP
#define FAMLIES_LEAF_REFERENCE_HPP

#include <famlies/core/conj.hpp>
#include <famlies/core/control.hpp>
#include <famlies/core/error.hpp>
#include <famlies/core/operand.hpp>
#include <famlies/core/tags.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <expected>
#include <utility>

namespace famlies
{

/// The leaf that runs BLAS-level ops as plain loops, over any field type.
/// It's the tests' oracle.
struct reference
{
};

template <> inline constexpr bool enable_leaf<reference> = true;

template <class A> inline constexpr bool leaf_takes<reference, A> = operand<A>;

template <field T> len_type default_block_size(const reference&)
{ return 32; }

// The reference leaf's ops, with BLIS's conventions: beta = 0 reads nothing
// of the output, and a Hermitian output's diagonal is left real.

/// B := alpha·B·op(A)⁻¹ or alpha·op(A)⁻¹·B, by substitution.
template <side S, triangle Tri, diagonal D, class A, class B>
void
leaf_trsm(const reference& leaf, S, Tri, D, element_t<B> alpha, A&& a, B&& b)
{
    using namespace detail;
    using T = element_t<B>;
    constexpr bool unit = std::same_as<D, implicit_unit_diagonal_t>;
    auto x = as_view(std::forward<B>(b));

    if constexpr (rank_v<A> == 0)
    {
        for (len_type i = 0; i < x.length(0); ++i)
        {
            T v = alpha * x[i];
            if constexpr (!unit)
                v /= value(a);
            x[i] = v;
        }
    }
    else if constexpr (std::same_as<S, right_side_t>)
    {
        // X·op(A) = alpha·B is op(A)ᵀ·Xᵀ = alpha·Bᵀ, whose triangle is the
        // other one. A vector is its own transpose.
        using Flipped = std::conditional_t<std::same_as<Tri, lower_triangle_t>,
                                           upper_triangle_t,
                                           lower_triangle_t>;
        if constexpr (rank_v<B> == 1)
            leaf_trsm(leaf, left_side, Flipped{}, D{}, alpha, a.T(), x);
        else
            leaf_trsm(leaf, left_side, Flipped{}, D{}, alpha, a.T(), x.T());
    }
    else
    {
        // Column by column, from the top for a lower triangle and from the
        // bottom for an upper one.
        constexpr bool lower = std::same_as<Tri, lower_triangle_t>;
        const len_type m = a.length(0);
        const len_type n = rank_v<B> == 1 ? 1 : x.length(1);
        for (len_type j = 0; j < n; ++j)
        {
            const auto xj = [&](len_type i) -> T&
            {
                if constexpr (rank_v<B> == 1)
                    return x[i];
                else
                    return x[i][j];
            };
            for (len_type s = 0; s < m; ++s)
            {
                const len_type i = lower ? s : m - 1 - s;
                T v = alpha * xj(i);
                for (len_type k = lower ? 0 : i + 1; k < (lower ? i : m); ++k)
                    v -= value(a[i][k]) * xj(k);
                if constexpr (!unit)
                    v /= value(a[i][i]);
                xj(i) = v;
            }
        }
    }
}

/// C := alpha·A·Aᴴ + beta·C in C's triangle.
template <triangle Tri, class A, class C>
void leaf_herk(const reference&,
               Tri,
               real_t<element_t<C>> alpha,
               A&& a,
               real_t<element_t<C>> beta,
               C&& c)
{
    using namespace detail;
    using T = element_t<C>;
    const auto real_part = [](const T& v)
    {
        if constexpr (is_complex_v<T>)
            return T(std::real(v));
        else
            return v;
    };

    if constexpr (rank_v<C> == 0)
    {
        // a is a row.
        T sum(0);
        for (len_type p = 0; p < a.length(0); ++p)
            sum += value(a[p]) * conjugate(value(a[p]));
        T v = T(alpha) * sum;
        if (beta != 0)
            v += T(beta) * c;
        c = real_part(v);
    }
    else
    {
        // A vector a is a column.
        constexpr bool lower = std::same_as<Tri, lower_triangle_t>;
        const auto at = [&](len_type i, len_type p)
        {
            if constexpr (rank_v<A> == 1)
                return value(a[i]);
            else
                return value(a[i][p]);
        };
        auto cv = as_view(std::forward<C>(c));
        const len_type m = cv.length(0);
        const len_type k = rank_v<A> == 1 ? 1 : a.length(1);
        for (len_type j = 0; j < m; ++j)
            for (len_type i = lower ? j : 0; i < (lower ? m : j + 1); ++i)
            {
                T sum(0);
                for (len_type p = 0; p < k; ++p)
                    sum += at(i, p) * conjugate(at(j, p));
                T v = T(alpha) * sum;
                if (beta != 0)
                    v += T(beta) * cv[i][j];
                cv[i][j] = i == j ? real_part(v) : v;
            }
    }
}

/// C := alpha·A·B + beta·C, as an m × k by k × n product: a vector A is a
/// row unless B is one too (a column), and a vector B is a column.
template <class A, class B, class C>
void leaf_gemm(const reference&,
               element_t<C> alpha,
               A&& a,
               B&& b,
               element_t<C> beta,
               C&& c)
{
    using namespace detail;
    using T = element_t<C>;
    const auto at = [&](len_type i, len_type p)
    {
        if constexpr (rank_v<A> == 2)
            return value(a[i][p]);
        else
            return value(a[p]);
    };
    const auto bt = [&](len_type p, len_type j)
    {
        if constexpr (rank_v<B> == 2)
            return value(b[p][j]);
        else
            return value(b[p]);
    };
    const len_type m = rank_v<A> == 2 ? a.length(0) : 1;
    const len_type n = rank_v<B> == 2 ? b.length(1) : 1;
    const len_type k = rank_v<A> == 2 ? a.length(1) : a.length(0);

    const auto update = [&](len_type i, len_type j, T& cij)
    {
        T sum(0);
        for (len_type p = 0; p < k; ++p) sum += at(i, p) * bt(p, j);
        T v = alpha * sum;
        if (beta != T(0))
            v += beta * cij;
        cij = v;
    };
    if constexpr (rank_v<C> == 0)
        update(0, 0, c);
    else
    {
        auto cv = as_view(std::forward<C>(c));
        for (len_type j = 0; j < n; ++j)
            for (len_type i = 0; i < m; ++i)
            {
                if constexpr (rank_v<C> == 2)
                    update(i, j, cv[i][j]);
                else if constexpr (rank_v<A> == 2)
                    update(i, j, cv[i]);
                else
                    update(i, j, cv[j]);
            }
    }
}

/// chol on an element.
template <class A>
std::expected<void, not_positive_definite> leaf_chol(const reference&, A&& a)
{ return detail::chol_element(std::forward<A>(a)); }

} // namespace famlies

#endif // FAMLIES_LEAF_REFERENCE_HPP
