#ifndef FAMLIES_TESTS_SUPPORT_WIDE_HPP
#define FAMLIES_TESTS_SUPPORT_WIDE_HPP

#include <famlies/core/conj.hpp>
#include <famlies/core/operand.hpp>

#include <marray/marray.hpp>

#include <cmath>
#include <complex>
#include <limits>

// The oracle's side: operands copied into a wider type (long double and its
// complex; on targets where long double is double, the same width), where
// the reference leaf computes what a result should be.

namespace famlies::test
{

template <class T> struct wide_type
{
    using type = long double;
};

template <class R> struct wide_type<std::complex<R>>
{
    using type = std::complex<long double>;
};

template <class T> using wide_t = typename wide_type<T>::type;

template <class T> using wide_matrix = MArray::marray<wide_t<T>, 2>;

/// Machine epsilon of T's real type.
template <class T>
inline constexpr long double eps = std::numeric_limits<real_t<T>>::epsilon();

/// The rounding tolerance of a result that sums k products, relative to the
/// sum of their magnitudes: 8(k + 4) eps.
template <class T> long double tolerance(len_type k)
{ return 8 * static_cast<long double>(k + 4) * eps<T>; }

/// Operand a (a matrix, vector or element, conjugated or not) in the wide
/// type: an owned column-major copy, or a value.
template <class A> auto widen(A&& a)
{
    using W = wide_t<element_t<A>>;
    if constexpr (rank_v<A> == 0)
        return W(detail::value(a));
    else if constexpr (rank_v<A> == 1)
    {
        MArray::marray<W, 1> w({a.length(0)});
        for (len_type i = 0; i < a.length(0); ++i)
            w[i] = W(detail::value(a[i]));
        return w;
    }
    else
    {
        MArray::marray<W, 2> w({a.length(0), a.length(1)},
                               W(0),
                               MArray::COLUMN_MAJOR);
        for (len_type i = 0; i < a.length(0); ++i)
            for (len_type j = 0; j < a.length(1); ++j)
                w[i][j] = W(detail::value(a[i][j]));
        return w;
    }
}

/// |x| elementwise, as long double: what bounds a result's rounding.
template <class W> long double magnitude(const W& x)
{ return static_cast<long double>(std::abs(x)); }

template <class W>
MArray::marray<long double, 1> magnitude(const MArray::marray<W, 1>& x)
{
    MArray::marray<long double, 1> m({x.length(0)});
    for (len_type i = 0; i < x.length(0); ++i) m[i] = magnitude(x[i]);
    return m;
}

template <class W>
MArray::marray<long double, 2> magnitude(const MArray::marray<W, 2>& x)
{
    MArray::marray<long double, 2> m({x.length(0), x.length(1)},
                                     0.0L,
                                     MArray::COLUMN_MAJOR);
    for (len_type i = 0; i < x.length(0); ++i)
        for (len_type j = 0; j < x.length(1); ++j) m[i][j] = magnitude(x[i][j]);
    return m;
}

/// Whether x is within tol·bound of expected.
template <class T, class W>
bool within(const T& x, const W& expected, long double tol, long double bound)
{ return std::abs(W(x) - expected) <= tol * bound; }

} // namespace famlies::test

#endif // FAMLIES_TESTS_SUPPORT_WIDE_HPP
