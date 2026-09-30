#ifndef FAMLIES_TESTS_SUPPORT_RANDOM_HPP
#define FAMLIES_TESTS_SUPPORT_RANDOM_HPP

#include <famlies/core/conj.hpp>
#include <famlies/core/operand.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <marray/marray.hpp>

#include <complex>
#include <random>

namespace famlies::test
{

/// A generator seeded with Catch's seed (--rng-seed), fresh for each test.
inline std::mt19937_64 make_rng()
{ return std::mt19937_64(Catch::getSeed()); }

/// Uniform in [-1, 1], in each part of a complex.
template <class T> T random_value(std::mt19937_64& rng)
{
    std::uniform_real_distribution<real_t<T>> part(-1, 1);
    if constexpr (is_complex_v<T>)
    {
        const real_t<T> re = part(rng);
        return {re, part(rng)};
    }
    else
        return part(rng);
}

template <class T>
void fill_random(MArray::marray_view<T, 2> a, std::mt19937_64& rng)
{
    for (len_type i = 0; i < a.length(0); ++i)
        for (len_type j = 0; j < a.length(1); ++j)
            a[i][j] = random_value<T>(rng);
}

template <class T>
void fill_random(MArray::marray_view<T, 1> x, std::mt19937_64& rng)
{
    for (len_type i = 0; i < x.length(0); ++i) x[i] = random_value<T>(rng);
}

/// Fill the elements of the square a where `where(i, j)`, a triangle, with
/// random values, and its diagonal with n plus a random value: diagonally
/// dominant, so well conditioned.
template <class T, class Where>
void
fill_triangular(MArray::marray_view<T, 2> a, Where where, std::mt19937_64& rng)
{
    const len_type n = a.length(0);
    for (len_type i = 0; i < n; ++i)
        for (len_type j = 0; j < n; ++j)
            if (i == j)
                a[i][j] = T(real_t<T>(n)) + random_value<T>(rng);
            else if (where(i, j))
                a[i][j] = random_value<T>(rng);
}

/// A Hermitian positive definite n x n matrix, B·Bᴴ + n·I for a random B,
/// with an exactly real diagonal.
template <class T>
MArray::marray<T, 2> random_hpd(len_type n, std::mt19937_64& rng)
{
    MArray::marray<T, 2> b({n, n}, T(0), MArray::COLUMN_MAJOR);
    fill_random(b.view(), rng);
    MArray::marray<T, 2> h({n, n}, T(0), MArray::COLUMN_MAJOR);
    for (len_type i = 0; i < n; ++i)
        for (len_type j = 0; j <= i; ++j)
        {
            T sum(0);
            for (len_type p = 0; p < n; ++p)
                sum += b[i][p] * detail::conjugate(b[j][p]);
            h[i][j] = sum;
            h[j][i] = detail::conjugate(sum);
        }
    for (len_type i = 0; i < n; ++i)
        h[i][i] = T(std::real(h[i][i]) + real_t<T>(n));
    return h;
}

} // namespace famlies::test

#endif // FAMLIES_TESTS_SUPPORT_RANDOM_HPP
