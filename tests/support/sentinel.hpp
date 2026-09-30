#ifndef FAMLIES_TESTS_SUPPORT_SENTINEL_HPP
#define FAMLIES_TESTS_SUPPORT_SENTINEL_HPP

#include <famlies/core/operand.hpp>

#include <marray/marray.hpp>

#include <bit>
#include <complex>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <vector>

namespace famlies::test
{

/// What fills whatever an op must neither read nor write: a NaN with a
/// payload of its own. A read spreads NaN into the results, and a write of
/// any other value changes its bits.
template <class T> T sentinel()
{
    if constexpr (is_complex_v<T>)
        return {sentinel<real_t<T>>(), sentinel<real_t<T>>()};
    else if constexpr (std::same_as<T, float>)
        return std::bit_cast<float>(std::uint32_t{0x7fe5a5a5});
    else
    {
        static_assert(std::same_as<T, double>);
        return std::bit_cast<double>(std::uint64_t{0x7ffa5a5a5a5a5a5a});
    }
}

/// Whether x has the sentinel's bits.
template <class T> bool is_sentinel(const T& x)
{
    const T s = sentinel<T>();
    return std::memcmp(&x, &s, sizeof(T)) == 0;
}

/// Where predicates over a matrix element's (i, j).
inline constexpr auto everywhere = [](len_type, len_type) { return true; };
inline constexpr auto strictly_lower = [](len_type i, len_type j)
{ return i > j; };
inline constexpr auto strictly_upper = [](len_type i, len_type j)
{ return i < j; };
inline constexpr auto on_diagonal = [](len_type i, len_type j)
{ return i == j; };

/// Fill the elements of a where `where(i, j)` with the sentinel.
template <class T, class Where>
void fill_sentinel(MArray::marray_view<T, 2> a, Where where)
{
    for (len_type i = 0; i < a.length(0); ++i)
        for (len_type j = 0; j < a.length(1); ++j)
            if (where(i, j))
                a[i][j] = sentinel<T>();
}

/// Whether the elements of a where `where(i, j)` still hold the sentinel.
template <class T, class Where>
bool sentinel_intact(MArray::marray_view<T, 2> a, Where where)
{
    for (len_type i = 0; i < a.length(0); ++i)
        for (len_type j = 0; j < a.length(1); ++j)
            if (where(i, j) && !is_sentinel(a[i][j]))
                return false;
    return true;
}

/// A's elements, to check later that an op left them bit for bit.
template <class T> std::vector<T> snapshot(MArray::marray_view<T, 2> a)
{
    std::vector<T> out;
    for (len_type i = 0; i < a.length(0); ++i)
        for (len_type j = 0; j < a.length(1); ++j) out.push_back(a[i][j]);
    return out;
}

/// Whether a's elements have the bits they had in `before`.
template <class T>
bool unchanged(MArray::marray_view<T, 2> a, const std::vector<T>& before)
{
    std::size_t k = 0;
    for (len_type i = 0; i < a.length(0); ++i)
        for (len_type j = 0; j < a.length(1); ++j)
            if (std::memcmp(&a[i][j], &before[k++], sizeof(T)) != 0)
                return false;
    return true;
}

} // namespace famlies::test

#endif // FAMLIES_TESTS_SUPPORT_SENTINEL_HPP
