#ifndef FAMLIES_CORE_OPERAND_HPP
#define FAMLIES_CORE_OPERAND_HPP

#include <marray/marray.hpp>

#include <complex>
#include <concepts>
#include <type_traits>

namespace famlies
{

using MArray::len_type;
using MArray::stride_type;

/// Element types: floating-point types and their complex types. Specialize
/// `enable_field` to admit another field type.
template <class T>
inline constexpr bool enable_field = std::is_floating_point_v<T>;

template <class R>
inline constexpr bool enable_field<std::complex<R>> = enable_field<R>;

template <class T> concept field = enable_field<std::remove_cv_t<T>>;

template <class T> inline constexpr bool is_complex_v = false;

template <class R> inline constexpr bool is_complex_v<std::complex<R>> = true;

template <class T> struct real_type
{
    using type = T;
};

template <class R> struct real_type<std::complex<R>>
{
    using type = R;
};

template <class T> using real_t = typename real_type<T>::type;

/// An operand type's rank and element type, and whether it's a view, which
/// may be passed by value. Anything else (an owning array, an element) is an
/// operand only as an lvalue. Specialize it for a new operand type.
template <class A> struct operand_traits
{
};

template <class T, int N> requires(N >= 1)
struct operand_traits<MArray::marray_view<T, N>>
{
    static constexpr int rank = N;
    using element_type = std::remove_const_t<T>;
    static constexpr bool is_view = true;
};

template <class T, int N, class Allocator> requires(N >= 1)
struct operand_traits<MArray::marray<T, N, Allocator>>
{
    static constexpr int rank = N;
    using element_type = std::remove_const_t<T>;
    static constexpr bool is_view = false;
};

template <field T> struct operand_traits<T>
{
    static constexpr int rank = 0;
    using element_type = std::remove_const_t<T>;
    static constexpr bool is_view = false;
};

template <class A>
inline constexpr int rank_v = operand_traits<std::remove_cvref_t<A>>::rank;

template <class A>
using element_t = typename operand_traits<std::remove_cvref_t<A>>::element_type;

namespace detail
{

// A, as a forwarding reference deduces it, has operand traits and a value
// category it may be passed with.
template <class A>
concept operand_shape =
    requires {
        {
            operand_traits<std::remove_cvref_t<A>>::rank
        } -> std::convertible_to<int>;
        typename operand_traits<std::remove_cvref_t<A>>::element_type;
        {
            operand_traits<std::remove_cvref_t<A>>::is_view
        } -> std::convertible_to<bool>;
    }
    && (operand_traits<std::remove_cvref_t<A>>::is_view
        || std::is_lvalue_reference_v<A>);

template <class A, int N>
concept operand_of_rank = operand_shape<A> && rank_v<A> == N;

template <class V>
concept array_interface = requires(const V& a, int d) {
    { a.length(d) } -> std::convertible_to<len_type>;
    { a.T() } -> operand_of_rank<rank_v<V>>;
};

// A rank-1 range index gives an marray_slice, not a view, so it's followed
// by .view().
template <class V>
concept vector_indexing =
    requires(const V& a, MArray::range_t<len_type> r, len_type i) {
        { a[i] } -> operand_of_rank<0>;
        { a[r].view() } -> operand_of_rank<1>;
    };

template <class V>
concept matrix_indexing =
    requires(const V& a, MArray::range_t<len_type> r, len_type i) {
        { a[r][r] } -> operand_of_rank<2>;
        { a[r][i] } -> operand_of_rank<1>;
        { a[i][r] } -> operand_of_rank<1>;
        { a[i][i] } -> operand_of_rank<0>;
        { a[r].view() } -> operand_of_rank<2>;
        { a[i].view() } -> operand_of_rank<1>;
    };

// What an operand of V's rank provides, past its traits.
template <class V> consteval bool has_operand_interface()
{
    if constexpr (rank_v<V> == 0)
        return true;
    else if constexpr (!array_interface<V>)
        return false;
    else if constexpr (rank_v<V> == 1)
        return vector_indexing<V>;
    else if constexpr (rank_v<V> == 2)
        return matrix_indexing<V>;
    else
        return true;
}

} // namespace detail

/// What ops and algorithms take: A as a forwarding reference deduces it.
/// Arrays have lengths, `.T()`, and indices or ranges in each dimension that
/// give operands of lower or equal rank. An element (rank 0) is an lvalue.
template <class A>
concept operand = detail::operand_shape<A>
               && detail::has_operand_interface<std::remove_cvref_t<A>>();

namespace detail
{

// An element of A can be assigned through A: not a conjugate, nor an
// operand of const elements. Checked at ranks 0 to 2.
template <class A> consteval bool has_writable_elements()
{
    if constexpr (rank_v<A> == 0)
        return requires(A&& a, const element_t<A>& x) {
            static_cast<A&&>(a) = x;
        };
    else if constexpr (rank_v<A> == 1)
        return requires(A&& a, len_type i, const element_t<A>& x) { a[i] = x; };
    else if constexpr (rank_v<A> == 2)
        return requires(A&& a, len_type i, const element_t<A>& x) {
            a[i][i] = x;
        };
    else
        return true;
}

} // namespace detail

/// An operand an op may write: not conjugated, and not of const elements.
template <class A>
concept output_operand = operand<A> && detail::has_writable_elements<A>();

namespace detail
{

/// The operand as one that can be held: an owning array as its view, and
/// anything else as it is (an element stays a reference).
template <operand A> decltype(auto) as_view(A&& a)
{
    using U = std::remove_cvref_t<A>;
    if constexpr (field<U>)
        return static_cast<A&&>(a);
    else if constexpr (operand_traits<U>::is_view)
        return U(a);
    else
        return a.view();
}

} // namespace detail

} // namespace famlies

#endif // FAMLIES_CORE_OPERAND_HPP
