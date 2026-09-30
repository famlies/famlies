#ifndef FAMLIES_CORE_CONJ_HPP
#define FAMLIES_CORE_CONJ_HPP

#include <famlies/core/operand.hpp>

#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

namespace famlies
{

template <class V> class conjugated_view;

template <class V, class... I> class conjugated_slice;

namespace detail
{

template <class A> inline constexpr bool is_conjugated = false;

template <class V>
inline constexpr bool is_conjugated<conjugated_view<V>> = true;

// The conjugate of x[i]...: of a view (an marray_slice is viewed first), or
// of an element reference. MArray's slices can't be copied or moved, so the
// indices are applied in one expression.
template <class X> auto conjugate_indexed(X&& x)
{
    if constexpr (field<std::remove_cvref_t<X>>)
        return conjugated_view<X>(x);
    else
    {
        auto v = x.view();
        return conjugated_view<decltype(v)>(v);
    }
}

template <class X, class I, class... Is>
auto conjugate_indexed(X&& x, const I& i, const Is&... is)
{ return conjugate_indexed(std::forward<X>(x)[i], is...); }

} // namespace detail

template <class A>
inline constexpr bool is_conjugated_v =
    detail::is_conjugated<std::remove_cvref_t<A>>;

/// The conjugate of an operand V: a view, or an element reference (rank 0).
/// It forwards `length`, indexing (giving conjugated operands) and `.T()`.
/// Ops take it only as an input, since BLIS can't conjugate an output. Made
/// by `conjugated` and `adjoint`, which also undo it.
template <class V> class conjugated_view
{
  public:
    using base_type = V;

    explicit conjugated_view(V base) : base_(base) {}

    /// The operand this conjugates.
    V base() const { return base_; }

    len_type length(int d) const requires(rank_v<V> >= 1)
    { return base_.length(d); }

    auto T() const requires(rank_v<V> >= 1)
    {
        auto t = base_.T();
        return conjugated_view<decltype(t)>(t);
    }

    /// Itself, so `x[R].view()` reads the same for every rank-1 operand.
    conjugated_view view() const requires(rank_v<V> >= 1) { return *this; }

    template <class I>
    auto operator[](const I& i) const requires(rank_v<V> >= 1)
    {
        if constexpr (rank_v<V> == 1)
            return detail::conjugate_indexed(base_, i);
        else
            return conjugated_slice<V, I>(base_, i);
    }

  private:
    V base_;
};

/// A conjugated operand indexed in some of its dimensions, as MArray's
/// marray_slice is for a view: index the rest, or take `.view()`.
template <class V, class... I> class conjugated_slice
{
  public:
    conjugated_slice(V base, const I&... i) : base_(base), indices_(i...) {}

    template <class J> auto operator[](const J& j) const
    {
        return std::apply(
            [&](const I&... i)
            {
                if constexpr (sizeof...(I) + 1 == rank_v<V>)
                    return detail::conjugate_indexed(base_, i..., j);
                else
                    return conjugated_slice<V, I..., J>(base_, i..., j);
            },
            indices_);
    }

    auto view() const
    {
        return std::apply([&](const I&... i)
                          { return detail::conjugate_indexed(base_, i...); },
                          indices_);
    }

  private:
    V base_;
    std::tuple<I...> indices_;
};

template <class V> struct operand_traits<conjugated_view<V>>
{
    static constexpr int rank = rank_v<V>;
    using element_type = element_t<V>;
    static constexpr bool is_view = true;
};

/// The conjugate of A. Conjugating twice gives A back, and on real types
/// it's A itself (an owning array as its view).
template <operand A> decltype(auto) conjugated(A&& a)
{
    if constexpr (is_conjugated_v<A>)
        return a.base();
    else if constexpr (!is_complex_v<element_t<A>>)
        return detail::as_view(std::forward<A>(a));
    else
    {
        using V = decltype(detail::as_view(std::forward<A>(a)));
        return conjugated_view<V>(detail::as_view(std::forward<A>(a)));
    }
}

/// The conjugate transpose of A: `conjugated(A.T())`, or `conjugated(A)` for
/// an element.
template <operand A> decltype(auto) adjoint(A&& a)
{
    if constexpr (rank_v<A> == 0)
        return conjugated(std::forward<A>(a));
    else
        return conjugated(a.T());
}

namespace detail
{

/// The unconjugated operand under A: what a leaf reads, with
/// `is_conjugated_v<A>` saying whether to conjugate it.
template <operand A> decltype(auto) strip_conj(A&& a)
{
    if constexpr (is_conjugated_v<A>)
        return a.base();
    else
        return as_view(std::forward<A>(a));
}

/// x's conjugate, of x's own type (std::conj makes a real x complex).
template <field T> T conjugate(const T& x)
{
    if constexpr (is_complex_v<T>)
        return std::conj(x);
    else
        return x;
}

/// An element's value, conjugated if the element is.
template <operand A> requires(rank_v<A> == 0) element_t<A> value(A&& a)
{
    if constexpr (is_conjugated_v<A>)
        return conjugate(element_t<A>(a.base()));
    else
        return a;
}

} // namespace detail

} // namespace famlies

#endif // FAMLIES_CORE_CONJ_HPP
