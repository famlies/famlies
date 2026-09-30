#ifndef FAMLIES_CORE_ERROR_HPP
#define FAMLIES_CORE_ERROR_HPP

#include <famlies/core/operand.hpp>

#include <cmath>
#include <complex>
#include <concepts>
#include <expected>

// Numerical failures, returned as std::expected<void, E>. Each E carries
// LAPACK's info meaning, relative to the operand the op was passed.

namespace famlies
{

/// The leading minor of this order (1-based) isn't positive definite, as
/// ?potrf's info reports it.
struct not_positive_definite
{
    len_type order;

    bool operator==(const not_positive_definite&) const = default;
};

/// e, moved from a block starting at index k to the operand holding it.
inline not_positive_definite offset(not_positive_definite e, len_type k)
{
    e.order += k;
    return e;
}

/// r, moved from a sub-operation on the block at `at` (an index, or a range)
/// to the operand holding that block.
template <class E, class At>
std::expected<void, E> offset(const std::expected<void, E>& r, const At& at)
{
    if (r)
        return r;
    if constexpr (std::integral<At>)
        return std::unexpected(offset(r.error(), len_type(at)));
    else
        return std::unexpected(offset(r.error(), len_type(at.from())));
}

namespace detail
{

// chol on an element, what the leaves' leaf_chol runs: its square root,
// unless its real part isn't positive (NaN included).
template <class A>
std::expected<void, not_positive_definite> chol_element(A&& a)
{
    using R = real_t<element_t<A>>;
    const R d = std::real(a);
    if (!(d > R(0)))
        return std::unexpected(not_positive_definite{1});
    a = element_t<A>(std::sqrt(d));
    return {};
}

} // namespace detail

} // namespace famlies

#endif // FAMLIES_CORE_ERROR_HPP
