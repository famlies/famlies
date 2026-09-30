#ifndef FAMLIES_CORE_SWEEP_HPP
#define FAMLIES_CORE_SWEEP_HPP

#include <famlies/core/operand.hpp>
#include <famlies/core/precondition.hpp>

#include <marray/marray.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace famlies
{

/// A sweep's direction: from index 0 up, or from n down.
struct forward_t
{ explicit forward_t() = default; };

inline constexpr forward_t forward{};

struct backward_t
{ explicit backward_t() = default; };

inline constexpr backward_t backward{};

/// The compile-time block size 1, with which a sweep's R1 is an index.
using one_t = std::integral_constant<len_type, 1>;

inline constexpr one_t one{};

/// One step of a sweep over [0, n): R0, R1 and R2 split it in that order,
/// R1 being the block the step exposes.
template <class R1> struct sweep_step
{
    MArray::range_t<len_type> r0;
    R1 r1;
    MArray::range_t<len_type> r2;
};

/// The steps of `sweep`. Step is `one_t` or `len_type`.
template <class Direction, class Step> class sweep_view
{
    static constexpr bool is_forward = std::same_as<Direction, forward_t>;

  public:
    class iterator
    {
      public:
        using value_type =
            sweep_step<std::conditional_t<std::same_as<Step, one_t>,
                                          len_type,
                                          MArray::range_t<len_type>>>;
        using difference_type = std::ptrdiff_t;

        iterator() = default;

        value_type operator*() const
        {
            // k is where R1 starts going forward, and where it ends going
            // backward, so the block that's short is the last one.
            const len_type b = width();
            if constexpr (is_forward)
                return step(k_, k_ + b);
            else
                return step(k_ - b, k_);
        }

        iterator& operator++()
        {
            if constexpr (is_forward)
                k_ += width();
            else
                k_ -= width();
            return *this;
        }

        iterator operator++(int)
        {
            iterator old = *this;
            ++*this;
            return old;
        }

        friend bool operator==(const iterator& a, const iterator& b)
        { return a.n_ == b.n_ && a.k_ == b.k_ && a.bs_ == b.bs_; }

        friend bool operator==(const iterator& it, std::default_sentinel_t)
        { return it.k_ == (is_forward ? it.n_ : 0); }

      private:
        friend sweep_view;

        iterator(len_type n, len_type k, Step bs) : n_(n), k_(k), bs_(bs) {}

        len_type width() const
        { return std::min<len_type>(bs_, is_forward ? n_ - k_ : k_); }

        value_type step(len_type from, len_type to) const
        {
            if constexpr (std::same_as<Step, one_t>)
                return {MArray::range(len_type(0), from),
                        from,
                        MArray::range(to, n_)};
            else
                return {MArray::range(len_type(0), from),
                        MArray::range(from, to),
                        MArray::range(to, n_)};
        }

        len_type n_ = 0;
        len_type k_ = 0;
        Step bs_ = {};
    };

    sweep_view(len_type n, Step bs) : n_(n), bs_(bs) {}

    iterator begin() const { return {n_, is_forward ? 0 : n_, bs_}; }

    std::default_sentinel_t end() const { return {}; }

  private:
    len_type n_;
    Step bs_;
};

/// A sweep's block size: `one`, or an integer.
template <class B>
concept sweep_block_size = std::same_as<B, one_t> || std::integral<B>;

template <class D>
concept sweep_direction =
    std::same_as<D, forward_t> || std::same_as<D, backward_t>;

/// The partitioning loop: `for (auto [R0, R1, R2] : sweep(n, bs, dir))`
/// exposes [0, n) block by block as R1, with R0 before it and R2 after.
/// Forward goes from 0 up and backward from n down, and each block has bs
/// indices except the last, which has what's left. With bs = `one`, R1 is an
/// index; otherwise every R is a range.
template <sweep_block_size B, sweep_direction Direction>
auto sweep(len_type n, B bs, Direction)
{
    FAMLIES_EXPECTS("sweep", n >= 0, n);
    if constexpr (std::same_as<B, one_t>)
        return sweep_view<Direction, one_t>(n, bs);
    else
    {
        FAMLIES_EXPECTS("sweep", bs >= 1, bs);
        return sweep_view<Direction, len_type>(n, static_cast<len_type>(bs));
    }
}

} // namespace famlies

#endif // FAMLIES_CORE_SWEEP_HPP
