#ifndef FAMLIES_TESTS_SUPPORT_STORAGE_HPP
#define FAMLIES_TESTS_SUPPORT_STORAGE_HPP

#include "support/sentinel.hpp"

#include <famlies/core/operand.hpp>

#include <marray/marray.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace famlies::test
{

enum class storage
{
    column_major,
    row_major,
    general, // non-unit row and column strides
};

inline constexpr std::array storages = {storage::column_major,
                                        storage::row_major,
                                        storage::general};

inline const char* to_string(storage s)
{
    switch (s)
    {
        case storage::column_major: return "column-major";
        case storage::row_major:    return "row-major";
        case storage::general:      return "general";
    }
    return "?";
}

/// An owned m x n matrix in the given storage, with gaps between its
/// elements and guards before and after them. Gaps, guards and elements all
/// start as the sentinel.
template <class T> class test_matrix
{
    static constexpr len_type guard = 2;

  public:
    test_matrix(len_type m, len_type n, storage s)
    {
        // Column-major and row-major pad each column or row by one; general
        // leaves a gap after every element of a column and one more after
        // the column.
        const stride_type rs = s == storage::column_major ? 1
                             : s == storage::row_major    ? n + 1
                                                          : 2;
        const stride_type cs = s == storage::column_major ? m + 1
                             : s == storage::row_major    ? 1
                                                          : 2 * m + 1;
        const len_type span =
            m > 0 && n > 0 ? (m - 1) * rs + (n - 1) * cs + 1 : 0;

        buffer_.assign(static_cast<std::size_t>(span + 2 * guard),
                       sentinel<T>());
        is_element_.assign(buffer_.size(), false);
        for (len_type i = 0; i < m; ++i)
            for (len_type j = 0; j < n; ++j)
                is_element_[static_cast<std::size_t>(guard + i * rs + j * cs)] =
                    true;
        view_.reset({m, n}, buffer_.data() + guard, {rs, cs});
    }

    // The view points into the buffer.
    test_matrix(const test_matrix&) = delete;
    test_matrix& operator=(const test_matrix&) = delete;

    MArray::marray_view<T, 2> view() const { return view_; }

    /// Whether every position that isn't an element holds the sentinel.
    bool gaps_intact() const
    {
        for (std::size_t p = 0; p < buffer_.size(); ++p)
            if (!is_element_[p] && !is_sentinel(buffer_[p]))
                return false;
        return true;
    }

  private:
    std::vector<T> buffer_;
    std::vector<bool> is_element_;
    MArray::marray_view<T, 2> view_;
};

/// An owned vector of length n: the column of an n x 1 test_matrix, so its
/// stride is 1 (column-major) or 2 (the others), with gaps and guards.
template <class T> class test_vector
{
  public:
    test_vector(len_type n, storage s) : matrix_(n, 1, s) {}

    MArray::marray_view<T, 1> view() const
    { return matrix_.view()[MArray::slice::all][0]; }

    bool gaps_intact() const { return matrix_.gaps_intact(); }

  private:
    test_matrix<T> matrix_;
};

} // namespace famlies::test

#endif // FAMLIES_TESTS_SUPPORT_STORAGE_HPP
