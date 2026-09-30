#ifndef FAMLIES_CORE_TAGS_HPP
#define FAMLIES_CORE_TAGS_HPP

#include <concepts>

// Structure tags, spelled as in std::linalg.

namespace famlies
{

struct left_side_t
{ explicit left_side_t() = default; };

inline constexpr left_side_t left_side{};

struct right_side_t
{ explicit right_side_t() = default; };

inline constexpr right_side_t right_side{};

struct lower_triangle_t
{ explicit lower_triangle_t() = default; };

inline constexpr lower_triangle_t lower_triangle{};

struct upper_triangle_t
{ explicit upper_triangle_t() = default; };

inline constexpr upper_triangle_t upper_triangle{};

struct implicit_unit_diagonal_t
{ explicit implicit_unit_diagonal_t() = default; };

inline constexpr implicit_unit_diagonal_t implicit_unit_diagonal{};

struct explicit_diagonal_t
{ explicit explicit_diagonal_t() = default; };

inline constexpr explicit_diagonal_t explicit_diagonal{};

template <class S>
concept side = std::same_as<S, left_side_t> || std::same_as<S, right_side_t>;

template <class T>
concept triangle =
    std::same_as<T, lower_triangle_t> || std::same_as<T, upper_triangle_t>;

template <class D>
concept diagonal = std::same_as<D, implicit_unit_diagonal_t>
                || std::same_as<D, explicit_diagonal_t>;

} // namespace famlies

#endif // FAMLIES_CORE_TAGS_HPP
