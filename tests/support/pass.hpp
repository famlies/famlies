#ifndef FAMLIES_TESTS_SUPPORT_PASS_HPP
#define FAMLIES_TESTS_SUPPORT_PASS_HPP

#include <famlies/core/conj.hpp>
#include <famlies/core/operand.hpp>

#include <array>
#include <utility>

// Running one test body over tags and operands chosen at run time, which
// have a type each.

namespace famlies::test
{

/// How a test passes an operand: as it is, or conjugated, transposed or
/// both.
enum class pass
{
    as_is,
    conjugated,
    transposed,
    adjoint,
};

inline constexpr std::array passes = {pass::as_is,
                                      pass::conjugated,
                                      pass::transposed,
                                      pass::adjoint};

/// For vectors and elements, where transposing does nothing.
inline constexpr std::array vector_passes = {pass::as_is, pass::conjugated};

inline const char* to_string(pass p)
{
    switch (p)
    {
        case pass::as_is:      return "as is";
        case pass::conjugated: return "conjugated";
        case pass::transposed: return "transposed";
        case pass::adjoint:    return "adjoint";
    }
    return "?";
}

/// Whether p transposes.
inline bool transposes(pass p)
{ return p == pass::transposed || p == pass::adjoint; }

/// f(a as p says). A vector or element is never transposed.
template <class A, class F> void passed(pass p, A&& a, F&& f)
{
    switch (p)
    {
        case pass::as_is:      return f(std::forward<A>(a));
        case pass::conjugated: return f(conjugated(std::forward<A>(a)));
        case pass::transposed:
            if constexpr (rank_v<A> == 2)
                return f(a.T());
            else
                return f(std::forward<A>(a));
        case pass::adjoint: return f(adjoint(std::forward<A>(a)));
    }
}

/// f(a) if c, else f(b): a runtime choice between two tags.
template <class A, class B, class F> void with(bool c, A a, B b, F&& f)
{
    if (c)
        f(a);
    else
        f(b);
}

} // namespace famlies::test

#endif // FAMLIES_TESTS_SUPPORT_PASS_HPP
