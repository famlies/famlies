#ifndef FAMLIES_CORE_PRECONDITION_HPP
#define FAMLIES_CORE_PRECONDITION_HPP

#include <famlies/core/operand.hpp>

#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

/// FAMLIES_EXPECTS(op, condition, args...): unless `condition` holds, print
/// `famlies::<op>: precondition failed: <condition>` and each arg (an
/// operand's lengths, an integer's value), then abort. With
/// FAMLIES_NO_PRECONDITION_CHECKS defined, `op` and `condition` are compiled
/// but not evaluated. Define it in every TU or none (ODR); the CMake option
/// of that name defines it for every target linking famlies.
#ifdef FAMLIES_NO_PRECONDITION_CHECKS
#define FAMLIES_EXPECTS(op, condition, ...) \
    (static_cast<void>(sizeof(op)),         \
     static_cast<void>(sizeof(static_cast<bool>(condition))))
#else
#define FAMLIES_EXPECTS(op, condition, ...)                                   \
    (static_cast<bool>(condition)                                             \
         ? static_cast<void>(0)                                               \
         : ::famlies::detail::precondition_failed(op,                         \
                                                  #condition,                 \
                                                  #__VA_ARGS__ __VA_OPT__(, ) \
                                                      __VA_ARGS__))
#endif

namespace famlies::detail
{

// The next name in a stringized argument list ("A, B[R1][R2], f(x, y)"),
// which ends at a comma outside brackets.
inline std::string_view next_name(std::string_view& names)
{
    int depth = 0;
    std::size_t end = 0;
    for (; end < names.size(); ++end)
    {
        const char c = names[end];
        if (c == '(' || c == '[' || c == '{')
            ++depth;
        else if (c == ')' || c == ']' || c == '}')
            --depth;
        else if (c == ',' && depth == 0)
            break;
    }
    std::string_view name = names.substr(0, end);
    names.remove_prefix(end < names.size() ? end + 1 : end);
    while (!name.empty() && name.front() == ' ') name.remove_prefix(1);
    while (!name.empty() && name.back() == ' ') name.remove_suffix(1);
    return name;
}

template <class A>
void describe(std::string& out, std::string_view name, const A& a)
{
    out += "  ";
    out += name;
    if constexpr (std::integral<A>)
    {
        out += " = ";
        out += std::to_string(a);
    }
    else if constexpr (rank_v<A> == 0)
        out += ": element";
    else
    {
        out += ": ";
        for (int d = 0; d < rank_v<A>; ++d)
        {
            if (d > 0)
                out += " x ";
            out += std::to_string(a.length(d));
        }
    }
    out += '\n';
}

template <class... Args>
std::string precondition_message(std::string_view op,
                                 std::string_view condition,
                                 [[maybe_unused]] std::string_view names,
                                 const Args&... args)
{
    std::string out = "famlies::";
    out += op;
    out += ": precondition failed: ";
    out += condition;
    out += '\n';
    (describe(out, next_name(names), args), ...);
    return out;
}

template <class... Args>
[[noreturn]] void precondition_failed(std::string_view op,
                                      std::string_view condition,
                                      std::string_view names,
                                      const Args&... args)
{
    const std::string message =
        precondition_message(op, condition, names, args...);
    std::fputs(message.c_str(), stderr);
    std::fflush(stderr);
    std::abort();
}

} // namespace famlies::detail

#endif // FAMLIES_CORE_PRECONDITION_HPP
