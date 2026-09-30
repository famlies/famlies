#ifndef FAMLIES_CORE_CONTROL_HPP
#define FAMLIES_CORE_CONTROL_HPP

#include <famlies/core/operand.hpp>

#include <type_traits>

// Control trees. Each op defines its node structs: <op>_unblocked and
// <op>_blocked (and <op>_recursive where it has one), aggregates whose first
// member is the variant and whose type deduces from their initializers:
//
//     chol_blocked{.variant = chol_right_looking{}, .nb = 256,
//                  .chol = chol_blocked{.variant = chol_right_looking{},
//                                       .nb = 32}}
//
// A variant is a tag type the op defines, named after what its loop does
// (chol_bordered, chol_left_looking, chol_right_looking). A node's `variant`
// is initialized with a tag's prvalue, not a variable: Clang can't deduce a
// node's type inside a template from a designated initializer naming a
// namespace-scope variable.
//
// Every child is a member named after the sub-operation it runs, typed by a
// template parameter whose default is the child's default subtree.

namespace famlies
{

/// Leaves run BLAS-level ops. A leaf L sets this and specializes
/// `leaf_takes`, and in L's namespace, where argument-dependent lookup finds
/// them, defines `default_block_size<T>(const L&)` and the `leaf_<op>(const
/// L&, ...)` its trees reach, as `famlies::blis` and `famlies::reference` do.
/// The public ops check ranks and lengths before calling a `leaf_<op>`.
template <class N> inline constexpr bool enable_leaf = false;

template <class N> concept leaf = enable_leaf<std::remove_cvref_t<N>>;

/// Chooses a subtree at run time: `then` where `when`, called with the op's
/// operands as lvalues in the op's order, is true, and `otherwise` where it's
/// false. The subtree gets the operands as lvalues too. An op rejects a
/// select whose `when` can't be called so, or doesn't give a bool. Nest them
/// for more choices.
template <class When, class Then, class Otherwise> struct select
{
    When when;
    Then then;
    Otherwise otherwise;
};

template <class N> inline constexpr bool is_select_v = false;

template <class When, class Then, class Otherwise>
inline constexpr bool is_select_v<select<When, Then, Otherwise>> = true;

namespace detail
{

template <class N> inline constexpr bool is_leaf_tree = leaf<N>;

template <class When, class Then, class Otherwise>
inline constexpr bool is_leaf_tree<select<When, Then, Otherwise>> =
    is_leaf_tree<Then> && is_leaf_tree<Otherwise>;

} // namespace detail

/// What runs a BLAS-level op, or an op on an element: a leaf, or a select
/// of leaf trees.
template <class N>
concept leaf_tree = detail::is_leaf_tree<std::remove_cvref_t<N>>;

/// Whether leaf N takes operand A, as a forwarding reference deduces it, and
/// with it the operands that indexing A gives and their conjugates. Each
/// leaf's header specializes it.
template <class N, class A> inline constexpr bool leaf_takes = false;

/// Whether every leaf in leaf tree N takes each of the operands As.
template <class N, class... As>
inline constexpr bool leaf_tree_takes = (leaf_takes<N, As> && ...);

template <class When, class Then, class Otherwise, class... As>
inline constexpr bool leaf_tree_takes<select<When, Then, Otherwise>, As...> =
    leaf_tree_takes<Then, As...> && leaf_tree_takes<Otherwise, As...>;

/// A blocked node's block size when its `nb` is 0 comes from a leaf tree:
/// each leaf's header defines it for its leaf, and a select's is its
/// `then`'s.
template <class T, class When, class Then, class Otherwise>
len_type default_block_size(const select<When, Then, Otherwise>& s)
{ return default_block_size<T>(s.then); }

} // namespace famlies

#endif // FAMLIES_CORE_CONTROL_HPP
