// The control-tree spelling, on a mock op shaped like Cholesky: a
// diagonal-block sub-op (`op`, run by a leaf on an element in unblocked
// nodes) and a BLAS-level sub-op (`gemm`), with two variants, `op_eager` and
// `op_lazy`. Running a tree records what runs.

#include <famlies/core/control.hpp>
#include <famlies/core/operand.hpp>
#include <famlies/core/sweep.hpp>
#include <famlies/leaf/blis.hpp>
#include <famlies/leaf/reference.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <format>
#include <string>
#include <type_traits>
#include <vector>

namespace mock
{

using famlies::len_type;

/// Each step runs the sub-op on the diagonal block, then gemm on what's ahead.
struct op_eager
{
};

/// Each step runs gemm on what's behind, then the sub-op on the diagonal block.
struct op_lazy
{
};

inline std::string name(op_eager)
{ return "eager"; }

inline std::string name(op_lazy)
{ return "lazy"; }

template <class V, class Op = famlies::blis, class Gemm = famlies::blis>
struct op_unblocked
{
    [[no_unique_address]] V variant;
    Op op{};
    Gemm gemm{};
};

template <class V, class Op = op_unblocked<V>, class Gemm = famlies::blis>
struct op_blocked
{
    [[no_unique_address]] V variant;
    len_type nb = 0;
    Op op{};
    Gemm gemm{};
};

template <class Op = op_unblocked<op_eager>, class Gemm = famlies::blis>
struct op_recursive
{
    len_type cutoff = 16;
    Op op{};
    Gemm gemm{};
};

template <class V>
inline constexpr bool is_variant =
    std::same_as<V, op_eager> || std::same_as<V, op_lazy>;

// What runs a BLAS-level sub-op, or the op on an element.
template <class N> inline constexpr bool leaf_tree = famlies::leaf<N>;

template <class W, class T, class O>
inline constexpr bool leaf_tree<famlies::select<W, T, O>> =
    leaf_tree<T> && leaf_tree<O>;

template <class N> inline constexpr bool op_tree = false;

template <class V, class Op, class Gemm>
inline constexpr bool op_tree<op_unblocked<V, Op, Gemm>> =
    is_variant<V> && leaf_tree<Op> && leaf_tree<Gemm>;

template <class V, class Op, class Gemm>
inline constexpr bool op_tree<op_blocked<V, Op, Gemm>> =
    is_variant<V> && op_tree<Op> && leaf_tree<Gemm>;

template <class Op, class Gemm>
inline constexpr bool op_tree<op_recursive<Op, Gemm>> =
    op_tree<Op> && leaf_tree<Gemm>;

template <class W, class T, class O>
inline constexpr bool op_tree<famlies::select<W, T, O>> =
    op_tree<T> && op_tree<O>;

template <class N> concept op_control = op_tree<std::remove_cvref_t<N>>;

template <class N> inline constexpr bool is_recursive = false;

template <class Op, class Gemm>
inline constexpr bool is_recursive<op_recursive<Op, Gemm>> = true;

template <class N> inline constexpr bool is_blocked = false;

template <class V, class Op, class Gemm>
inline constexpr bool is_blocked<op_blocked<V, Op, Gemm>> = true;

using trace = std::vector<std::string>;

inline std::string name(const famlies::blis&)
{ return "blis"; }

inline std::string name(const famlies::reference&)
{ return "reference"; }

template <class Leaf>
void run_leaf(trace& t, const char* sub, len_type n, const Leaf& leaf)
{
    if constexpr (famlies::is_select_v<Leaf>)
        return leaf.when(n) ? run_leaf(t, sub, n, leaf.then)
                            : run_leaf(t, sub, n, leaf.otherwise);
    else
        t.push_back(std::format("{}:{} {}", sub, name(leaf), n));
}

template <op_control Ctl> void op(trace& t, len_type n, const Ctl& ctl)
{
    if constexpr (famlies::is_select_v<Ctl>)
        return ctl.when(n) ? op(t, n, ctl.then) : op(t, n, ctl.otherwise);
    else if constexpr (is_recursive<Ctl>)
    {
        if (n <= ctl.cutoff)
            return op(t, n, ctl.op);
        t.push_back(std::format("recursive {}", n));
        const len_type n1 = n / 2;
        op(t, n1, ctl);
        run_leaf(t, "gemm", n - n1, ctl.gemm);
        op(t, n - n1, ctl);
    }
    else
    {
        // One variant text serves unblocked and blocked nodes.
        using K = decltype(ctl.variant);
        // nb = 0 is gemm's leaf's default.
        const auto bs = [&]
        {
            if constexpr (is_blocked<Ctl>)
            {
                if constexpr (famlies::leaf<decltype(ctl.gemm)>)
                    if (ctl.nb == 0)
                        return famlies::default_block_size<double>(ctl.gemm);
                return ctl.nb;
            }
            else
                return famlies::one;
        }();
        t.push_back(
            is_blocked<Ctl>
                ? std::format("blocked {} nb={} {}",
                              name(ctl.variant),
                              len_type(bs),
                              n)
                : std::format("unblocked {} {}", name(ctl.variant), n));
        for (auto [R0, R1, R2] : famlies::sweep(n, bs, famlies::forward))
        {
            const auto sub = [&]
            {
                if constexpr (is_blocked<Ctl>)
                    op(t, R1.size(), ctl.op);
                else
                    run_leaf(t, "op", 1, ctl.op);
            };
            if constexpr (std::same_as<K, op_eager>)
            {
                sub();
                run_leaf(t, "gemm", R2.size(), ctl.gemm);
            }
            else
            {
                run_leaf(t, "gemm", R0.size(), ctl.gemm);
                sub();
            }
        }
    }
}

/// The call form with the tree as a template argument.
template <auto Tree> requires op_control<decltype(Tree)>
void op(trace& t, len_type n)
{ op(t, n, Tree); }

} // namespace mock

using mock::op;
using mock::op_blocked;
using mock::op_control;
using mock::op_eager;
using mock::op_lazy;
using mock::op_recursive;
using mock::op_unblocked;
using mock::trace;

namespace
{

template <class Ctl> trace run(famlies::len_type n, const Ctl& ctl)
{
    trace t;
    op(t, n, ctl);
    return t;
}

// A tag the op doesn't have as a variant.
struct not_a_variant
{
};

template <class Ctl>
constexpr bool callable = requires(trace& t, const Ctl& ctl) { op(t, 3, ctl); };

} // namespace

TEST_CASE("a node's type deduces from its initializers", "[control]")
{
    constexpr op_blocked tree{
        .variant = op_lazy{},
        .nb = 256,
        .op = op_blocked{.variant = op_eager{}, .nb = 32}
    };
    STATIC_CHECK(
        std::is_same_v<
            decltype(tree),
            const op_blocked<op_lazy,
                             op_blocked<op_eager, op_unblocked<op_eager>>>>);
    STATIC_CHECK(tree.op.nb == 32);

    // Children not given take their defaults: the diagonal-block sub-op an
    // unblocked node of the same variant, BLAS-level sub-ops the blis leaf.
    constexpr op_blocked defaults{.variant = op_eager{}};
    STATIC_CHECK(
        std::is_same_v<decltype(defaults.op), op_unblocked<op_eager>>);
    STATIC_CHECK(std::is_same_v<decltype(defaults.gemm), famlies::blis>);
    STATIC_CHECK(defaults.nb == 0);

    // Any child is tuned by name, leaves included.
    constexpr op_blocked tuned{.variant = op_eager{},
                               .nb = 64,
                               .gemm = famlies::blis{.threads = 4}};
    STATIC_CHECK(tuned.gemm.threads == 4);
    constexpr op_unblocked on_reference{.variant = op_lazy{},
                                        .op = famlies::reference{},
                                        .gemm = famlies::reference{}};
    STATIC_CHECK(op_control<decltype(on_reference)>);
}

TEMPLATE_TEST_CASE("a node's type deduces inside a template",
                   "[control]",
                   int,
                   double)
{
    constexpr op_blocked tree{
        .variant = op_lazy{},
        .nb = 8,
        .op =
            op_unblocked{.variant = op_eager{}, .gemm = famlies::reference{}}
    };
    STATIC_CHECK(
        std::is_same_v<
            decltype(tree),
            const op_blocked<
                op_lazy,
                op_unblocked<op_eager, famlies::blis, famlies::reference>>>);
    CHECK(run(2, tree).at(0) == "blocked lazy nb=8 2");
}

TEST_CASE("an unblocked node runs its leaves element by element", "[control]")
{
    CHECK(run(3, op_unblocked{.variant = op_eager{}})
          == trace{"unblocked eager 3",
                   "op:blis 1",
                   "gemm:blis 2",
                   "op:blis 1",
                   "gemm:blis 1",
                   "op:blis 1",
                   "gemm:blis 0"});
    CHECK(run(2,
              op_unblocked{.variant = op_lazy{}, .gemm = famlies::reference{}})
          == trace{"unblocked lazy 2",
                   "gemm:reference 0",
                   "op:blis 1",
                   "gemm:reference 1",
                   "op:blis 1"});
}

TEST_CASE("a blocked node runs its child on each diagonal block", "[control]")
{
    CHECK(run(3, op_blocked{.variant = op_lazy{}, .nb = 2})
          == trace{"blocked lazy nb=2 3",
                   "gemm:blis 0",
                   "unblocked lazy 2",
                   "gemm:blis 0",
                   "op:blis 1",
                   "gemm:blis 1",
                   "op:blis 1",
                   "gemm:blis 2",
                   "unblocked lazy 1",
                   "gemm:blis 0",
                   "op:blis 1"});

    // Nested blocking, each level its own variant.
    const trace t =
        run(4,
            op_blocked{
                .variant = op_eager{},
                .nb = 4,
                .op = op_blocked{.variant = op_lazy{},
                                 .nb = 2,
                                 .op = op_unblocked{.variant = op_eager{}}}
    });
    CHECK(t.at(0) == "blocked eager nb=4 4");
    CHECK(t.at(1) == "blocked lazy nb=2 4");
    CHECK(t.at(3) == "unblocked eager 2");

    // nb = 0 is the leaf's default block size.
    const auto kc = famlies::default_block_size<double>(famlies::blis{});
    CHECK(run(5, op_blocked{.variant = op_eager{}}).at(0)
          == std::format("blocked eager nb={} 5", kc));
}

TEST_CASE("a recursive node halves down to its cutoff", "[control]")
{
    const trace t = run(
        5,
        op_recursive{.cutoff = 2, .op = op_unblocked{.variant = op_lazy{}}});
    CHECK(t
          == trace{"recursive 5",
                   "unblocked lazy 2",
                   "gemm:blis 0",
                   "op:blis 1",
                   "gemm:blis 1",
                   "op:blis 1",
                   "gemm:blis 3",
                   "recursive 3",
                   "unblocked lazy 1",
                   "gemm:blis 0",
                   "op:blis 1",
                   "gemm:blis 2",
                   "unblocked lazy 2",
                   "gemm:blis 0",
                   "op:blis 1",
                   "gemm:blis 1",
                   "op:blis 1"});
}

TEST_CASE("select chooses a subtree from the operands", "[control]")
{
    const famlies::select small_or_blocked{
        .when = [](famlies::len_type n) { return n <= 2; },
        .then = op_unblocked{.variant = op_lazy{}},
        .otherwise = op_blocked{.variant = op_eager{}, .nb = 2}
    };
    STATIC_CHECK(op_control<decltype(small_or_blocked)>);
    CHECK(run(2, small_or_blocked).at(0) == "unblocked lazy 2");
    CHECK(run(3, small_or_blocked).at(0) == "blocked eager nb=2 3");

    // As a child, where it chooses per block, and between leaves.
    const trace t =
        run(3,
            op_blocked{
                .variant = op_eager{},
                .nb = 2,
                .op = small_or_blocked,
                .gemm = famlies::select{.when = [](famlies::len_type n)
                                        { return n == 0; },
                                        .then = famlies::reference{},
                                        .otherwise = famlies::blis{}}
    });
    CHECK(t.at(1) == "unblocked lazy 2");
    CHECK(t.back() == "gemm:reference 0");
}

TEST_CASE("a structural tree is a template argument", "[control]")
{
    constexpr op_blocked tree{.variant = op_lazy{},
                              .nb = 2,
                              .op = op_unblocked{.variant = op_eager{}}};
    trace t;
    op<tree>(t, 3);
    CHECK(t == run(3, tree));

    trace u;
    op<op_unblocked{.variant = op_eager{}}>(u, 2);
    CHECK(u == run(2, op_unblocked{.variant = op_eager{}}));
}

TEST_CASE("a tree with a variant the op doesn't have is rejected anywhere",
          "[control]")
{
    STATIC_CHECK(op_control<op_blocked<op_eager>>);
    STATIC_CHECK(!op_control<op_unblocked<not_a_variant>>);
    STATIC_CHECK(!op_control<op_blocked<not_a_variant>>);
    STATIC_CHECK(
        !op_control<op_blocked<op_eager, op_unblocked<not_a_variant>>>);
    STATIC_CHECK(!op_control<
                 op_recursive<op_blocked<op_lazy, op_blocked<not_a_variant>>>>);

    // A node where a leaf goes, or a leaf where a node goes.
    STATIC_CHECK(!op_control<op_unblocked<op_eager, op_unblocked<op_eager>>>);
    STATIC_CHECK(!op_control<op_blocked<op_eager, famlies::blis>>);
    STATIC_CHECK(!op_control<op_blocked<op_eager,
                                        op_unblocked<op_eager>,
                                        op_unblocked<op_eager>>>);
    STATIC_CHECK(!op_control<int>);

    // Either side of a select.
    const auto when = [](famlies::len_type) { return true; };
    STATIC_CHECK(!op_control<famlies::select<decltype(when),
                                             op_unblocked<op_eager>,
                                             op_unblocked<not_a_variant>>>);

    // At the call site.
    STATIC_CHECK(callable<op_blocked<op_eager, op_unblocked<op_lazy>>>);
    STATIC_CHECK(!callable<op_blocked<op_eager, op_unblocked<not_a_variant>>>);
}
