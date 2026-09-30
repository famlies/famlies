#include "support/random.hpp"
#include "support/storage.hpp"
#include "support/types.hpp"

#include <famlies/core/conj.hpp>
#include <famlies/core/operand.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <marray/marray.hpp>

#include <complex>
#include <type_traits>
#include <utility>

using famlies::adjoint;
using famlies::conjugated;
using famlies::conjugated_view;
using famlies::is_complex_v;
using famlies::len_type;
using famlies::operand;
using famlies::rank_v;
using famlies::detail::strip_conj;
using famlies::detail::value;
using namespace famlies::test;

namespace
{

template <class T> T conj_of(const T& x)
{
    if constexpr (is_complex_v<T>)
        return std::conj(x);
    else
        return x;
}

// Whether a and b view the same elements the same way.
template <class A, class B> bool same_view(const A& a, const B& b)
{
    static_assert(rank_v<A> == rank_v<B>);
    if (a.data() != b.data())
        return false;
    for (int d = 0; d < rank_v<A>; ++d)
        if (a.length(d) != b.length(d) || a.stride(d) != b.stride(d))
            return false;
    return true;
}

} // namespace

TEMPLATE_LIST_TEST_CASE("conjugated reads the conjugate of each element",
                        "[conj]",
                        element_types)
{
    using T = TestType;
    const storage s = GENERATE(from_range(storages));
    CAPTURE(to_string(s));
    auto rng = make_rng();
    test_matrix<T> m(4, 3, s);
    fill_random(m.view(), rng);
    const MArray::marray_view<T, 2> A = m.view();

    const auto C = conjugated(A);
    CHECK(C.length(0) == 4);
    CHECK(C.length(1) == 3);
    for (len_type i = 0; i < 4; ++i)
        for (len_type j = 0; j < 3; ++j)
            CHECK(value(C[i][j]) == conj_of(A[i][j]));

    // Indexing a conjugated operand gives conjugated operands: blocks,
    // vectors, elements and partially indexed views.
    const auto R = MArray::range(1, 3);
    const auto block = C[R][MArray::range(0, 2)];
    STATIC_CHECK(rank_v<decltype(block)> == 2);
    for (len_type i = 0; i < 2; ++i)
        for (len_type j = 0; j < 2; ++j)
            CHECK(value(block[i][j]) == conj_of(A[1 + i][j]));

    const auto column = C[R][2];
    STATIC_CHECK(rank_v<decltype(column)> == 1);
    for (len_type i = 0; i < 2; ++i)
        CHECK(value(column[i]) == conj_of(A[1 + i][2]));

    const auto row = C[3][MArray::range(1, 3)];
    STATIC_CHECK(rank_v<decltype(row)> == 1);
    for (len_type j = 0; j < 2; ++j)
        CHECK(value(row[j]) == conj_of(A[3][1 + j]));

    // A rank-1 range is followed by .view(), as for MArray's own views.
    const auto part = column[MArray::range(1, 2)].view();
    CHECK(same_view(strip_conj(part), A[R][2][MArray::range(1, 2)].view()));

    const auto rows = C[R].view();
    CHECK(same_view(strip_conj(rows), A[R].view()));

    CHECK(m.gaps_intact());
}

TEMPLATE_LIST_TEST_CASE(
    "conjugated wraps complex operands and is the identity on real ones",
    "[conj]",
    element_types)
{
    using T = TestType;
    using V = MArray::marray_view<T, 2>;
    test_matrix<T> m(3, 3, storage::column_major);
    const V A = m.view();

    if constexpr (is_complex_v<T>)
    {
        STATIC_CHECK(
            std::is_same_v<decltype(conjugated(A)), conjugated_view<V>>);
        STATIC_CHECK(
            std::is_same_v<decltype(conjugated(A)[1][1]), conjugated_view<T&>>);
        STATIC_CHECK(
            std::is_same_v<decltype(conjugated(A[1][1])), conjugated_view<T&>>);
    }
    else
    {
        STATIC_CHECK(std::is_same_v<decltype(conjugated(A)), V>);
        STATIC_CHECK(std::is_same_v<decltype(conjugated(A[1][1])), T&>);
    }
    CHECK(same_view(strip_conj(conjugated(A)), A));
    CHECK(&strip_conj(conjugated(A[1][2])) == &A[1][2]);
}

TEMPLATE_LIST_TEST_CASE("conjugating twice gives the operand back",
                        "[conj]",
                        element_types)
{
    using T = TestType;
    using V = MArray::marray_view<T, 2>;
    const storage s = GENERATE(from_range(storages));
    CAPTURE(to_string(s));
    test_matrix<T> m(4, 3, s);
    const V A = m.view();
    const auto R = MArray::range(1, 3);

    STATIC_CHECK(std::is_same_v<decltype(conjugated(conjugated(A))), V>);
    CHECK(same_view(conjugated(conjugated(A)), A));

    // Through indexing: a block, a vector and an element of the conjugate.
    CHECK(same_view(conjugated(conjugated(A)[R][R]), A[R][R]));
    CHECK(same_view(conjugated(conjugated(A)[R][0]), A[R][0]));
    STATIC_CHECK(std::is_same_v<decltype(conjugated(conjugated(A)[1][2])), T&>);
    CHECK(&conjugated(conjugated(A)[1][2]) == &A[1][2]);
    CHECK(&conjugated(conjugated(A[1][2])) == &A[1][2]);
}

TEMPLATE_LIST_TEST_CASE(".T() goes through a conjugated operand",
                        "[conj]",
                        element_types)
{
    using T = TestType;
    using V = MArray::marray_view<T, 2>;
    const storage s = GENERATE(from_range(storages));
    CAPTURE(to_string(s));
    auto rng = make_rng();
    test_matrix<T> m(4, 3, s);
    fill_random(m.view(), rng);
    const V A = m.view();

    const auto CT = conjugated(A).T();
    STATIC_CHECK(std::is_same_v<std::remove_const_t<decltype(CT)>,
                                decltype(conjugated(A.T()))>);
    CHECK(same_view(strip_conj(CT), A.T()));
    for (len_type i = 0; i < 3; ++i)
        for (len_type j = 0; j < 4; ++j)
            CHECK(value(CT[i][j]) == conj_of(A[j][i]));

    // adjoint(A) is conjugated(A.T()), and undoes itself.
    const auto H = adjoint(A);
    STATIC_CHECK(std::is_same_v<std::remove_const_t<decltype(H)>,
                                decltype(conjugated(A.T()))>);
    CHECK(same_view(strip_conj(H), A.T()));
    STATIC_CHECK(std::is_same_v<decltype(adjoint(H)), V>);
    CHECK(same_view(adjoint(H), A));
    CHECK(same_view(strip_conj(H.T()), A));

    // An element's adjoint is its conjugate.
    CHECK(value(adjoint(A[1][2])) == conj_of(A[1][2]));
}

TEMPLATE_LIST_TEST_CASE("a conjugated operand is a read-only operand",
                        "[conj]",
                        element_types)
{
    using T = TestType;
    using V = MArray::marray_view<T, 2>;
    test_matrix<T> m(3, 3, storage::column_major);
    const V A = m.view();
    const auto R = MArray::range(0, 2);

    STATIC_CHECK(operand<decltype(conjugated(A))>);
    STATIC_CHECK(operand<decltype(conjugated(A)[R][R])>);
    STATIC_CHECK(operand<decltype(conjugated(A)[R][1])>);
    STATIC_CHECK(operand<decltype(conjugated(A)[1][1])>);
    if constexpr (is_complex_v<T>)
    {
        STATIC_CHECK(!requires { conjugated(A)[1][1] = T(1); });
    }

    // An owning array is conjugated as its view, and only as an lvalue.
    MArray::marray<T, 2> owned({2, 2}, T(1));
    STATIC_CHECK(std::is_same_v<decltype(conjugated(owned)),
                                decltype(conjugated(std::declval<V>()))>);
    CHECK(same_view(strip_conj(conjugated(owned)), owned.view()));
    STATIC_CHECK(!requires { conjugated(MArray::marray<T, 2>({2, 2})); });
}
