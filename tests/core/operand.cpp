#include "support/types.hpp"

#include <famlies/core/operand.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <marray/marray.hpp>

#include <complex>
#include <type_traits>
#include <vector>

using famlies::element_t;
using famlies::operand;
using famlies::rank_v;
using namespace famlies::test;

TEMPLATE_LIST_TEST_CASE("MArray's views and elements are operands",
                        "[operand]",
                        element_types)
{
    using T = TestType;

    STATIC_CHECK(operand<MArray::marray_view<T, 1>>);
    STATIC_CHECK(operand<MArray::marray_view<T, 2>>);
    STATIC_CHECK(operand<MArray::marray_view<T, 3>>);
    STATIC_CHECK(operand<MArray::marray_view<const T, 2>>);
    STATIC_CHECK(operand<const MArray::marray_view<T, 2>&>);
    STATIC_CHECK(!operand<MArray::marray_view<T, MArray::DYNAMIC>>);

    STATIC_CHECK(rank_v<MArray::marray_view<T, 2>> == 2);
    STATIC_CHECK(std::is_same_v<element_t<MArray::marray_view<const T, 2>>, T>);

    // An owning array, or an element, is an operand only as an lvalue.
    STATIC_CHECK(operand<MArray::marray<T, 2>&>);
    STATIC_CHECK(operand<const MArray::marray<T, 2>&>);
    STATIC_CHECK(!operand<MArray::marray<T, 2>>);
    STATIC_CHECK(operand<T&>);
    STATIC_CHECK(operand<const T&>);
    STATIC_CHECK(!operand<T>);
    STATIC_CHECK(rank_v<T&> == 0);

    STATIC_CHECK(!operand<int&>);
    STATIC_CHECK(!operand<std::vector<T>&>);
}

TEST_CASE("the operand concept takes any field type", "[operand]")
{
    STATIC_CHECK(operand<long double&>);
    STATIC_CHECK(operand<std::complex<long double>&>);
    STATIC_CHECK(operand<MArray::marray_view<long double, 2>>);
}
