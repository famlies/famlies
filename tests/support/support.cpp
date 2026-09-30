// The test support catches what it's there to catch.

#include "support/random.hpp"
#include "support/sentinel.hpp"
#include "support/storage.hpp"
#include "support/types.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <cmath>
#include <complex>
#include <limits>

using famlies::len_type;
using namespace famlies::test;

namespace
{

template <class T> bool is_nan(const T& x)
{
    if constexpr (famlies::is_complex_v<T>)
        return std::isnan(x.real()) && std::isnan(x.imag());
    else
        return std::isnan(x);
}

} // namespace

TEMPLATE_LIST_TEST_CASE("the sentinel is a NaN of its own",
                        "[support]",
                        element_types)
{
    using T = TestType;
    const T s = sentinel<T>();
    CHECK(is_nan(s));
    CHECK(is_sentinel(s));
    CHECK(is_nan(s + T(1)));
    CHECK_FALSE(is_sentinel(T(0)));
    CHECK_FALSE(
        is_sentinel(T(std::numeric_limits<famlies::real_t<T>>::quiet_NaN())));
}

TEMPLATE_LIST_TEST_CASE("test matrices have their storage's strides",
                        "[support]",
                        element_types)
{
    using T = TestType;
    test_matrix<T> col(3, 4, storage::column_major);
    CHECK(col.view().stride(0) == 1);
    CHECK(col.view().stride(1) == 4);
    test_matrix<T> row(3, 4, storage::row_major);
    CHECK(row.view().stride(0) == 5);
    CHECK(row.view().stride(1) == 1);
    test_matrix<T> gen(3, 4, storage::general);
    CHECK(gen.view().stride(0) == 2);
    CHECK(gen.view().stride(1) == 7);

    for (const test_matrix<T>* m : {&col, &row, &gen})
    {
        CHECK(m->view().length(0) == 3);
        CHECK(m->view().length(1) == 4);
        CHECK(sentinel_intact(m->view(), everywhere));
        CHECK(m->gaps_intact());
    }

    test_matrix<T> empty(0, 4, storage::general);
    CHECK(empty.view().length(0) == 0);
    CHECK(empty.gaps_intact());
}

TEMPLATE_LIST_TEST_CASE("a write outside a test matrix's elements is caught",
                        "[support]",
                        element_types)
{
    using T = TestType;
    const storage s = GENERATE(from_range(storages));
    CAPTURE(to_string(s));
    auto rng = make_rng();
    test_matrix<T> m(3, 4, s);
    const auto A = m.view();
    fill_random(A, rng);
    REQUIRE(m.gaps_intact());

    // Just past the last element of the first row (row-major) or column (the
    // others), which is a gap, and just before the first element, a guard.
    T* gap = s == storage::row_major ? &A[0][3] + 1 : &A[2][0] + 1;
    T* guard = &A[0][0] - 1;
    for (T* p : {gap, guard})
    {
        *p = T(0);
        CHECK_FALSE(m.gaps_intact());
        *p = sentinel<T>();
        CHECK(m.gaps_intact());
    }
}

TEMPLATE_LIST_TEST_CASE("what must not be touched is checked by position",
                        "[support]",
                        element_types)
{
    using T = TestType;
    auto rng = make_rng();
    test_matrix<T> m(4, 4, storage::general);
    const auto A = m.view();
    fill_random(A, rng);
    fill_sentinel(A, strictly_upper);
    CHECK(sentinel_intact(A, strictly_upper));
    CHECK_FALSE(sentinel_intact(A, strictly_lower));

    A[0][3] = random_value<T>(rng);
    CHECK_FALSE(sentinel_intact(A, strictly_upper));
}
