#ifndef FAMLIES_TESTS_SUPPORT_TYPES_HPP
#define FAMLIES_TESTS_SUPPORT_TYPES_HPP

#include <complex>
#include <tuple>

namespace famlies::test
{

/// The element types every test case runs over, for TEMPLATE_LIST_TEST_CASE.
using element_types =
    std::tuple<float, double, std::complex<float>, std::complex<double>>;

} // namespace famlies::test

#endif // FAMLIES_TESTS_SUPPORT_TYPES_HPP
