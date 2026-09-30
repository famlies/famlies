// FAMLIES_NO_PRECONDITION_CHECKS, in a TU of its own. It includes nothing
// whose inline functions check preconditions, since those would then differ
// from the other TUs' (an ODR violation).

#ifndef FAMLIES_NO_PRECONDITION_CHECKS
#define FAMLIES_NO_PRECONDITION_CHECKS
#endif
#include <famlies/core/precondition.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("FAMLIES_NO_PRECONDITION_CHECKS compiles checks out",
          "[precondition]")
{
    int evaluated = 0;
    const int n = -1;
    FAMLIES_EXPECTS("test", (++evaluated, n >= 0), n);
    CHECK(evaluated == 0);
}
