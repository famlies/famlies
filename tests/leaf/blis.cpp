#include "support/random.hpp"
#include "support/storage.hpp"
#include "support/types.hpp"
#include "support/wide.hpp"

#include <famlies/core/conj.hpp>
#include <famlies/leaf/blis.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <blis.h>

#include <algorithm>
#include <cmath>
#include <complex>

using famlies::blis;
using famlies::len_type;
using famlies::detail::blis_runtime;
using namespace famlies::test;

TEST_CASE("the blis leaf's default block size is BLIS's KC, at most 256",
          "[blis]")
{
    INFO("BLIS " << bli_info_get_version_str());
    const cntx_t* cntx = bli_gks_query_cntx();
    const auto kc = [&](num_t dt) -> len_type
    {
        return std::min<len_type>(bli_cntx_get_blksz_def_dt(dt, BLIS_KC, cntx),
                                  256);
    };
    CHECK(famlies::default_block_size<float>(blis{}) == kc(BLIS_FLOAT));
    CHECK(famlies::default_block_size<double>(blis{}) == kc(BLIS_DOUBLE));
    CHECK(famlies::default_block_size<std::complex<float>>(blis{})
          == kc(BLIS_SCOMPLEX));
    CHECK(famlies::default_block_size<std::complex<double>>(blis{})
          == kc(BLIS_DCOMPLEX));
    CHECK(famlies::default_block_size<double>(blis{}) > 0);
}

TEST_CASE("the blis leaf's thread counts reach BLIS", "[blis]")
{
    SECTION("none: BLIS's global setting")
    { CHECK(blis_runtime("test", blis{}).get() == nullptr); }

    if (!bli_info_get_enable_threading())
        SKIP("BLIS was built without threading");

    SECTION("a total")
    {
        const blis_runtime rt("test", blis{.threads = 3});
        REQUIRE(rt.get() != nullptr);
        CHECK(bli_rntm_num_threads(rt.get()) == 3);
    }

    SECTION("ways per loop")
    {
        const blis_runtime rt("test", blis{.jc = 2, .ir = 3});
        REQUIRE(rt.get() != nullptr);
        CHECK(bli_rntm_jc_ways(rt.get()) == 2);
        CHECK(bli_rntm_pc_ways(rt.get()) == 1);
        CHECK(bli_rntm_ic_ways(rt.get()) == 1);
        CHECK(bli_rntm_jr_ways(rt.get()) == 1);
        CHECK(bli_rntm_ir_ways(rt.get()) == 3);
    }
}

// C := op(A)·B through bli_?gemm_ex, as the leaf's ops will call BLIS:
// dispatch on T, pointers cast to BLIS's types, strides from the views, a
// conjugated A as a conj flag, and the leaf's rntm_t.
TEMPLATE_LIST_TEST_CASE("the blis leaf calls BLIS's typed API",
                        "[blis]",
                        element_types)
{
    using T = TestType;
    using famlies::detail::blis_ptr;
    const storage s = GENERATE(from_range(storages));
    const bool conjugate_a = GENERATE(false, true);
    const int threads = GENERATE(0, 2);
    CAPTURE(to_string(s), conjugate_a, threads);

    auto rng = make_rng();
    test_matrix<T> a(3, 4, s), b(4, 2, s), c(3, 2, s);
    fill_random(a.view(), rng);
    fill_random(b.view(), rng);
    // beta = 0: BLIS reads nothing of C, which holds the sentinel (a NaN).

    const auto run = [&](const auto& A)
    {
        const auto gemm = famlies::detail::blis_pick<T>(bli_sgemm_ex,
                                                        bli_dgemm_ex,
                                                        bli_cgemm_ex,
                                                        bli_zgemm_ex);
        const auto a_base = famlies::detail::strip_conj(A);
        const auto B = b.view();
        const auto C = c.view();
        const T alpha(1), beta(0);
        REQUIRE(famlies::detail::blis_fits(A));
        const blis_runtime rt("gemm", blis{.threads = threads});
        gemm(famlies::detail::blis_trans<decltype(A)>,
             BLIS_NO_TRANSPOSE,
             C.length(0),
             C.length(1),
             a_base.length(1),
             blis_ptr(&alpha),
             blis_ptr(a_base.data()),
             a_base.stride(0),
             a_base.stride(1),
             blis_ptr(B.data()),
             B.stride(0),
             B.stride(1),
             blis_ptr(&beta),
             blis_ptr(C.data()),
             C.stride(0),
             C.stride(1),
             nullptr,
             rt.get());
    };
    if (conjugate_a)
        run(famlies::conjugated(a.view()));
    else
        run(a.view());

    const auto A = a.view();
    const auto B = b.view();
    const auto C = c.view();
    // Within tolerance<T>(4) of sum |a| |b|, with alpha = 1 and beta = 0.
    for (len_type i = 0; i < 3; ++i)
        for (len_type j = 0; j < 2; ++j)
        {
            T expected(0);
            famlies::real_t<T> magnitude(0);
            for (len_type k = 0; k < 4; ++k)
            {
                T aik = A[i][k];
                if constexpr (famlies::is_complex_v<T>)
                    if (conjugate_a)
                        aik = std::conj(aik);
                expected += aik * B[k][j];
                magnitude += std::abs(aik) * std::abs(B[k][j]);
            }
            CHECK(std::abs(C[i][j] - expected)
                  <= tolerance<T>(4) * magnitude);
        }
    CHECK(a.gaps_intact());
    CHECK(b.gaps_intact());
    CHECK(c.gaps_intact());
}
