# famlies_bench times famlies's ops against libflame. It's built only with
# FAMLIES_BUILD_BENCHMARKS, and libflame only for it.
#
# libflame calls BLAS, which it gets from this project's BLIS, so both sides
# run on the same BLIS. That BLIS needs its BLAS layer, which is BLIS's
# default, so the benchmark wants a build tree of its own where
# BLIS_CONFIGURE_OPTIONS leaves it on.

if(BLIS_CONFIGURE_OPTIONS MATCHES "(^|;)--disable-blas(;|$)")
    message(FATAL_ERROR
            "FAMLIES_BUILD_BENCHMARKS needs BLIS's BLAS layer, which libflame "
            "calls, but BLIS_CONFIGURE_OPTIONS has --disable-blas. Configure "
            "the benchmark in a build tree of its own, without it.")
endif()

find_package(libflame REQUIRED)

add_executable(famlies_bench bench.cpp libflame.c libflame.h)
target_link_libraries(famlies_bench PRIVATE famlies libflame::libflame)
