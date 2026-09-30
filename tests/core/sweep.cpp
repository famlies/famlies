#include <famlies/core/sweep.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <marray/marray.hpp>

#include <complex>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <vector>

using famlies::backward;
using famlies::forward;
using famlies::len_type;
using famlies::one;
using famlies::sweep;

namespace
{

// A step's R0, R1 and R2 as [from, to) pairs.
struct step
{
    len_type r0_from, r0_to, r1_from, r1_to, r2_from, r2_to;

    bool operator==(const step&) const = default;
};

template <class Sweep> std::vector<step> steps(const Sweep& s)
{
    std::vector<step> out;
    for (auto [R0, R1, R2] : s)
    {
        if constexpr (std::is_same_v<decltype(R1), len_type>)
            out.push_back({R0.from(), R0.to(), R1, R1 + 1, R2.from(), R2.to()});
        else
            out.push_back(
                {R0.from(), R0.to(), R1.from(), R1.to(), R2.from(), R2.to()});
    }
    return out;
}

} // namespace

TEST_CASE("sweep forward, block size dividing n", "[sweep]")
{
    CHECK(steps(sweep(6, 3, forward))
          == std::vector<step>{
              {0, 0, 0, 3, 3, 6},
              {0, 3, 3, 6, 6, 6}
    });
}

TEST_CASE("sweep forward, block size not dividing n", "[sweep]")
{
    // The short block is the last one, at the end.
    CHECK(steps(sweep(7, 3, forward))
          == std::vector<step>{
              {0, 0, 0, 3, 3, 7},
              {0, 3, 3, 6, 6, 7},
              {0, 6, 6, 7, 7, 7}
    });
}

TEST_CASE("sweep backward, block size dividing n", "[sweep]")
{
    CHECK(steps(sweep(6, 3, backward))
          == std::vector<step>{
              {0, 3, 3, 6, 6, 6},
              {0, 0, 0, 3, 3, 6}
    });
}

TEST_CASE("sweep backward, block size not dividing n", "[sweep]")
{
    // The short block is the last one, at the start.
    CHECK(steps(sweep(7, 3, backward))
          == std::vector<step>{
              {0, 4, 4, 7, 7, 7},
              {0, 1, 1, 4, 4, 7},
              {0, 0, 0, 1, 1, 7}
    });
}

TEST_CASE("sweep with a block size larger than n", "[sweep]")
{
    CHECK(steps(sweep(2, 5, forward))
          == std::vector<step>{
              {0, 0, 0, 2, 2, 2}
    });
    CHECK(steps(sweep(2, 5, backward))
          == std::vector<step>{
              {0, 0, 0, 2, 2, 2}
    });
}

TEST_CASE("sweep with block size one", "[sweep]")
{
    CHECK(steps(sweep(3, one, forward))
          == std::vector<step>{
              {0, 0, 0, 1, 1, 3},
              {0, 1, 1, 2, 2, 3},
              {0, 2, 2, 3, 3, 3}
    });
    CHECK(steps(sweep(3, one, backward))
          == std::vector<step>{
              {0, 2, 2, 3, 3, 3},
              {0, 1, 1, 2, 2, 3},
              {0, 0, 0, 1, 1, 3}
    });

    // A runtime 1 gives ranges, and the same steps.
    CHECK(steps(sweep(3, 1, forward)) == steps(sweep(3, one, forward)));
    CHECK(steps(sweep(3, 1, backward)) == steps(sweep(3, one, backward)));
}

TEST_CASE("sweep over n = 0", "[sweep]")
{
    CHECK(steps(sweep(0, 4, forward)).empty());
    CHECK(steps(sweep(0, 4, backward)).empty());
    CHECK(steps(sweep(0, one, forward)).empty());
    CHECK(steps(sweep(0, one, backward)).empty());
}

TEST_CASE("sweep partitions [0, n) in order", "[sweep]")
{
    const len_type n = GENERATE(range(0, 20));
    const len_type bs = GENERATE(range(1, 9));
    CAPTURE(n, bs);

    const auto check = [&](const std::vector<step>& s, bool is_forward)
    {
        // Every step splits [0, n) as R0, R1, R2, and R1 moves by bs, but
        // for the last, which has what's left.
        REQUIRE(static_cast<len_type>(s.size()) == (n + bs - 1) / bs);
        len_type k = is_forward ? 0 : n;
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            const bool last = i + 1 == s.size();
            const len_type b = last ? n - (len_type(s.size()) - 1) * bs : bs;
            CHECK(s[i].r0_from == 0);
            CHECK(s[i].r0_to == s[i].r1_from);
            CHECK(s[i].r1_to - s[i].r1_from == b);
            CHECK(s[i].r1_to == s[i].r2_from);
            CHECK(s[i].r2_to == n);
            CHECK((is_forward ? s[i].r1_from : s[i].r1_to) == k);
            k = is_forward ? s[i].r1_to : s[i].r1_from;
        }
        CHECK(k == (is_forward ? n : 0));
    };
    check(steps(sweep(n, bs, forward)), true);
    check(steps(sweep(n, bs, backward)), false);
}

TEST_CASE("sweep's ranges index MArray operands", "[sweep]")
{
    using T = std::complex<double>;
    MArray::marray<T, 2> a({5, 5}, T(0));
    MArray::marray_view<T, 2> A = a;

    for (auto [R0, R1, R2] : sweep(A.length(0), one, forward))
    {
        static_assert(std::is_same_v<decltype(A[R1][R1]), T&>);
        static_assert(
            std::is_same_v<decltype(A[R2][R1]), MArray::marray_view<T, 1>>);
        static_assert(
            std::is_same_v<decltype(A[R1][R0]), MArray::marray_view<T, 1>>);
        static_assert(
            std::is_same_v<decltype(A[R2][R2]), MArray::marray_view<T, 2>>);
        A[R1][R1] += 1.0;
        CHECK(A[R2][R1].length(0) == A.length(0) - R1 - 1);
    }
    for (auto [R0, R1, R2] : sweep(A.length(0), 2, backward))
    {
        static_assert(
            std::is_same_v<decltype(A[R1][R1]), MArray::marray_view<T, 2>>);
        CHECK(A[R2][R1].length(0) == R2.size());
        CHECK(A[R2][R1].length(1) == R1.size());
    }
    for (len_type i = 0; i < 5; ++i) CHECK(a[i][i] == T(1));
}

TEST_CASE("sweep is a forward range", "[sweep]")
{
    STATIC_CHECK(std::ranges::forward_range<decltype(sweep(4, one, forward))>);
    STATIC_CHECK(std::ranges::forward_range<decltype(sweep(4, 2, backward))>);

    // Iterators compare by position.
    const auto s = sweep(4, 2, forward);
    auto it = s.begin();
    CHECK(it++ == s.begin());
    CHECK(it != s.begin());
    CHECK(++it == s.end());
}
