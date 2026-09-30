// famlies_bench: times famlies's ops, on trees of interest and on the
// default tree, against libflame with its default control trees. Both sides
// use one BLIS and its global thread count. usage() lists the options.
//
// The trees are the default one, each blocked variant at each --nb (0 is
// the leaf's default), and recursion down to each --cutoff, all over the
// default tree's diagonal-block subtree.
//
// --gate runs gate() for d and z and exits 1 if either fails.

#include "libflame.h"

#include <famlies/famlies.hpp>

#include <blis.h>
#include <marray/marray.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using famlies::len_type;

struct options
{
    std::string op = "chol";
    std::string types = "d,z";
    std::vector<len_type> sizes = {500, 1000, 2000, 4000};
    std::vector<int> threads = {1};
    int runs = 3;
    std::string trees = "all";
    std::vector<len_type> nbs = {0};
    std::vector<len_type> cutoffs = {128};
    bool upper = false;
    bool gate = false;
};

[[noreturn]] void usage(const char* why)
{
    std::fprintf(stderr, "famlies_bench: %s\n", why);
    std::fputs(
        "usage: famlies_bench [--op chol] [--types s,d,c,z]\n"     //
        "           [--n 1000,2000] [--threads 1,14] [--runs 3]\n" //
        "           [--trees all|default,bordered,left-looking,right-looking,\n" //
        "                    recursive]\n"                   //
        "           [--nb 0,256] [--cutoff 128] [--upper]\n" //
        "       famlies_bench --gate [--runs 3]\n",
        stderr);
    std::exit(2);
}

template <class T> std::vector<T> parse_list(std::string_view s)
{
    std::vector<T> out;
    while (!s.empty())
    {
        const auto comma = s.find(',');
        out.push_back(T(std::stol(std::string(s.substr(0, comma)))));
        s.remove_prefix(comma == s.npos ? s.size() : comma + 1);
    }
    return out;
}

options parse(int argc, char** argv)
{
    options o;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        const auto value = [&]
        {
            if (i + 1 == argc)
                usage("a value is missing");
            return std::string_view(argv[++i]);
        };
        if (arg == "--op")
            o.op = value();
        else if (arg == "--types")
            o.types = value();
        else if (arg == "--n")
            o.sizes = parse_list<len_type>(value());
        else if (arg == "--threads")
            o.threads = parse_list<int>(value());
        else if (arg == "--runs")
            o.runs = int(std::stol(std::string(value())));
        else if (arg == "--trees")
            o.trees = value();
        else if (arg == "--nb")
            o.nbs = parse_list<len_type>(value());
        else if (arg == "--cutoff")
            o.cutoffs = parse_list<len_type>(value());
        else if (arg == "--upper")
            o.upper = true;
        else if (arg == "--gate")
            o.gate = true;
        else
            usage("unknown option");
    }
    if (o.op != "chol")
        usage("the only op so far is chol");
    if (o.runs < 1)
        usage("--runs must be at least 1");
    return o;
}

int all_cores()
{ return int(std::max(1u, std::thread::hardware_concurrency())); }

template <class T> constexpr int libflame_type = 0;
template <> constexpr int libflame_type<double> = 1;
template <> constexpr int libflame_type<std::complex<float>> = 2;
template <> constexpr int libflame_type<std::complex<double>> = 3;

template <class T> constexpr char type_name = 's';
template <> constexpr char type_name<double> = 'd';
template <> constexpr char type_name<std::complex<float>> = 'c';
template <> constexpr char type_name<std::complex<double>> = 'z';

// Cholesky's flop count, LAPACK's: real flops, a complex multiply being 6
// and a complex add 2.
template <class T> double chol_flops(double n)
{
    const double muls = n * n * n / 6 + n * n / 2 + n / 3;
    const double adds = n * n * n / 6 - n / 6;
    return famlies::is_complex_v<T> ? 6 * muls + 2 * adds : muls + adds;
}

// An n x n Hermitian positive definite matrix, column-major: random in
// [-1, 1] off the diagonal and n on it, so diagonally dominant.
template <class T> MArray::marray<T, 2> hpd(len_type n)
{
    using R = famlies::real_t<T>;
    std::mt19937_64 rng(n);
    std::uniform_real_distribution<R> part(-1, 1);
    MArray::marray<T, 2> a({n, n}, T(0), MArray::COLUMN_MAJOR);
    for (len_type j = 0; j < n; ++j)
    {
        a[j][j] = T(R(n));
        for (len_type i = j + 1; i < n; ++i)
        {
            T x;
            if constexpr (famlies::is_complex_v<T>)
            {
                const R re = part(rng);
                x = T(re, part(rng));
            }
            else
                x = part(rng);
            a[i][j] = x;
            a[j][i] = famlies::detail::conjugate(x);
        }
    }
    return a;
}

// A factorization in place, which says whether it succeeded.
template <class T> using factor = std::function<bool(MArray::marray<T, 2>&)>;

// The best time of each of fs, each run on a fresh copy of a made outside
// the timing, at least `runs` times and until its runs add up to a second.
// The fs take turns, so a slow stretch of the machine, common at small n on
// all cores (which may be of two kinds), slows them alike.
template <class T>
std::vector<double> best_seconds(const MArray::marray<T, 2>& a,
                                 int runs,
                                 const std::vector<factor<T>>& fs)
{
    const len_type n = a.length(0);
    MArray::marray<T, 2> work({n, n}, T(0), MArray::COLUMN_MAJOR);
    std::vector<double> best(fs.size(),
                             std::numeric_limits<double>::infinity());
    std::vector<double> total(fs.size(), 0);
    for (int r = 0; r < runs || std::ranges::min(total) < 1; ++r)
        for (std::size_t i = 0; i < fs.size(); ++i)
        {
            std::copy_n(a.data(), n * n, work.data());
            const auto start = std::chrono::steady_clock::now();
            const bool ok = fs[i](work);
            const std::chrono::duration<double> t =
                std::chrono::steady_clock::now() - start;
            if (!ok)
            {
                std::fputs("famlies_bench: a factorization failed\n", stderr);
                std::exit(1);
            }
            best[i] = std::min(best[i], t.count());
            total[i] += t.count();
        }
    return best;
}

template <class T> factor<T> with_libflame(bool upper)
{
    return [upper](MArray::marray<T, 2>& w)
    {
        return bench_libflame_chol(libflame_type<T>,
                                   !upper,
                                   w.length(0),
                                   w.data(),
                                   w.stride(0),
                                   w.stride(1))
            != 0;
    };
}

template <class T, class Tree> factor<T> with_tree(Tree tree, bool upper)
{
    return [tree, upper](MArray::marray<T, 2>& w)
    {
        const auto r = upper ? famlies::chol(w, famlies::upper_triangle, tree)
                             : famlies::chol(w, famlies::lower_triangle, tree);
        return r.has_value();
    };
}

// The trees of interest, as options name them, and libflame last.
template <class T>
std::vector<std::pair<std::string, factor<T>>> implementations(const options& o)
{
    // The default tree's diagonal-block subtree. Local: Clang can't deduce a
    // node's type inside a template from a designated initializer naming a
    // namespace-scope variable.
    constexpr auto inner = famlies::chol_default.chol;
    const auto wanted = [&](std::string_view name)
    {
        return o.trees
            == "all"
            || ("," + o.trees + ",").find("," + std::string(name) + ",")
            != std::string::npos;
    };
    std::vector<std::pair<std::string, factor<T>>> out;
    const auto add = [&](std::string name, auto tree)
    { out.push_back({std::move(name), with_tree<T>(tree, o.upper)}); };
    if (wanted("default"))
        add("default", famlies::chol_default);
    for (const len_type nb : o.nbs)
    {
        const std::string at = "/" + std::to_string(nb);
        if (wanted("bordered"))
            add("bordered" + at,
                famlies::chol_blocked{.variant = famlies::chol_bordered{},
                                      .nb = nb,
                                      .chol = inner});
        if (wanted("left-looking"))
            add("left-looking" + at,
                famlies::chol_blocked{.variant = famlies::chol_left_looking{},
                                      .nb = nb,
                                      .chol = inner});
        if (wanted("right-looking"))
            add("right-looking" + at,
                famlies::chol_blocked{.variant = famlies::chol_right_looking{},
                                      .nb = nb,
                                      .chol = inner});
    }
    for (const len_type cutoff : o.cutoffs)
        if (wanted("recursive"))
            add("rec/" + std::to_string(cutoff),
                famlies::chol_recursive{.cutoff = cutoff, .chol = inner});
    out.push_back({"libflame", with_libflame<T>(o.upper)});
    return out;
}

void print_header()
{
    std::printf("%-5s %-5s %8s %6s %-18s %12s %10s\n",
                "op",
                "type",
                "threads",
                "n",
                "impl",
                "seconds",
                "GFLOPS");
}

void print_row(char type,
               int threads,
               len_type n,
               const std::string& impl,
               double seconds,
               double flops)
{
    std::printf("%-5s %-5c %8d %6ld %-18s %12.6f %10.2f\n",
                "chol",
                type,
                threads,
                long(n),
                impl.c_str(),
                seconds,
                flops / seconds / 1e9);
    std::fflush(stdout);
}

template <class T> void run(const options& o)
{
    const auto impls = implementations<T>(o);
    std::vector<factor<T>> fs;
    for (const auto& impl : impls) fs.push_back(impl.second);
    for (const int t : o.threads)
    {
        bli_thread_set_num_threads(t);
        for (const len_type n : o.sizes)
        {
            const auto a = hpd<T>(n);
            const auto seconds = best_seconds(a, o.runs, fs);
            for (std::size_t i = 0; i < impls.size(); ++i)
                print_row(type_name<T>,
                          t,
                          n,
                          impls[i].first,
                          seconds[i],
                          chol_flops<T>(double(n)));
        }
    }
}

// The default tree against libflame at n = 1000, 2000, 4000 and 8000, on one
// thread and on all cores: whether every ratio of GFLOPS is at least 95%.
template <class T> bool gate(int runs)
{
    const std::vector<factor<T>> fs = {
        with_tree<T>(famlies::chol_default, false),
        with_libflame<T>(false)};
    bool pass = true;
    for (const int t : {1, all_cores()})
    {
        bli_thread_set_num_threads(t);
        for (const len_type n : {1000, 2000, 4000, 8000})
        {
            const auto a = hpd<T>(n);
            const double flops = chol_flops<T>(double(n));
            const auto seconds = best_seconds(a, runs, fs);
            print_row(type_name<T>, t, n, "default", seconds[0], flops);
            print_row(type_name<T>, t, n, "libflame", seconds[1], flops);
            const double ratio = seconds[1] / seconds[0];
            const bool ok = ratio >= 0.95;
            std::printf("gate  %-5c %8d %6ld ratio %.3f %s\n",
                        type_name<T>,
                        t,
                        long(n),
                        ratio,
                        ok ? "pass" : "FAIL");
            std::fflush(stdout);
            pass = pass && ok;
        }
    }
    return pass;
}

} // namespace

int main(int argc, char** argv)
{
    const options o = parse(argc, argv);
    bench_libflame_init();
    std::printf("BLIS %s, %s, KC d=%ld z=%ld, %d cores\n",
                bli_info_get_version_str(),
                bli_arch_string(bli_arch_query_id()),
                long(famlies::default_block_size<double>(famlies::blis{})),
                long(famlies::default_block_size<std::complex<double>>(
                    famlies::blis{})),
                all_cores());
    print_header();
    int status = 0;
    if (o.gate)
    {
        const bool d = gate<double>(o.runs);
        const bool z = gate<std::complex<double>>(o.runs);
        status = d && z ? 0 : 1;
    }
    else
        for (const char type : o.types) switch (type)
            {
                case 's': run<float>(o); break;
                case 'd': run<double>(o); break;
                case 'c': run<std::complex<float>>(o); break;
                case 'z': run<std::complex<double>>(o); break;
                case ',': break;
                default:  usage("types are s, d, c and z");
            }
    bench_libflame_finalize();
    return status;
}
