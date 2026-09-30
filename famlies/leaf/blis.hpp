#ifndef FAMLIES_LEAF_BLIS_HPP
#define FAMLIES_LEAF_BLIS_HPP

#include <famlies/core/conj.hpp>
#include <famlies/core/control.hpp>
#include <famlies/core/error.hpp>
#include <famlies/core/operand.hpp>
#include <famlies/core/precondition.hpp>
#include <famlies/core/tags.hpp>

#include <blis.h>
#include <marray/marray.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

namespace famlies
{

/// The leaf that runs BLAS-level ops through BLIS's typed API.
///
/// Thread counts: `threads` is how many BLIS may use in all, or else the ways
/// split the loops of its level-3 algorithms (jc, ic, jr, ir, as
/// `bli_rntm_set_ways` names them; BLIS doesn't split pc). Set one or the
/// other. With neither, BLIS's global setting holds.
struct blis
{
    int threads = 0;
    int jc = 0;
    int ic = 0;
    int jr = 0;
    int ir = 0;
};

template <> inline constexpr bool enable_leaf<blis> = true;

/// The element types BLIS computes in.
template <class T>
concept blis_element = std::same_as<T, float>
                    || std::same_as<T, double>
                    || std::same_as<T, std::complex<float>>
                    || std::same_as<T, std::complex<double>>;

namespace detail
{

template <class V> inline constexpr bool is_blis_view = false;

template <class T, int N>
inline constexpr bool is_blis_view<MArray::marray_view<T, N>> =
    blis_element<std::remove_const_t<T>> && (N == 1 || N == 2);

// The unconjugated view, or element, under A.
template <class A>
using blis_view_t =
    std::remove_cvref_t<decltype(strip_conj(std::declval<A>()))>;

/// What the blis leaf takes: a host marray_view of rank 1 or 2 (an owning
/// marray as its view), an element, or the conjugate of either, of an
/// element type BLIS computes in.
template <class A>
concept blis_operand = operand<A>
                    && blis_element<element_t<A>>
                    && ((rank_v<A> == 0) || is_blis_view<blis_view_t<A>>);

} // namespace detail

template <class A>
inline constexpr bool leaf_takes<blis, A> = detail::blis_operand<A>;

namespace detail
{

template <blis_element T>
inline constexpr num_t blis_datatype =
    std::same_as<T, float>                 ? BLIS_FLOAT
    : std::same_as<T, double>              ? BLIS_DOUBLE
    : std::same_as<T, std::complex<float>> ? BLIS_SCOMPLEX
                                           : BLIS_DCOMPLEX;

// BLIS_ENABLE_STD_COMPLEX would make these the same types, but it
// #defines __typeof__(x) as auto in every TU, so BLIS's own complex types
// are cast to instead.
static_assert(sizeof(std::complex<float>) == sizeof(scomplex));
static_assert(alignof(std::complex<float>) == alignof(scomplex));
static_assert(sizeof(std::complex<double>) == sizeof(dcomplex));
static_assert(alignof(std::complex<double>) == alignof(dcomplex));

template <class T> struct blis_native
{
    using type = T;
};

template <> struct blis_native<std::complex<float>>
{
    using type = scomplex;
};

template <> struct blis_native<std::complex<double>>
{
    using type = dcomplex;
};

template <class T> struct blis_native<const T>
{
    using type = const typename blis_native<T>::type;
};

/// p as the pointer type BLIS's functions for T take.
template <class T> requires blis_element<std::remove_const_t<T>>
auto blis_ptr(T* p)
{ return reinterpret_cast<typename blis_native<T>::type*>(p); }

/// Of `bli_s<op>`, `bli_d<op>`, `bli_c<op>` and `bli_z<op>`, the one for T.
template <blis_element T, class S, class D, class C, class Z>
auto blis_pick(S s, D d, C c, Z z)
{
    if constexpr (std::same_as<T, float>)
        return s;
    else if constexpr (std::same_as<T, double>)
        return d;
    else if constexpr (std::same_as<T, std::complex<float>>)
        return c;
    else
        return z;
}

/// How BLIS reads A: conjugated or not.
template <class A>
inline constexpr conj_t blis_conj =
    is_conjugated_v<A> ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE;

template <class A>
inline constexpr trans_t blis_trans =
    is_conjugated_v<A> ? BLIS_CONJ_NO_TRANSPOSE : BLIS_NO_TRANSPOSE;

/// How BLIS reads A transposed: conjugated or not.
template <class A>
inline constexpr trans_t blis_transposed =
    is_conjugated_v<A> ? BLIS_CONJ_TRANSPOSE : BLIS_TRANSPOSE;

template <side S>
inline constexpr side_t blis_side =
    std::same_as<S, left_side_t> ? BLIS_LEFT : BLIS_RIGHT;

template <triangle Tri>
inline constexpr uplo_t blis_uplo =
    std::same_as<Tri, lower_triangle_t> ? BLIS_LOWER : BLIS_UPPER;

template <diagonal D>
inline constexpr diag_t blis_diag =
    std::same_as<D, implicit_unit_diagonal_t> ? BLIS_UNIT_DIAG
                                              : BLIS_NONUNIT_DIAG;

/// Whether A's lengths fit dim_t and its strides inc_t.
template <operand A> bool blis_fits(A&& a)
{
    if constexpr (rank_v<A> == 0)
        return true;
    else
    {
        const auto& v = strip_conj(a);
        for (int d = 0; d < rank_v<A>; ++d)
            if (!std::in_range<dim_t>(v.length(d))
                || !std::in_range<inc_t>(v.stride(d)))
                return false;
        return true;
    }
}

/// The rntm_t a call passes: null, for BLIS's global setting, unless the
/// leaf sets thread counts. It starts from the global setting, since one
/// from BLIS_RNTM_INITIALIZER asks for a single thread, whatever is set.
class blis_runtime
{
  public:
    blis_runtime(const char* op, const blis& leaf)
    {
        const int ways[] = {leaf.jc, leaf.ic, leaf.jr, leaf.ir};
        FAMLIES_EXPECTS(op,
                        leaf.threads >= 0 && std::ranges::min(ways) >= 0,
                        leaf.threads,
                        leaf.jc,
                        leaf.ic,
                        leaf.jr,
                        leaf.ir);
        const bool by_ways = std::ranges::max(ways) > 0;
        FAMLIES_EXPECTS(op,
                        leaf.threads == 0 || !by_ways,
                        leaf.threads,
                        leaf.jc,
                        leaf.ic,
                        leaf.jr,
                        leaf.ir);
        if (by_ways)
        {
            bli_rntm_init_from_global(&rntm_);
            bli_rntm_set_ways(std::max(leaf.jc, 1),
                              1,
                              std::max(leaf.ic, 1),
                              std::max(leaf.jr, 1),
                              std::max(leaf.ir, 1),
                              &rntm_);
            set_ = true;
        }
        else if (leaf.threads > 0)
        {
            bli_rntm_init_from_global(&rntm_);
            bli_rntm_set_num_threads(leaf.threads, &rntm_);
            set_ = true;
        }
    }

    const rntm_t* get() const { return set_ ? &rntm_ : nullptr; }

  private:
    rntm_t rntm_ = {};
    bool set_ = false;
};

} // namespace detail

// The blis leaf's ops. Each form is one BLIS call, or two where BLIS's op
// lacks alpha or beta.

/// B := alpha·B·op(A)⁻¹ or alpha·op(A)⁻¹·B: trsm, trsv on a vector B, and
/// invscalv on a vector B with an element A.
template <side S, triangle Tri, diagonal D, class A, class B>
void leaf_trsm(const blis& leaf, S, Tri, D, element_t<B> alpha, A&& a, B&& b)
{
    using namespace detail;
    using T = element_t<B>;
    FAMLIES_EXPECTS("trsm", blis_fits(a) && blis_fits(b), a, b);
    const blis_runtime rt("trsm", leaf);
    auto&& av = strip_conj(a);
    auto bv = as_view(std::forward<B>(b));

    if constexpr (rank_v<A> == 0)
    {
        // b := alpha·b / op(a), or alpha·b with a unit diagonal.
        if (alpha != T(1))
            blis_pick<T>(bli_sscalv_ex,
                         bli_dscalv_ex,
                         bli_cscalv_ex,
                         bli_zscalv_ex)(BLIS_NO_CONJUGATE,
                                        bv.length(0),
                                        blis_ptr(&alpha),
                                        blis_ptr(bv.data()),
                                        bv.stride(0),
                                        nullptr,
                                        rt.get());
        if constexpr (std::same_as<D, explicit_diagonal_t>)
            blis_pick<T>(bli_sinvscalv_ex,
                         bli_dinvscalv_ex,
                         bli_cinvscalv_ex,
                         bli_zinvscalv_ex)(blis_conj<A>,
                                           bv.length(0),
                                           blis_ptr(&av),
                                           blis_ptr(bv.data()),
                                           bv.stride(0),
                                           nullptr,
                                           rt.get());
    }
    else if constexpr (rank_v<B> == 1)
    {
        // On the right, b := alpha·b·op(A)⁻¹ is b := alpha·op(A)⁻ᵀ·b.
        constexpr trans_t trans =
            std::same_as<S, left_side_t> ? blis_trans<A> : blis_transposed<A>;
        blis_pick<T>(bli_strsv_ex, bli_dtrsv_ex, bli_ctrsv_ex, bli_ztrsv_ex)(
            blis_uplo<Tri>,
            trans,
            blis_diag<D>,
            bv.length(0),
            blis_ptr(&alpha),
            blis_ptr(av.data()),
            av.stride(0),
            av.stride(1),
            blis_ptr(bv.data()),
            bv.stride(0),
            nullptr,
            rt.get());
    }
    else
        blis_pick<T>(bli_strsm_ex, bli_dtrsm_ex, bli_ctrsm_ex, bli_ztrsm_ex)(
            blis_side<S>,
            blis_uplo<Tri>,
            blis_trans<A>,
            blis_diag<D>,
            bv.length(0),
            bv.length(1),
            blis_ptr(&alpha),
            blis_ptr(av.data()),
            av.stride(0),
            av.stride(1),
            blis_ptr(bv.data()),
            bv.stride(0),
            bv.stride(1),
            nullptr,
            rt.get());
}

/// C := alpha·A·Aᴴ + beta·C in C's triangle: herk, her on a vector A (after
/// scalm for beta, which her lacks), and dotxv on a vector A and an element
/// C. As BLIS's herk and her do, the dotxv form leaves C's diagonal real.
template <triangle Tri, class A, class C>
void leaf_herk(const blis& leaf,
               Tri,
               real_t<element_t<C>> alpha,
               A&& a,
               real_t<element_t<C>> beta,
               C&& c)
{
    using namespace detail;
    using T = element_t<C>;
    FAMLIES_EXPECTS("herk", blis_fits(a) && blis_fits(c), a, c);
    const blis_runtime rt("herk", leaf);
    auto&& av = strip_conj(a);

    if constexpr (rank_v<C> == 0)
    {
        // c := alpha·a·aᴴ + beta·c, a a row.
        const T alpha_t(alpha);
        const T beta_t(beta);
        blis_pick<T>(bli_sdotxv_ex,
                     bli_ddotxv_ex,
                     bli_cdotxv_ex,
                     bli_zdotxv_ex)(blis_conj<A>,
                                    is_conjugated_v<A> ? BLIS_NO_CONJUGATE
                                                       : BLIS_CONJUGATE,
                                    av.length(0),
                                    blis_ptr(&alpha_t),
                                    blis_ptr(av.data()),
                                    av.stride(0),
                                    blis_ptr(av.data()),
                                    av.stride(0),
                                    blis_ptr(&beta_t),
                                    blis_ptr(&c),
                                    nullptr,
                                    rt.get());
        if constexpr (is_complex_v<T>)
            c.imag(0);
    }
    else
    {
        auto cv = as_view(std::forward<C>(c));
        if constexpr (rank_v<A> == 1)
        {
            if (beta != 1)
            {
                const T beta_t(beta);
                blis_pick<T>(bli_sscalm_ex,
                             bli_dscalm_ex,
                             bli_cscalm_ex,
                             bli_zscalm_ex)(BLIS_NO_CONJUGATE,
                                            0,
                                            BLIS_NONUNIT_DIAG,
                                            blis_uplo<Tri>,
                                            cv.length(0),
                                            cv.length(1),
                                            blis_ptr(&beta_t),
                                            blis_ptr(cv.data()),
                                            cv.stride(0),
                                            cv.stride(1),
                                            nullptr,
                                            rt.get());
            }
            blis_pick<T>(bli_sher_ex, bli_dher_ex, bli_cher_ex, bli_zher_ex)(
                blis_uplo<Tri>,
                blis_conj<A>,
                cv.length(0),
                &alpha,
                blis_ptr(av.data()),
                av.stride(0),
                blis_ptr(cv.data()),
                cv.stride(0),
                cv.stride(1),
                nullptr,
                rt.get());
        }
        else
            blis_pick<T>(bli_sherk_ex,
                         bli_dherk_ex,
                         bli_cherk_ex,
                         bli_zherk_ex)(blis_uplo<Tri>,
                                       blis_trans<A>,
                                       cv.length(0),
                                       av.length(1),
                                       &alpha,
                                       blis_ptr(av.data()),
                                       av.stride(0),
                                       av.stride(1),
                                       &beta,
                                       blis_ptr(cv.data()),
                                       cv.stride(0),
                                       cv.stride(1),
                                       nullptr,
                                       rt.get());
    }
}

/// C := alpha·A·B + beta·C: gemm, gemv on a matrix and a vector (either
/// way round), and dotxv on two vectors and an element C.
template <class A, class B, class C>
void leaf_gemm(const blis& leaf,
               element_t<C> alpha,
               A&& a,
               B&& b,
               element_t<C> beta,
               C&& c)
{
    using namespace detail;
    using T = element_t<C>;
    FAMLIES_EXPECTS("gemm",
                    blis_fits(a) && blis_fits(b) && blis_fits(c),
                    a,
                    b,
                    c);
    const blis_runtime rt("gemm", leaf);
    auto&& av = strip_conj(a);
    auto&& bv = strip_conj(b);

    if constexpr (rank_v<C> == 0)
        // c := alpha·a·b + beta·c, a a row and b a column.
        blis_pick<T>(bli_sdotxv_ex,
                     bli_ddotxv_ex,
                     bli_cdotxv_ex,
                     bli_zdotxv_ex)(blis_conj<A>,
                                    blis_conj<B>,
                                    av.length(0),
                                    blis_ptr(&alpha),
                                    blis_ptr(av.data()),
                                    av.stride(0),
                                    blis_ptr(bv.data()),
                                    bv.stride(0),
                                    blis_ptr(&beta),
                                    blis_ptr(&c),
                                    nullptr,
                                    rt.get());
    else if constexpr (rank_v<C> == 1)
    {
        auto cv = as_view(std::forward<C>(c));
        const auto gemv = blis_pick<T>(bli_sgemv_ex,
                                       bli_dgemv_ex,
                                       bli_cgemv_ex,
                                       bli_zgemv_ex);
        // gemv's m and n are its matrix's own, before transposing.
        if constexpr (rank_v<A> == 2)
            // c := alpha·A·b + beta·c, b and c columns.
            gemv(blis_trans<A>,
                 blis_conj<B>,
                 av.length(0),
                 av.length(1),
                 blis_ptr(&alpha),
                 blis_ptr(av.data()),
                 av.stride(0),
                 av.stride(1),
                 blis_ptr(bv.data()),
                 bv.stride(0),
                 blis_ptr(&beta),
                 blis_ptr(cv.data()),
                 cv.stride(0),
                 nullptr,
                 rt.get());
        else
            // c := alpha·a·B + beta·c, a and c rows: c := alpha·Bᵀ·a + beta·c.
            gemv(blis_transposed<B>,
                 blis_conj<A>,
                 bv.length(0),
                 bv.length(1),
                 blis_ptr(&alpha),
                 blis_ptr(bv.data()),
                 bv.stride(0),
                 bv.stride(1),
                 blis_ptr(av.data()),
                 av.stride(0),
                 blis_ptr(&beta),
                 blis_ptr(cv.data()),
                 cv.stride(0),
                 nullptr,
                 rt.get());
    }
    else
    {
        auto cv = as_view(std::forward<C>(c));
        blis_pick<T>(bli_sgemm_ex, bli_dgemm_ex, bli_cgemm_ex, bli_zgemm_ex)(
            blis_trans<A>,
            blis_trans<B>,
            cv.length(0),
            cv.length(1),
            av.length(1),
            blis_ptr(&alpha),
            blis_ptr(av.data()),
            av.stride(0),
            av.stride(1),
            blis_ptr(bv.data()),
            bv.stride(0),
            bv.stride(1),
            blis_ptr(&beta),
            blis_ptr(cv.data()),
            cv.stride(0),
            cv.stride(1),
            nullptr,
            rt.get());
    }
}

/// chol on an element, which BLIS lacks.
template <class A>
std::expected<void, not_positive_definite> leaf_chol(const blis&, A&& a)
{ return detail::chol_element(std::forward<A>(a)); }

/// BLIS's KC for T on this machine, at most 256. A rank-KC update is one
/// pass of BLIS's gemm, but KC can be large enough to leave a blocked loop
/// one block.
template <blis_element T> len_type default_block_size(const blis&)
{
    return std::min<len_type>(
        bli_cntx_get_blksz_def_dt(detail::blis_datatype<T>,
                                  BLIS_KC,
                                  bli_gks_query_cntx()),
        256);
}

} // namespace famlies

#endif // FAMLIES_LEAF_BLIS_HPP
