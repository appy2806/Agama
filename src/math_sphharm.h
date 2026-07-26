/** \file   math_sphharm.h
    \brief  Legendre polynomials and spherical-harmonic transformations
    \date   2015-2016
    \author Eugene Vasiliev
*/
#pragma once
#include "coord.h"
#include "gpu_device.h"  // AGAMA_DEVICE_INLINE
#include "math_core.h"      // math::powT
#include "math_specfunc.h"  // math::factorial / dfactorial (host-only m>LEGENDRE_MMAX fallback)
#include <vector>
#include <utility>
#include <cmath>
#ifndef __CUDA_ARCH__
#include <stdexcept>
#endif

namespace math {

/** Highest order m for which legendrePmm() below carries tabulated normalization
    constants, and hence the highest order at which the Legendre recurrence is
    callable from device code. Orders m=0..16 are upstream's literals; 17..32 were
    generated from the same closed-form fallback expressions and round-trip to the
    identical doubles, so the CPU result is unchanged at every m <= LEGENDRE_MMAX
    (verified bit-for-bit in tests/test_gpu_policy.cpp).
    Beyond this the fallback needs math::factorial / math::dfactorial, which are
    GSL-backed and host-only; see legendrePmm(). */
const int LEGENDRE_MMAX = 32;

/** Array of normalized associate Legendre polynomials W and their derivatives for l=m..lmax
    (theta-dependent factors in spherical-harmonic expansion):
    \f$  Y_l^m(\theta, \phi) = W_l^m(\theta) \{\sin,\cos\}(m\phi) ,
         W_l^m = \sqrt{\frac{ (2l+1) (l-m)! }{ 4\pi (l+m)! }} P_l^m(\cos(\theta))  \f$,
    where P are un-normalized associated Legendre functions.
    \param[in]  lmax - the maximum degree l of computed polynomials.
    \param[in]  m    - the order m: 0 <= m <= lmax.
    \param[in]  tau  - the argument of the function: tau = cos(theta) / (sin(theta) + 1),
    where theta is the usual polar angle. The rationale for choosing this combination,
    which is simply tan( (pi/2 - theta) / 2), is twofold: on the one hand, using cos(theta)
    as the argument leads to loss of precision when |cos|->1, which happens already for
    theta as large as 1e-8; on the other hand, using theta as argument would incur additional
    expense to compute both sin and cos theta. By contrast, they are simply computed from tau
    as sin(theta) = (1 - tau^2) / (1 + tau^2), cos(theta) = 2*tau / (1 + tau^2),  and
    at the same time |d tau / d theta| is always between 1/2 and 1, without loss of precision.
    \param[out] resultArray - the computed array of W_l^m,  which must be pre-allocated
    to contain lmax-m+1 elements, stored as  l=m, m+1, ..., lmax;
    \param[out] derivArray - the computed array of derivatives dW_l^m / d theta
    (not cos theta, nor tau); if NULL, they are not computed, otherwise must be pre-allocated
    with the same size as resultArray. 
    \param[out] deriv2Array - the computed array of second derivatives d2W_l^m / d theta^2;
    if NULL, they are not computed, and if not NULL, derivArray must be not NULL too.
    The derivatives for |tau| -> 1 (i.e. theta->0 or theta->pi) are computed accurately
    using asymptotic expressions.

    Defined inline in this header (and tagged AGAMA_DEVICE_INLINE) so it is callable
    from both CPU code and __device__ kernels: it writes only into caller-provided
    arrays and needs no dynamic allocation. Device callers must keep m <= LEGENDRE_MMAX
    (see legendrePmm below).

    Templated on the value type NumT, exactly as evalQuinticSplines<K>/evalCubicSplines<K>
    in math_spline.h already are: NumT is deduced from the INPUT argument (tau) only --
    the three output pointers go through nondeduced<> so call sites passing a literal
    NULL for derivArray/deriv2Array (there are several, via the defaults below and at
    call sites) still compile. Every pre-existing call site is written
    `sphHarmArray(lmax, m, tau, ...)` with tau a double, so NumT deduces to double and
    the generated code for existing callers is unchanged.
*/
template<typename NumT>
AGAMA_DEVICE_INLINE
void sphHarmArray(const unsigned int lmax, const unsigned int m, const NumT tau,
    typename nondeduced<NumT>::type* resultArray,
    typename nondeduced<NumT>::type* derivArray=NULL,
    typename nondeduced<NumT>::type* deriv2Array=NULL);

/** Calculate P_m^m(theta) from the analytic result:
    P_m^m(theta) = (-1)^m (2m-1)!! (sin(theta))^m , m > 0 ;
                 = 1 , m = 0 .
    store the pre-factor sqrt[ (2*m+1) / (4 pi (2m)!) ] in prefact,
    the value of Pmm in val, and optionally its first/second derivative w.r.t theta
    in der/der2 if they are not NULL.

    Device-callable for m <= LEGENDRE_MMAX, which is what the tabulated PREFACT/COEF
    constants cover. For larger m the normalization has to be built from
    math::factorial / math::dfactorial, which are GSL-backed and therefore host-only;
    on the device that case writes NAN rather than silently disagreeing with the host.
    Callers that build device descriptors must reject expansions with m > LEGENDRE_MMAX
    on the host (fail closed) so the NAN branch is unreachable in practice.

    Templated on the value type NumT, exactly as evalQuinticSplines<K> in math_spline.h:
    NumT is deduced from the INPUT arguments (costheta, sintheta) only -- prefact is an
    in/out reference of the same NumT (never NULL-able, so it can safely participate in
    deduction), while value/der/der2 go through nondeduced<> since they are pure outputs.
    The PREFACT/COEF tables below stay `double` in storage (coefficient storage is always
    fp64 per CLAUDE.md hard constraint #4) and are cast to NumT at the point of use, so an
    NumT=float instantiation still starts from the exact fp64 constant rather than a
    re-rounded float table. For NumT=double every such cast is the identity (double(x)==x
    for x already double), so this is bit-for-bit unchanged from before templating. */
template<typename NumT>
AGAMA_DEVICE_INLINE
void legendrePmm(int m, NumT costheta, NumT sintheta,
    NumT& prefact,
    typename nondeduced<NumT>::type* value,
    typename nondeduced<NumT>::type* der,
    typename nondeduced<NumT>::type* der2)
{
    const int MMAX = LEGENDRE_MMAX;  // # of pre-computed coefs in the tables below
    // entries 0..16 are upstream's literals; 17..32 round-trip the closed-form
    // fallback expressions below, so no m <= MMAX changes value w.r.t. upstream.
    // Kept `double` (not NumT[]) -- see the comment above the function.
    static const double PREFACT[MMAX+1] = { 0.2820947917738782,
        0.3454941494713355,    0.1287580673410632,    0.02781492157551894,   0.004214597070904597,
        0.0004911451888263050, 4.647273819914057e-05, 3.700296470718545e-06, 2.542785532478802e-07,
        1.536743406172476e-08, 8.287860012085477e-10, 4.035298721198747e-11, 1.790656309174350e-12,
        7.299068453727266e-14, 2.751209457796109e-15, 9.643748535232993e-17, 3.159120301003413e-18,
        9.7128523792757242e-20, 2.8133797388083946e-21, 7.7031283932527599e-23, 1.9996982303404461e-24,
        4.9350344437027061e-26, 1.1606510034403698e-27, 2.607108770058365e-29,  5.6045237507440517e-31,
        1.155161536689941e-32,  2.2866979724449814e-34, 4.3542905194887195e-36, 7.9872656096230965e-38,
        1.4133029977144382e-39, 2.4153082278063579e-41, 3.991325582870159e-43,  6.3847411917613422e-45 };
    static const double COEF[MMAX+1] =  { 0.2820947917738782,
        -0.3454941494713355, 0.3862742020231896, -0.4172238236327841, 0.4425326924449826,
        -0.4641322034408582, 0.4830841135800662, -0.5000395635705506, 0.5154289843972843,
        -0.5295529414924496, 0.5426302919442215, -0.5548257538066191, 0.5662666637421912,
        -0.5770536647012670, 0.5872677968601020, -0.5969753602424046, 0.6062313441538353,
        -0.61508190492882853, 0.62356619406092162, -0.63171773211594939, 0.63956545825776223,
        -0.64713454371390633, 0.65444703055069287, -0.66152233920742165, 0.66837767607862275,
        -0.6750283640247019,  0.68148811277807653, -0.68776924198859157, 0.69388286659278342,
        -0.69983905194657747, 0.70564694449371002, -0.71131488249010422, 0.71685049035448933 };

    if(m<=MMAX)
        prefact = NumT(PREFACT[m]);  // cast on load: table stays fp64, prefact is NumT
    else {
#ifdef __CUDA_ARCH__
        // no device path for the GSL-backed factorial; fail loudly rather than
        // return a value that disagrees with the host (see the comment above)
        prefact = NAN;
#else
        // rare, host-only fallback (m>32): the GSL factorial call is already double,
        // so this stays double arithmetic until the final NumT cast -- harmless, since
        // device code (the performance-sensitive path) never reaches m>LEGENDRE_MMAX.
        prefact = NumT(0.5/M_SQRTPI * sqrt( (2*m+1) / factorial(2*m) ));
#endif
    }
    if(m == 0) {
        if(der)
            *der = 0;
        if(der2)
            *der2= 0;
        *value   = prefact;
        return;
    }
    if(m == 1) {
        if(der)
            *der = -costheta * prefact;
        if(der2)
            *der2=  sintheta * prefact;
        *value   = -sintheta * prefact;
        return;
    }
    NumT coef;
    if(m<=MMAX)
        coef = NumT(COEF[m]);        // cast on load, as PREFACT above
    else {
#ifdef __CUDA_ARCH__
        coef = NAN;
#else
        coef = NumT(prefact * dfactorial(2*m-1) * (m%2 == 1 ? -1 : 1));
#endif
    }
    NumT sinm2 = math::powT(sintheta, m-2);
    if(der)
        *der = m * coef * sinm2 * sintheta * costheta;
    if(der2)
        *der2= m * coef * sinm2 * (m * pow_2(costheta) - 1);
    *value   =     coef * sinm2 * pow_2(sintheta);
}

/** Threshold in sin(theta) below which sphHarmArray's derivative recurrence switches
    from the direct formula (which cancels catastrophically as sin(theta)->0, then
    divides by that same small number) to closed-form asymptotic expressions -- but
    ONLY for m<=2 (m==0,1,2 are the special-cased branches below); for m>2 the direct
    formula is used unconditionally whenever st>0, exactly as upstream, and this
    threshold plays no role there (see the comment at the sqrt-threshold sweep in
    tests/test_gpu_policy.cpp for the evidence that m>2 is unaffected by this choice).

    fp64: this MUST stay the literal upstream used, 1e-8, bit-for-bit -- hard
    constraint #3 (bit-identical legacy fp64 behaviour). Do NOT replace it with
    std::sqrt(std::numeric_limits<double>::epsilon()): that evaluates to
    1.4901161193847656e-08, not 1e-8, and would move every existing BFE/Multipole
    result computed via the m<=2 branches.

    fp32: NOT simply sqrt(FLT_EPSILON)=3.45e-4, even though that is the scaling that
    picked the NFW Pade-guard fp32 threshold (potential_analytic.h, nfw_pade_guard<float>).
    Measured value is 3e-3, ~10x larger. The mechanism, which is worth understanding
    because it also explains why upstream's fp64 choice looks so aggressive:

    The direct formula subtracts two O(1) quantities to obtain a result of order st^2,
    then divides by st. So its RELATIVE error behaves as ~eps/st^2, and the crossover
    st = sqrt(eps) is precisely where that reaches ~100%. That model reproduces the whole
    upper half of the table below: sqrt(FLT_EPSILON) -> predicted ~1.0, measured 0.93;
    st=1e-4 -> predicted ~12, measured 3.8; st=1e-5 -> predicted ~1e3, measured 3.0e2.
    In other words the sqrt(eps) crossover is the point of TOTAL relative-error loss, not
    the point of balanced error, and the errors at 3.45e-4 come from the direct formula
    used just ABOVE the threshold -- not from the asymptotic form.

    Upstream can live with that at fp64 because dPlm is itself O(st) near the pole, so the
    ABSOLUTE error there is only ~eps/st = 2e-8; the fp64 threshold is effectively chosen
    on an absolute-error criterion. At fp32 the same criterion would leave ~1e-3 absolute
    and ~93% relative error, so the value below is instead chosen to minimize WORST-CASE
    RELATIVE error -- a deliberately stricter criterion than upstream's, which is why the
    two thresholds are not related by a clean sqrt(eps) factor. Do not "restore
    consistency" by scaling one from the other.

    The lower half of the table IS the asymptotic form's own truncation error, which grows
    with st (an l-dependent leading-order expression), and is what stops the threshold from
    being pushed higher still. The minimum sits between the two mechanisms.

    Empirically swept (m=0,1,2,
    l up to 24, tau densely sampled so st spans ~1e-8..1 straddling every candidate,
    fp32 output vs a same-formula fp64 reference fed the identical float-rounded tau):

        EPS       worst rel err (dPlm/d2Plm, m in {0,1,2})
        1e-5      3.0e+02   (direct-formula cancellation in fp32 near the pole)
        1e-4      3.8e+00
        3.45e-4   9.3e-01   (sqrt(FLT_EPSILON): ~100% rel err, as the eps/st^2 model predicts)
        1e-3      1.1e-01
        3e-3      2.0e-02   <- minimum, flat through 5e-3
        1e-2      2.4e-02   (asymptotic truncation error growing again)
        1e-1      7.0e+01

    3e-3 sits at the flat bottom of that curve and is the value used below. See
    tests/test_gpu_policy.cpp for the full table and the regression test that pins it. */
template<typename NumT> struct sphharm_deriv_eps;  // no default: an unsupported NumT won't compile
template<> struct sphharm_deriv_eps<double> {
    static AGAMA_DEVICE_INLINE double value() { return 1e-8; }
};
template<> struct sphharm_deriv_eps<float> {
    static AGAMA_DEVICE_INLINE float value() { return 3e-3f; }
};

template<typename NumT>
AGAMA_DEVICE_INLINE
void sphHarmArray(const unsigned int lmax, const unsigned int m, const NumT tau,
    typename nondeduced<NumT>::type* resultArray,
    typename nondeduced<NumT>::type* derivArray,
    typename nondeduced<NumT>::type* deriv2Array)
{
    if(m>lmax || resultArray==NULL || (deriv2Array!=NULL && derivArray==NULL)) {
#ifdef __CUDA_ARCH__
        return;   // cannot throw from a kernel; the host-side caller validates
#else
        throw std::domain_error("Invalid parameters in sphHarmArray");
#endif
    }
    if(lmax==0) {
        resultArray[0] = NumT(0.5)/NumT(M_SQRTPI);
        if(derivArray)
            derivArray[0] = 0;
        if(deriv2Array)
            deriv2Array[0] = 0;
        return;
    }
    const NumT ct =      2 * tau  / (1 + tau*tau);  // cos(theta)
    const NumT st = (1 - tau*tau) / (1 + tau*tau);  // sin(theta)
    NumT prefact; // will be initialized by legendrePmm
    legendrePmm(m, ct, st, prefact, resultArray, derivArray, deriv2Array);
    if(lmax == m)
        return;

    // values of two previous un-normalized polynomials needed in the recurrent relation
    NumT Plm1 = resultArray[0] / prefact, Plm = ct * (2*m+1) * Plm1, Plm2 = 0;
    // values of 2nd derivatives of un-normalized polynomials needed for the special case
    // m==1 and st<<1, since we need another recurrent relation for computing 2nd derivative -
    // the usual formula suffers from cancellation
    NumT d2Plm1 = st, d2Plm2 = 0, d2Plm = 12 * ct * st;
    // threshold in sin(theta) for applying asymptotic expressions for derivatives --
    // see sphharm_deriv_eps<NumT> above for why this is precision-dependent and how
    // the fp32 value was derived.
    const NumT EPS = sphharm_deriv_eps<NumT>::value();

    for(int l=m+1; l<=(int)lmax; l++) {
        unsigned int ind = l-m;  // index in the output array
        if(l>(int)m+1)  // skip first iteration which was assigned above
            Plm = (ct * (2*l-1) * Plm1 - (l+m-1) * Plm2) / (l-m);  // use recurrence for the rest
        // NumT(...) on every operand of the ratio below (not the `2*l+1.`-style bare
        // double literal upstream used) so a NumT=float instantiation computes this
        // sqrt in float, not double -- l and m are ints and convert to NumT exactly,
        // so for NumT=double this is bit-identical to the original expression.
        prefact *= std::sqrt(NumT(2*l+1) / NumT(2*l-1) * NumT(l-m) / NumT(l+m));
        resultArray[ind] = Plm * prefact;
        if(derivArray) {
            NumT dPlm = 0;
            if(st >= EPS || (m>2 && st>0))
                dPlm = (l * ct * Plm - (l+m) * Plm1) / st;
            else if(m==0)
                dPlm = -l*(l+1)/2 * st * (ct>0 || l%2==1 ? 1 : -1);
            else if(m==1)
                dPlm = -l*(l+1)/2 * (ct>0 || l%2==0 ? 1 : -1);
            else if(m==2)
                dPlm = l*(l+1)*(l+2)*(l-1)/4 * st * (ct>0 || l%2==1 ? 1 : -1);
            derivArray[ind] = prefact * dPlm;
        }
        if(deriv2Array!=NULL) {
            if(st >= EPS || (m>2 && st>0))
                deriv2Array[ind] = ct * derivArray[ind] / (-st) - (l*(l+1)-pow_2(m/st)) * resultArray[ind];
            else if(m==0)
                deriv2Array[ind] = -l*(l+1)/2 * prefact * (ct>0 || l%2==0 ? 1 : -1);
            else if(m==1) {
                if(l>(int)m+1) {
                    NumT twodPlm1 = -l*(l-1) * (ct>0 || l%2==1 ? 1 : -1);
                    d2Plm = ( (2*l-1) * (ct * (d2Plm1 - Plm1) - st * twodPlm1) - l * d2Plm2) / (l-1);
                }
                deriv2Array[ind] = prefact * d2Plm;
                d2Plm2 = d2Plm1;
                d2Plm1 = d2Plm;
            }
            else if(m==2)
                deriv2Array[ind] = l*(l+1)*(l+2)*(l-1)/4 * prefact * (ct>0 || l%2==0 ? 1 : -1);
            else
                deriv2Array[ind] = 0;
        }
        Plm2 = Plm1;
        Plm1 = Plm;
    }
}

/** Compute the values of cosines and optionally sines of an arithmetic progression of angles:
    cos(phi), cos(2 phi), ..., cos(m phi), [ sin(phi), sin(2 phi), ..., sin(m phi) ].
    \param[in]  phi - the angle;
    \param[in]  m   - the number of multiples of this angle to process, must be >=1;
    \param[in]  needSine - whether to compute sines as well (if false then only cosines are computed);
    \param[out] outputArray - pointer to an existing array of length m (if needSine==false)
    or 2m (if needSine==true) that will store the output values.

    Defined inline in the header (and tagged AGAMA_DEVICE_INLINE) so it is callable
    from both CPU code and __device__ kernels. The recurrence is from
    Num.Rec. 3rd ed. section 5.4: given alpha = 2 sin^2(phi/2) and beta = sin(phi),
    successive multiples are cos((k+1)phi) = cos(k phi) - (alpha cos(k phi) + beta sin(k phi)).

    Templated on the value type NumT, exactly as sphHarmArray/legendrePmm above: NumT is
    deduced from the INPUT argument (phi) only, outputArray goes through nondeduced<>.
    Every pre-existing call site passes a double phi, so NumT deduces to double and the
    generated code is unchanged. No fp64-tuned guard/threshold here (see the "no
    comparable constant" note in tests/test_gpu_policy.cpp / the commit message) --
    the recurrence is pure trigonometric arithmetic with no cancellation-avoidance
    branch, so there is nothing to re-derive for fp32 beyond the literal-wrapping below.
*/
template<typename NumT>
AGAMA_DEVICE_INLINE
void trigMultiAngle(const NumT phi, const unsigned int m, const bool needSine,
    typename nondeduced<NumT>::type* outputArray)
{
    if(m < 1) return;
    // sin/cos called separately (not via math::sincos) so the body is device-portable.
    // nvcc's optimizer fuses these into a single sincos under -O2.
    const NumT sinphi  = std::sin(phi);
    const NumT sinphi2 = std::sin(phi * NumT(0.5));
    // `2`/`1`/`0` below are bare int literals (not 2.0/1.0/0.0 as upstream spelled them):
    // an int operand converts exactly to NumT without ever becoming double (recipe item
    // 6), whereas the original `2.0 * sinphi2 * sinphi2` would force this multiplication
    // into double for a NumT=float instantiation, exactly the promotion this templating
    // exists to avoid. For NumT=double the value is identical either way (2 converts to
    // 2.0 exactly), so this changes nothing for existing callers.
    const NumT alpha   = 2 * sinphi2 * sinphi2;  // 2 sin^2(phi/2) = 1 - cos(phi)
    const NumT beta    = sinphi;
    NumT cosphi1 = 1, sinphi1 = 0;
    for(unsigned int k = 0; k < m; ++k) {
        const NumT cosphi = cosphi1 - (alpha * cosphi1 + beta * sinphi1);
        const NumT sinphi_k = sinphi1 - (alpha * sinphi1 - beta * cosphi1);
        outputArray[k] = cosphi;
        if(needSine) outputArray[k + m] = sinphi_k;
        cosphi1 = cosphi;
        sinphi1 = sinphi_k;
    }
}

/** Indexing scheme for spherical-harmonic transformation.
    It defines the maximum order of expansion in theta (lmax) and phi (mmax),
    that should satisfy 0 <= mmax <= lmax (0 means using only one term),
    and also defines which coefficients to skip (they are assumed to be identically zero
    due to particular symmetries of the function), specified by step, lmin and mmin.
    Namely, the loop over non-zero coefficients should look like
    \code
    for(int m=ind.mmin(); m<=ind.mmax; m++)
       for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
           doSomethingWithCoefficient(ind.index(l, m));
    \endcode
    This scheme is used in the forward and inverse SH transformation routines.

    The coefficients are arranged in the triangular shape, as follows:
    \code
    m=   -4  -3  -2  -1   0   1   2   3   4
    l=0                   0
    l=1               1   2   3
    l=2           4   5   6   7   8
    l=3       9  10  11  12  13  14  15
    l=4  16  17  18  19  20  21  22  23  24
    \endcode
    m>=0 correspond to cos(m phi) and m<0 - to sin(|m| phi).
    Regardless of mmax, the length of the coefficients array is always (lmax+1)^2.

    Various symmetries imply that some of the terms must be zero, as detailed in the table below,
    where 0 stands for a coefficient that must be zero, and dot - for a possibly non-zero one
    \code
    +-----------------------------------------------------------+
    | invariance       |       coefficients that are zero       |
    | under transform  |     l is odd    |      l is even  |    |
    |                  |m>=0 m>0 m<0  m<0|m>=0 m>0 m<0  m<0|m!=0|
    |                  |even odd even odd|even odd even odd|    |
    |------------------+----------------------------------------|
    |x => -x           | .    0   0    .    .   0   0    .    . | ST_XREFLECTION
    |y => -y           | .    .   0    0    .   .   0    0    . | ST_YREFLECTION
    |z => -z           | 0    .   0    .    .   0   .    0    . | ST_ZREFLECTION
    |x,y,z => -x,-y,-z | 0    0   0    0    .   .   .    .    . | ST_REFLECTION
    |x,y=>-x,-y | z=>-z| 0    0   0    0    .   0   .    0    . | ST_BISYMMETRIC
    |triaxial sym.     | 0    0   0    0    .   0   0    0    . | ST_TRIAXIAL
    |z-axis rotation   | .    0   0    0    .   0   0    0    0 | ST_ZROTATION
    |axisymmetric      | 0    0   0    0    .   0   0    0    0 | ST_AXISYMMETRIC
    |spherical         |     only  l=0, m=0 remains nonzero     | ST_SPHERICAL
    +-----------------------------------------------------------+
    \endcode
    Another way of representing these symmetries is shown on the diagrams below:
    \code
                x-reflection                    z-reflection
    m=   -4 -3 -2 -1  0  1  2  3  4      -4 -3 -2 -1  0  1  2  3  4
    l=0               .                               .
    l=1            .  .  0                         .  0  .
    l=2         0  .  .  0  .                   .  0  .  0  .
    l=3      .  0  .  .  0  .  0             .  0  .  0  .  0  .
    l=4   0  .  0  .  .  0  .  0  .       .  0  .  0  .  0  .  0  .

                y-reflection                mirror (xyz-reflection)
    m=   -4 -3 -2 -1  0  1  2  3  4       -4 -3 -2 -1  0  1  2  3  4
    l=0               .                                .
    l=1            0  .  .                          0  0  0
    l=2         0  0  .  .  .                    .  .  .  .  .
    l=3      0  0  0  .  .  .  .              0  0  0  0  0  0  0
    l=4   0  0  0  0  .  .  .  .  .        .  .  .  .  .  .  .  .  .

    \endcode
*/
/** Device-callable POD mirror of SphHarmIndices (declared below).

    SphHarmIndices itself cannot cross into a CUDA kernel: it holds a std::vector<int>
    lmin_arr. That array is however a pure closed form of (lmax, mmax, sym) -- see the
    constructor in math_sphharm.cpp, whose only inputs are the symmetry predicates in
    coord.h, all of which are already AGAMA_DEVICE_INLINE. So the whole indexing scheme
    collapses to these four ints, and the Tier 2 Multipole descriptor needs no device
    pointer for it.

    Build one ONLY via SphHarmIndices::pod(). Reconstructing it from the lmax/mmax/sym a
    caller originally asked for is WRONG: the constructor augments sym after validating
    its arguments (mmax==0 implies z-rotation + x/y-reflection symmetry, lmax==0 implies
    full rotation + z- and xyz-reflection), and lmin() reads the augmented value. */
struct SphHarmIndicesPod {
    int lmax;  ///< order of expansion in theta (>=0)
    int mmax;  ///< order of expansion in phi (0<=mmax<=lmax)
    int step;  ///< 1 if all l terms are used, 2 if only every other l term for each m is used
    int sym;   ///< coord::SymmetryType, AFTER the constructor's augmentation

    /// number of elements in the array of spherical-harmonic coefficients
    AGAMA_DEVICE_INLINE int size() const { return (lmax+1)*(lmax+1); }

    /// index of coefficient with the given l and m (no range check, as in SphHarmIndices)
    AGAMA_DEVICE_INLINE static int index(int l, int m) { return l*(l+1)+m; }

    /// minimum m-index
    AGAMA_DEVICE_INLINE int mmin() const {
        return isYReflSymmetric(static_cast<coord::SymmetryType>(sym)) ? 0 : -mmax;
    }

    /** minimum l-index for the given m (if larger than lmax, this m is not used).
        Restatement of the loop body that fills SphHarmIndices::lmin_arr; the test in
        tests/test_gpu_policy.cpp asserts the two agree for every m over a sweep of
        (lmax, mmax, sym) including the lmax==0 and mmax==0 augmentation cases. */
    AGAMA_DEVICE_INLINE int lmin(int m) const {
        if(!(m>=-mmax && m<=mmax))
            return lmax+1;
        const coord::SymmetryType s = static_cast<coord::SymmetryType>(sym);
        const int absm = m<0 ? -m : m;
        const bool odd = m%2 != 0;
        if( (isYReflSymmetric(s) && m<0) ||
            (isXReflSymmetric(s) && ((m<0) ^ odd)) ||
            (isBisymmetric(s)    && odd) )
            return lmax+1;               // don't consider this m at all
        if(isReflSymmetric(s) && odd)
            return absm+1;               // start from the next even l, because step in l is 2
        return absm;
    }
};

class SphHarmIndices {
public:
    const int
    lmax,  ///< order of expansion in theta (>=0)
    mmax,  ///< order of expansion in phi (0<=mmax<=lmax)
    step;  ///< 1 if all l terms are used, 2 if only every other l term for each m is used

    /** Create the spherical-harmonic indexing scheme for the given symmetry type
        and expansion order.
        \param[in] sym  - the type of symmetry that determines which coefficients to omit;
        \param[in] lmax - order of expansion in polar angle (theta);
        \param[in] mmax - order of expansion in azimuthal angle (phi);
        \returns   the instance of indexing scheme to be passed to spherical harmonic transform.
    */
    SphHarmIndices(int lmax, int mmax, coord::SymmetryType sym);

    /// return symmetry properties of this index set
    coord::SymmetryType symmetry() const { return sym; }

    /// number of elements in the array of spherical-harmonic coefficients
    inline unsigned int size() const { return (lmax+1)*(lmax+1); }

    /// index of coefficient with the given l and m
    /// (0<=l<=lmax, -l<=m<=l, no range check performed!)
    static unsigned int index(int l, int m) { return l*(l+1)+m; }

    /// decode the l-index from the combined index of a coefficient
    static int index_l(unsigned int c);

    /// decode the m-index from the combined index of a coefficient
    static int index_m(unsigned int c);

    /// minimum l-index for the given m (if larger than lmax, it means that this value of m is not used)
    inline int lmin(int m) const { return m>=-mmax && m<=mmax ? lmin_arr[m+mmax] : lmax+1; }

    /// minimum m-index
    inline int mmin() const { return isYReflSymmetric(sym) ? 0 : -mmax; }

    /** the device-callable POD view of this indexing scheme -- the only sanctioned way to
        construct a SphHarmIndicesPod, because it is the only place the augmented `sym` is
        visible (see the comment on that struct). */
    inline SphHarmIndicesPod pod() const {
        SphHarmIndicesPod p;
        p.lmax = lmax;
        p.mmax = mmax;
        p.step = step;
        p.sym  = static_cast<int>(sym);
        return p;
    }

private:
    coord::SymmetryType sym;   ///< symmetry properties of this index set
    std::vector<int> lmin_arr; ///< array of minimum l-indices for each m
};

/** Determine the order of expansion and its symmetry properties from the list of coefficients,
    by analyzing which of them are non-zero.
    \param[in]  C - an array of coefficients with length (lmax+1)^2;
    \returns    an instance of indexing scheme.
*/
SphHarmIndices getIndicesFromCoefs(const std::vector<double> &C);

/** Collect the list of azimuthal harmonic indices (m) that are non-zero
    for the given symmetry type and expansion order */
std::vector<int> getIndicesAzimuthal(int mmax, coord::SymmetryType sym);

/** Class for performing forward Fourier transformation
    (computing Fourier coefficients from the values of a function at equally spaced angles).
    The transformation may involve only cosine terms, or both sine and cosine terms:
    \f$  C_m    = \int_0^{2\pi} f(\phi) \cos( m \phi) d\phi  \f$,
    \f$  C_{-m} = \int_0^{2\pi} f(\phi) \sin(|m|\phi) d\phi  \f$, 0 <= m <= mmax.
    For the given expansion order and symmetry requirement, it computes transformation
    coefficients and stores in an internal table; the user then may perform many transformations
    with the same setup, by collecting the values of function at the grid of angles and
    calling `transform()` method, as follows:
    \code
    FourierTransformForward trans(mmax, useSine);
    std::vector<double> input_values(trans.size());
    for(unsigned int i=0; i<trans.size(); i++)
        input_values[i] = my_function(trans.phi(i));
    std::vector<double> output_coefs(trans.size());
    trans.transform(&values.front(), &output_coefs.front());
    \endcode
    If useSine flag is set, the transformation includes both sine and cosine terms,
    which are stored in the following order:
    output_coefs[mmax-m] = C_{-m}, 1 <= m <= mmax  (sine terms),
    output_coefs[mmax+m] = C_m,    0 <= m <= mmax  (cosine terms).
    If useSine is false, then the output contains only cosine terms:
    output_coefs[m] = C_m,  0 <= m <= mmax.
*/
class FourierTransformForward {
public:
    /** create the transformation of given order and symmetry */
    FourierTransformForward(int mmax, bool useSine);

    /** return the size of both input array of function values
        and the output array of coefficients */
    inline unsigned int size() const { return useSine ? mmax*2+1 : mmax+1; }

    /** return the i-th node of angular grid:
        if useSine=true, 0 <= phi_i < 2 pi, otherwise 0 <= phi_i < pi */
    inline double phi(unsigned int i) const { return i*M_PI/(mmax+0.5); }

    /** perform the transform on the collected function values.
        \param[in]  values is the array of function values at a regular grid in phi,
        arranged so that  values[i*stride] = f(phi(i)), 0 <= i < size()
        \param[out] coefs must point to an existing array of length `size()`,
        which will be filled with Fourier coefficients as described above
        \param[in]  stride (optional) is the spacing between consecutive elements
        in the input array (no stride in the output)
    */
    void transform(const double values[] /*in*/, double coefs[] /*out*/, int stride=1) const;
private:
    const int mmax;               ///< order of expansion
    const bool useSine;           ///< whether to use sine terms (if no then only cosines)
    std::vector<double> trigFnc;  ///< values of sine/cosine at the nodes of angular grid
};

/** Class for performing forward spherical-harmonic transformation.
    For the given coefficient indexing scheme, specified by an instance of `SphHarmIndices`,
    it computes the S-H coefficients C_lm from the values of function f(theta,phi)
    at nodes of a 2d grid in theta and phi.
    The workflow is the following:
      - create the instance of forward transform class for the given indexing scheme;
      - for each function f(theta,phi) the user should collect its values at the nodes of grid
        specified by member functions `theta(i)` and `phi(i)` into an array with length `size()`:
        \code
        SphHarmTransformForward trans(ind);
        std::vector<double> input_values(trans.size());
        for(unsigned int i=0; i<trans.size(); i++)
            input_values[i] = my_function(trans.theta(i), trans.phi(i));
        std::vector<double> output_coefs(ind.size());
        trans.transform(&values.front(), &output_coefs.front());
        \endcode
    Depending on the symmetry properties specified by the indexing scheme, not all elements
    of input_values need to be filled by the user; the transform routine takes this into account.
    The implementation uses 'naive' summation approach without any FFT or fast Legendre transform
    algorithms, has complexity O(lmax^2*mmax) and is only suitable for lmax <~ few dozen.
    The transformation is 'lossless' (to machine precision) if the original function is
    band-limited, i.e. given by a sum of spherical harmonics with order up to lmax and mmax.
*/
class SphHarmTransformForward {
public:
    /// initialize the grid in theta and the table of transformation coefficients
    SphHarmTransformForward(const SphHarmIndices& ind);

    /// return the required size of input array for the forward transformation
    inline unsigned int size() const { return thetasize() * fourier.size(); }

    /// return the cos(theta) coordinate (-1:1) of i-th element of input array, 0 <= i < size()
    inline double costheta(unsigned int i) const { return costhnodes[i / fourier.size()]; }

    /// return the phi coordinate [0:2pi) of i-th element of input array, 0 <= i < size()
    inline double phi(unsigned int i) const { return fourier.phi(i % fourier.size()); }

    /** perform the transformation of input array (values) into the array of coefficients.
        \param[in]  values is the array of function values at a rectangular grid in (theta,phi),
        arranged so that  values[i*stride] = f(theta(i), phi(i)), 0 <= i < size()
        \param[out] coefs must point to an existing array of length `ind.size()`,
        which will be filled with spherical-harmonic expansion coefficients as follows:
        coefs[ind.index(l,m)] = C_lm, 0 <= l <= lmax, -l <= m <= l
        \param[in]  stride (optional) is the spacing between consecutive elements in the input array
    */
    void transform(const double values[] /*in*/, double coefs[] /*out*/, int stride=1) const;

private:
    /// coefficient indexing scheme (including lmax and mmax)
    const SphHarmIndices ind;

    /// Fourier transform in azimuthal angle
    const FourierTransformForward fourier;

    /// coordinates of the grid nodes in theta on (0:pi/2]
    std::vector<double> costhnodes;

    /// values of all associated Legendre functions of order <= lmax,mmax at nodes of theta-grid
    std::vector<double> legFnc;

    /// number of sample points in theta, spanning either (0:pi/2] or (0:pi)
    unsigned int thetasize() const { return isZReflSymmetric(ind.symmetry()) ? ind.lmax/2+1 : ind.lmax+1; }
};

/** Routine for performing inverse spherical-harmonic transformation.
    Given the array of coefficients obtained by the forward transformation,
    it computes the value of function at the given position on unit sphere (theta,phi).
    \param[in]  ind   - coefficient indexing scheme, defining lmax, mmax and skipped coefs;
    \param[in]  coefs - the array of coefficients;
    \param[in]  tau   - cos(theta) / (1 + sin(theta)), where theta is the polar angle;
    \param[in]  phi   - the azimuthal angle;
    \returns    the value of function at (theta,phi)
*/
double sphHarmTransformInverse(const SphHarmIndices& ind, const double coefs[],
    const double tau, const double phi);

}  // namespace math
