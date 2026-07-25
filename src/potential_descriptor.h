/** \file    potential_descriptor.h
    \brief   POD "force descriptor" for calling potential leaf math from inside
             GPU kernels (Tier 3): a tagged-union list of analytic-potential
             terms that a device thread can evaluate per RK stage without any
             virtual dispatch or host callback.
    \date    2026
    \author  GPU-unification fork

    A GpuPotDesc is a by-value snapshot of a (possibly Composite) potential:
    one GpuPotTerm per concrete member, holding a type tag and the constructor
    parameters. It is built on the host from a BasePotential (same
    dynamic_cast dispatch as potential_gpu.cpp's AGAMA_GPU_POT_LIST -- the two
    capability sets must be kept in sync), captured by value into the device
    lambda (so it travels in the kernel-argument constant bank; no device
    allocations), and evaluated point-wise by gpu_desc_phi_acc(), which calls
    the SAME AGAMA_DEVICE_INLINE leaves (plummer_eval, nfw_eval, ...) as every
    other backend -- single source of math preserved.

    Parameter storage is always double (hard constraint #4: coefficient
    storage is fp64); castGpuPotDesc<T>() converts once on the host when a
    kernel is instantiated at lower precision.

    Modifiers (Shifted / Tilted / Rotating / Scaled) are not terms of their own:
    buildGpuPotDesc() descends through the wrapper chain and collapses it into
    the GpuPotXform carried by each term it wraps (see GpuPotXform in
    potential_composite.h for why an arbitrarily deep chain is exactly 13
    numbers). Since the transform depends on time, a descriptor is in general
    valid only at the `time` it was built for -- buildGpuPotDesc()'s
    requireTimeIndependent flag is how a caller that reuses one descriptor
    across many times (the orbit kernel) declines the ones that would go stale.
*/
#pragma once
#include "potential_base.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_dehnen.h"
#include "potential_disk.h"
#include "gpu_device.h"
#include <cmath>

namespace potential {

/// concrete potential types representable in a GpuPotTerm;
/// must match the GPU-capable set in potential_gpu.cpp (AGAMA_GPU_POT_LIST),
/// EXCEPT GPU_POT_DEHNEN and GPU_POT_DISK_ANSATZ: Dehnen is only representable
/// for the spherical case (axisRatioY==axisRatioZ==1), and DiskAnsatz only for
/// the 5 recognized closed-form radial/vertical functor types -- both are
/// special-cased in buildGpuPotDesc() below (and in potential_gpu.cpp outside
/// its AGAMA_GPU_POT_LIST macro; see the comment there) rather than folded
/// into the blanket per-type capability of the X-macro list.
enum GpuPotTag {
    GPU_POT_PLUMMER,
    GPU_POT_ISOCHRONE,
    GPU_POT_NFW,
    GPU_POT_MIYAMOTONAGAI,
    GPU_POT_LOGARITHMIC,
    GPU_POT_HARMONIC,
    GPU_POT_DEHNEN,
    GPU_POT_DISK_ANSATZ
};

/// maximum number of flattened members a descriptor can hold; a composite
/// with more members fails to build (falls back to the CPU path)
enum { GPU_POT_DESC_MAX_TERMS = 16 };

/** One concrete potential member: type tag + constructor parameters.
    The meaning of p[] per tag is defined by the class's gpuTermParams()
    exporter and consumed by gpu_term_phi_acc() below -- these two live in
    different files, so any layout change must touch both:
      PLUMMER / ISOCHRONE / NFW : p = { mass, scaleRadius }
      MIYAMOTONAGAI             : p = { mass, scaleRadius, scaleHeight }
      LOGARITHMIC               : p = { v0squared, coreRadius2, p2, q2, lengthUnit2 }
      HARMONIC                  : p = { Omega2, p2, q2 }
      DEHNEN (spherical only)   : p = { mass, scalerad, gamma }
      DISK_ANSATZ (recognized radial/vertical functors only):
          aux = (radialType | verticalType << 4), a DiskRadialType/DiskVerticalType
                pair (see potential_disk.h), NOT a double -- kept as a separate
                int field rather than packed into p[] since it selects which
                closed-form leaf to call, not a continuous parameter;
          p = { surfaceDensity, invScaleRadius, innerCutoffRadius,
                modulationAmplitude, invSersicIndex, invScaleHeight }

    A term may additionally carry a modifier transform (see GpuPotXform in
    potential_composite.h): when hasXform is nonzero, the leaf is evaluated at
    xf-mapped coordinates and its outputs are rescaled on the way back out.
    That is how Shifted / Tilted / Rotating / Scaled reach the device -- folded
    into the term rather than run as separate pre/post kernels.               */
template<typename T>
struct GpuPotTerm {
    int tag;   ///< a GpuPotTag value
    int aux;   ///< auxiliary integer sub-selector; only DISK_ANSATZ uses this (see above); 0 otherwise
    T p[6];    ///< constructor parameters, layout per tag (unused slots zero)
    int hasXform;        ///< nonzero if `xf` must be applied (a modifier chain wraps this term)
    GpuPotXform<T> xf;   ///< the collapsed modifier chain; undefined when hasXform == 0
};

/** Flattened by-value snapshot of a (possibly Composite) potential. */
template<typename T>
struct GpuPotDesc {
    int nterms;
    GpuPotTerm<T> terms[GPU_POT_DESC_MAX_TERMS];
};

/** A descriptor travels into every kernel as a by-value lambda capture, i.e. in
    the CUDA kernel-argument space. That space is 4 KiB on pre-sm_70 hardware and
    32 KiB from sm_70 on; our floor is sm_75, but staying under the 4 KiB figure
    costs nothing today (a double descriptor is ~2.8 KiB) and keeps the option of
    an older target open. Adding fields to GpuPotTerm, or raising
    GPU_POT_DESC_MAX_TERMS, is what would break this -- so it fails at compile
    time here rather than as a launch error at runtime. If a future term genuinely
    needs the room, move the payload behind a device pointer instead of widening
    the by-value struct. */
static_assert(sizeof(GpuPotDesc<double>) <= 4096,
    "GpuPotDesc<double> no longer fits the 4 KiB CUDA kernel-argument budget; "
    "see the comment above before raising this limit");

/** STORE (not accumulate) Phi (optional) and the Cartesian acceleration
    a = -grad Phi of a single term's underlying concrete potential, ignoring any
    modifier transform the term carries -- the coordinates passed in are already
    in the leaf's own frame. The per-tag bodies are the same leaf-plus-glue
    sequences as the corresponding evalmanyPhiAccCarT lambdas in
    potential_analytic.h.

    Split out from gpu_term_phi_acc() below so that the modified and unmodified
    paths share one copy of the per-tag switch. */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_term_leaf_phi_acc(const GpuPotTerm<T>& t,
    T x, T y, T z, /*stored, nullable*/ T* phi, /*stored*/ T a[3])
{
    T pot = 0;
    switch(t.tag) {
    case GPU_POT_PLUMMER: {
        const T r = std::sqrt(x*x + y*y + z*z);
        T dPhidr;
        plummer_eval(t.p[0], t.p[1], r, phi ? &pot : (T*)NULL, &dPhidr, (T*)NULL);
        sph_acc_car(dPhidr, x, y, z, r, a);
        break;
    }
    case GPU_POT_ISOCHRONE: {
        const T r = std::sqrt(x*x + y*y + z*z);
        T dPhidr;
        isochrone_eval(t.p[0], t.p[1], r, phi ? &pot : (T*)NULL, &dPhidr, (T*)NULL);
        sph_acc_car(dPhidr, x, y, z, r, a);
        break;
    }
    case GPU_POT_NFW: {
        const T r = std::sqrt(x*x + y*y + z*z);
        T dPhidr;
        nfw_eval(t.p[0], t.p[1], r, phi ? &pot : (T*)NULL, &dPhidr, (T*)NULL);
        sph_acc_car(dPhidr, x, y, z, r, a);
        break;
    }
    case GPU_POT_MIYAMOTONAGAI: {
        const T R = std::sqrt(x*x + y*y);
        T dPhidR, dPhidz;
        miyamoto_nagai_eval(t.p[0], t.p[1], t.p[2], R, z,
            phi ? &pot : (T*)NULL, &dPhidR, &dPhidz, (T*)NULL, (T*)NULL, (T*)NULL);
        cyl_acc_car(dPhidR, dPhidz, x, y, R, a);
        break;
    }
    case GPU_POT_LOGARITHMIC: {
        T g[3];
        logarithmic_eval(t.p[0], t.p[1], t.p[2], t.p[3], t.p[4], x, y, z,
            phi ? &pot : (T*)NULL, g, (T*)NULL);
        a[0] = -g[0];  a[1] = -g[1];  a[2] = -g[2];
        break;
    }
    case GPU_POT_HARMONIC: {
        T g[3];
        harmonic_eval(t.p[0], t.p[1], t.p[2], x, y, z,
            phi ? &pot : (T*)NULL, g, (T*)NULL);
        a[0] = -g[0];  a[1] = -g[1];  a[2] = -g[2];
        break;
    }
    case GPU_POT_DEHNEN: {   // spherical only, guaranteed by buildGpuPotDesc
        const T r = std::sqrt(x*x + y*y + z*z);
        T dPhidr;
        dehnen_eval(t.p[0], t.p[1], t.p[2], r, phi ? &pot : (T*)NULL, &dPhidr, (T*)NULL);
        sph_acc_car(dPhidr, x, y, z, r, a);
        break;
    }
    case GPU_POT_DISK_ANSATZ: {   // recognized functor pair only, guaranteed by buildGpuPotDesc
        const T R = std::sqrt(x*x + y*y);
        const int radialType = t.aux & 0xF, verticalType = (t.aux >> 4) & 0xF;
        DiskRadialParams<T>   rp = { t.p[0], t.p[1], t.p[2], t.p[3], t.p[4] };
        DiskVerticalParams<T> vp = { t.p[5] };
        T dR, dz;
        disk_ansatz_eval(radialType, rp, verticalType, vp, R, z,
            phi ? &pot : (T*)NULL, &dR, &dz, (T*)NULL, (T*)NULL, (T*)NULL);
        cyl_acc_car(dR, dz, x, y, R, a);
        break;
    }
    default:  // unreachable if the descriptor was built by buildGpuPotDesc
        a[0] = a[1] = a[2] = 0;
        break;
    }
    if(phi) *phi = pot;
}

/** Accumulate Phi (optional) and the Cartesian acceleration a = -grad Phi of a
    single term into *phi / acc[3], applying the term's modifier transform if it
    has one.

    Unmodified terms (hasXform == 0) take exactly the same arithmetic path, in
    exactly the same order, as before modifiers existed -- the transform costs
    them one branch and nothing else, so their results stay bit-for-bit. */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_term_phi_acc(const GpuPotTerm<T>& t,
    T x, T y, T z, /*accumulated, nullable*/ T* phi, /*accumulated*/ T acc[3])
{
    T a[3];
    if(t.hasXform) {
        // Reproduces Scaled::evalCar's "amplitude is zero, skip evaluating the
        // wrapped potential" shortcut: the contribution is identically zero, and
        // short-circuiting also keeps a non-finite leaf value (e.g. a divergent
        // cusp) from turning into 0*inf = NaN.
        if(t.xf.kphi == 0)
            return;
        T q[3];
        gpu_xform_pos(t.xf, x, y, z, q);
        T pot = 0, ai[3];
        gpu_term_leaf_phi_acc(t, q[0], q[1], q[2], phi ? &pot : (T*)NULL, ai);
        gpu_xform_vec(t.xf, ai, a);
        if(phi) *phi += t.xf.kphi * pot;
    } else {
        T pot = 0;
        gpu_term_leaf_phi_acc(t, x, y, z, phi ? &pot : (T*)NULL, a);
        if(phi) *phi += pot;
    }
    acc[0] += a[0];
    acc[1] += a[1];
    acc[2] += a[2];
}

/** STORE the mass density of a single term's underlying concrete potential,
    ignoring any modifier transform (coordinates already in the leaf's frame).
    Mirrors the evalmanyDensCarT lambdas: closed-form rho where the class has an
    explicit density override, the Laplacian route where it does not. */
template<typename T>
AGAMA_DEVICE_INLINE T gpu_term_leaf_rho(const GpuPotTerm<T>& t, T x, T y, T z)
{
    switch(t.tag) {
    case GPU_POT_PLUMMER:
        return plummer_rho(t.p[0], t.p[1], std::sqrt(x*x + y*y + z*z));
    case GPU_POT_ISOCHRONE:
        return isochrone_rho(t.p[0], t.p[1], std::sqrt(x*x + y*y + z*z));
    case GPU_POT_NFW:
        return nfw_rho(t.p[0], t.p[1], std::sqrt(x*x + y*y + z*z));
    case GPU_POT_MIYAMOTONAGAI:
        return miyamoto_nagai_rho(t.p[0], t.p[1], t.p[2], std::sqrt(x*x + y*y), z);
    case GPU_POT_LOGARITHMIC:
        return logarithmic_rho(t.p[0], t.p[1], t.p[2], t.p[3], t.p[4], x, y, z);
    case GPU_POT_HARMONIC:
        return harmonic_rho(t.p[0], t.p[1], t.p[2], x, y, z);
    case GPU_POT_DEHNEN:
        // spherical only (buildGpuPotDesc rejects the triaxial case), so both
        // axis ratios are exactly 1 and are not stored in p[]
        return dehnen_rho(t.p[0], t.p[1], t.p[2], T(1), T(1), x, y, z);
    case GPU_POT_DISK_ANSATZ: {
        const int radialType = t.aux & 0xF, verticalType = (t.aux >> 4) & 0xF;
        DiskRadialParams<T>   rp = { t.p[0], t.p[1], t.p[2], t.p[3], t.p[4] };
        DiskVerticalParams<T> vp = { t.p[5] };
        return disk_ansatz_rho(radialType, rp, verticalType, vp, std::sqrt(x*x + y*y), z);
    }
    default:  // unreachable if the descriptor was built by buildGpuPotDesc
        return 0;
    }
}

/** Accumulate the mass density of a single term into *rho, applying the term's
    modifier transform if it has one. A similarity transform multiplies the
    Laplacian by sc^2 on top of the potential's own kphi (see GpuPotXform). */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_term_dens(const GpuPotTerm<T>& t, T x, T y, T z, /*accumulated*/ T* rho)
{
    if(t.hasXform) {
        // Short-circuit at zero amplitude, as the potential path does. Note that
        // Scaled::densityCar on the CPU does NOT short-circuit -- it evaluates
        // the wrapped density and multiplies by zero -- so the two differ only
        // where the wrapped density is non-finite (a central cusp sampled
        // exactly at r=0), giving 0 here versus NaN there. Both are defensible
        // for a component that has been scaled out of existence; this one does
        // not let one degenerate point poison the batch.
        if(t.xf.kphi == 0)
            return;
        T q[3];
        gpu_xform_pos(t.xf, x, y, z, q);
        *rho += t.xf.kphi * t.xf.sc * t.xf.sc * gpu_term_leaf_rho(t, q[0], q[1], q[2]);
    } else
        *rho += gpu_term_leaf_rho(t, x, y, z);
}

/** Evaluate Phi (optional; pass NULL to skip) and the Cartesian acceleration
    of the whole descriptor at one point: zero-initializes the outputs, then
    accumulates the terms in member order (matching the summation order of the
    Composite batch dispatch in potential_gpu.cpp). */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_desc_phi_acc(const GpuPotDesc<T>& d,
    T x, T y, T z, /*nullable*/ T* phi, T acc[3])
{
    if(phi) *phi = 0;
    acc[0] = acc[1] = acc[2] = 0;
    for(int c = 0; c < d.nterms; c++)
        gpu_term_phi_acc(d.terms[c], x, y, z, phi, acc);
}

/** Evaluate the mass density of the whole descriptor at one point: the sum of
    the members' densities, in member order (a composite's density is the sum of
    its components', same as its potential). */
template<typename T>
AGAMA_DEVICE_INLINE T gpu_desc_dens(const GpuPotDesc<T>& d, T x, T y, T z)
{
    T rho = 0;
    for(int c = 0; c < d.nterms; c++)
        gpu_term_dens(d.terms[c], x, y, z, &rho);
    return rho;
}

namespace detail {

/** Recursive worker behind buildGpuPotDesc(): appends one or more terms for
    `pot`, each carrying the modifier transform `xf` accumulated by the wrappers
    already descended through (`hasXform` false means `xf` is still the
    identity and should not be stored). See buildGpuPotDesc() for the contract. */
inline bool gpuDescAddTerms(const BasePotential& pot, GpuPotDesc<double>& desc,
    double time, bool requireTimeIndependent, GpuPotXform<double> xf, bool hasXform)
{
    // --- modifier wrappers: fold this stage into xf and descend into the wrapped
    // object. Because the transform rides on the term rather than on a separate
    // kernel, nesting depth is free: N wrappers still collapse to 13 numbers.
    #define AGAMA_GPU_MOD_UNWRAP(ModClass) \
        if(const ModClass<BasePotential>* m =                                          \
            dynamic_cast<const ModClass<BasePotential>*>(&pot))                        \
        {                                                                              \
            /* A caller that re-evaluates the descriptor per time step (batch eval,  */\
            /* one shared `time`) can take any modifier. A caller that builds the    */\
            /* descriptor once and then integrates across many times (the orbit      */\
            /* kernel) must refuse a time-varying stage rather than freeze it.       */\
            if(requireTimeIndependent && !m->gpuXformConstant())                       \
                return false;                                                          \
            double A[9], b[3], k, s;                                                   \
            m->gpuXformStage(time, A, b, k, s);                                         \
            gpuXformComposeInner(xf, A, b, k, s);                                       \
            return gpuDescAddTerms(*m->component(0), desc, time,                        \
                requireTimeIndependent, xf, true);                                      \
        }
    AGAMA_GPU_MOD_UNWRAP(Shifted)
    AGAMA_GPU_MOD_UNWRAP(Tilted)
    AGAMA_GPU_MOD_UNWRAP(Rotating)
    AGAMA_GPU_MOD_UNWRAP(Scaled)
    #undef AGAMA_GPU_MOD_UNWRAP

    // A modifier wrapping a Composite distributes over its members: each member
    // inherits the same accumulated transform, which is exactly right because
    // Phi is linear in the members.
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!gpuDescAddTerms(*comp->component(c), desc, time,
                requireTimeIndependent, xf, hasXform))
                return false;
        return true;
    }
    if(desc.nterms >= GPU_POT_DESC_MAX_TERMS)
        return false;
    GpuPotTerm<double>& term = desc.terms[desc.nterms];
    term.aux = 0;
    for(int k = 0; k < 6; k++)
        term.p[k] = 0;
    term.hasXform = hasXform ? 1 : 0;
    term.xf = xf;
    if(hasXform) {
        // Fail closed on a degenerate transform (scale(t) == 0 gives an infinite
        // s; a NaN anywhere in the chain would otherwise silently poison every
        // point in the batch) -- the CPU path is still available and correct.
        for(int k = 0; k < 9; k++)
            if(!isFinite(xf.mat[k]))
                return false;
        for(int k = 0; k < 3; k++)
            if(!isFinite(xf.off[k]))
                return false;
        if(!isFinite(xf.kphi) || !isFinite(xf.sc))
            return false;
    }
    if(const Plummer* p = dynamic_cast<const Plummer*>(&pot))
        { term.tag = GPU_POT_PLUMMER;        p->gpuTermParams(term.p); }
    else if(const Isochrone* p = dynamic_cast<const Isochrone*>(&pot))
        { term.tag = GPU_POT_ISOCHRONE;      p->gpuTermParams(term.p); }
    else if(const NFW* p = dynamic_cast<const NFW*>(&pot))
        { term.tag = GPU_POT_NFW;            p->gpuTermParams(term.p); }
    else if(const MiyamotoNagai* p = dynamic_cast<const MiyamotoNagai*>(&pot))
        { term.tag = GPU_POT_MIYAMOTONAGAI;  p->gpuTermParams(term.p); }
    else if(const Logarithmic* p = dynamic_cast<const Logarithmic*>(&pot))
        { term.tag = GPU_POT_LOGARITHMIC;    p->gpuTermParams(term.p); }
    else if(const Harmonic* p = dynamic_cast<const Harmonic*>(&pot))
        { term.tag = GPU_POT_HARMONIC;       p->gpuTermParams(term.p); }
    else if(const Dehnen* p = dynamic_cast<const Dehnen*>(&pot)) {
        // triaxial Dehnen needs math::integrate (no device path); fail cleanly
        // instead of silently building a descriptor for the wrong potential
        if(!isSpherical(p->symmetry()))
            return false;
        term.tag = GPU_POT_DEHNEN;
        p->gpuTermParams(term.p);
    }
    else if(const DiskAnsatz* p = dynamic_cast<const DiskAnsatz*>(&pot)) {
        // arbitrary-user-function DiskAnsatz (second constructor) needs no
        // device path; fail cleanly instead of building a wrong descriptor
        DiskAnsatzDesc<double> d;
        if(!p->gpuDesc(d))
            return false;
        term.tag = GPU_POT_DISK_ANSATZ;
        term.aux = (d.radialType & 0xF) | ((d.verticalType & 0xF) << 4);
        term.p[0] = d.radial.surfaceDensity;
        term.p[1] = d.radial.invScaleRadius;
        term.p[2] = d.radial.innerCutoffRadius;
        term.p[3] = d.radial.modulationAmplitude;
        term.p[4] = d.radial.invSersicIndex;
        term.p[5] = d.vertical.invScaleHeight;
    }
    else
        return false;
    desc.nterms++;
    return true;
}

}  // namespace detail

/** Host-side builder: flatten `pot` (recursing into Composite, and through any
    Shifted / Tilted / Rotating / Scaled modifier wrappers) into `desc`.

    \param[in]  time  the moment at which time-dependent modifier stages are
    evaluated and baked into the terms. The resulting descriptor is therefore
    valid AT THAT TIME ONLY unless requireTimeIndependent was set.
    \param[in]  requireTimeIndependent  if true, refuse any modifier stage whose
    transform varies with time, so that the returned descriptor is valid at every
    time. Callers that build once and evaluate at many times -- the orbit kernel,
    which sees a different t at each RK stage -- MUST set this; callers with a
    single shared `time` for the whole batch must not, since for them baking the
    stage in at `time` is exact.
    \return true on success; false if any member is not a representable type, a
    modifier stage was rejected per requireTimeIndependent, a transform came out
    non-finite, or the flattened member count exceeds GPU_POT_DESC_MAX_TERMS --
    in which case the caller should fall back to the CPU path (desc is left
    partially filled and must not be used). Use
    potential::unsupportedGPUPotentialName() for the user-facing error message. */
inline bool buildGpuPotDesc(const BasePotential& pot, GpuPotDesc<double>& desc,
    double time = 0, bool requireTimeIndependent = false)
{
    desc.nterms = 0;
    GpuPotXform<double> xf;
    gpu_xform_identity(xf);
    return detail::gpuDescAddTerms(pot, desc, time, requireTimeIndependent, xf, false);
}

/** Convert a double-precision descriptor to the kernel's working precision
    (identity for T=double; a one-time host-side cast for T=float). */
template<typename T>
inline GpuPotDesc<T> castGpuPotDesc(const GpuPotDesc<double>& d)
{
    GpuPotDesc<T> out;
    out.nterms = d.nterms;
    for(int c = 0; c < d.nterms; c++) {
        out.terms[c].tag = d.terms[c].tag;
        out.terms[c].aux = d.terms[c].aux;
        for(int k = 0; k < 6; k++)
            out.terms[c].p[k] = static_cast<T>(d.terms[c].p[k]);
        out.terms[c].hasXform = d.terms[c].hasXform;
        for(int k = 0; k < 9; k++)
            out.terms[c].xf.mat[k] = static_cast<T>(d.terms[c].xf.mat[k]);
        for(int k = 0; k < 3; k++)
            out.terms[c].xf.off[k] = static_cast<T>(d.terms[c].xf.off[k]);
        out.terms[c].xf.kphi = static_cast<T>(d.terms[c].xf.kphi);
        out.terms[c].xf.sc   = static_cast<T>(d.terms[c].xf.sc);
    }
    return out;
}

}  // namespace potential
