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
    numbers). Since the transform depends on time, such a collapsed descriptor is
    valid only at the `time` it was built for -- which is exactly right for batch
    evaluation, where one `time` is shared by the whole batch.

    A caller that reuses ONE descriptor across many times (the orbit kernel, a
    different t at every RK stage) instead passes a spline buffer to
    buildGpuPotDesc(), and gets back GpuModStage entries that the kernel
    recomposes per step via gpu_term_xform(). Constant parts of the chain are
    still folded on the host either way.
*/
#pragma once
#include "potential_base.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_dehnen.h"
#include "potential_disk.h"
#include "gpu_device.h"
#include <cmath>
#include <vector>

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

/// maximum number of time-varying modifier stages a descriptor can hold, summed
/// over all its terms. The Python factory applies each of the four modifiers at
/// most once, so a factory-built chain needs at most 4 (and fewer once constant
/// runs are merged); the headroom covers modifiers nested through composites.
/// A potential needing more fails to build and falls back to the CPU path.
enum { GPU_MOD_MAX_STAGES = 8 };

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
    /** A TIME-VARYING modifier chain cannot be collapsed once, so instead of `xf`
        the term names a run of stages in GpuPotDesc::stages that the kernel
        recomposes at each time it is asked for. stageCount == 0 means "use xf
        (or nothing)"; the two are mutually exclusive. */
    int stageBegin;      ///< index of this term's first stage in GpuPotDesc::stages
    int stageCount;      ///< number of stages; 0 => time-independent, use hasXform/xf
};

/** Flattened by-value snapshot of a (possibly Composite) potential. */
template<typename T>
struct GpuPotDesc {
    int nterms;
    GpuPotTerm<T> terms[GPU_POT_DESC_MAX_TERMS];
    /** Stages for the terms whose modifier chain varies with time (see
        GpuPotTerm::stageBegin). Held here rather than inside each term because
        most descriptors have none, and inlining a worst-case chain into all 16
        terms would multiply the by-value footprint for nothing. */
    int nstages;
    GpuModStage<T> stages[GPU_MOD_MAX_STAGES];
    /** Flat buffer holding the node arrays of every spline referenced by
        `stages`, in whichever memory space the kernel that reads it will run in
        -- host memory for the Serial/OpenMP policies, device memory for Cuda.
        buildGpuPotDesc() leaves this NULL and hands the host-side buffer back to
        the caller, because only the caller knows where the kernel will run and
        therefore where the data has to end up. NULL whenever nstages == 0. */
    const T* splineData;
};

/** A descriptor travels into every kernel as a by-value lambda capture, i.e. in
    the CUDA kernel-argument space. That space is 4 KiB on pre-sm_70 hardware and
    32 KiB from sm_70 on, and our floor is sm_75. The budget asserted below is
    8 KiB: a quarter of what every supported target provides, which leaves room
    for the modifier stages (a double descriptor is 4312 bytes with them) while
    still failing loudly long before a launch error. Adding fields to GpuPotTerm,
    or raising GPU_POT_DESC_MAX_TERMS / GPU_MOD_MAX_STAGES, is what would break
    it. If a future term genuinely needs more room -- Multipole's coefficient
    tables are ~1 MB, for instance -- put the payload behind a device pointer, as
    `splineData` does, instead of widening the by-value struct. */
static_assert(sizeof(GpuPotDesc<double>) <= 8192,
    "GpuPotDesc<double> no longer fits its CUDA kernel-argument budget; "
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
    /*nullable: NULL means no modifier chain*/ const GpuPotXform<T>* xfp,
    T x, T y, T z, /*accumulated, nullable*/ T* phi, /*accumulated*/ T acc[3])
{
    T a[3];
    if(xfp) {
        const GpuPotXform<T>& xf = *xfp;
        // Reproduces Scaled::evalCar's "amplitude is zero, skip evaluating the
        // wrapped potential" shortcut: the contribution is identically zero, and
        // short-circuiting also keeps a non-finite leaf value (e.g. a divergent
        // cusp) from turning into 0*inf = NaN.
        if(xf.kphi == 0)
            return;
        T q[3];
        gpu_xform_pos(xf, x, y, z, q);
        T pot = 0, ai[3];
        gpu_term_leaf_phi_acc(t, q[0], q[1], q[2], phi ? &pot : (T*)NULL, ai);
        gpu_xform_vec(xf, ai, a);
        if(phi) *phi += xf.kphi * pot;
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
AGAMA_DEVICE_INLINE void gpu_term_dens(const GpuPotTerm<T>& t,
    /*nullable: NULL means no modifier chain*/ const GpuPotXform<T>* xfp,
    T x, T y, T z, /*accumulated*/ T* rho)
{
    if(xfp) {
        const GpuPotXform<T>& xf = *xfp;
        // Short-circuit at zero amplitude, as the potential path does. Note that
        // Scaled::densityCar on the CPU does NOT short-circuit -- it evaluates
        // the wrapped density and multiplies by zero -- so the two differ only
        // where the wrapped density is non-finite (a central cusp sampled
        // exactly at r=0), giving 0 here versus NaN there. Both are defensible
        // for a component that has been scaled out of existence; this one does
        // not let one degenerate point poison the batch.
        if(xf.kphi == 0)
            return;
        T q[3];
        gpu_xform_pos(xf, x, y, z, q);
        *rho += xf.kphi * xf.sc * xf.sc * gpu_term_leaf_rho(t, q[0], q[1], q[2]);
    } else
        *rho += gpu_term_leaf_rho(t, x, y, z);
}

/** Evaluate Phi (optional; pass NULL to skip) and the Cartesian acceleration
    of the whole descriptor at one point: zero-initializes the outputs, then
    accumulates the terms in member order (matching the summation order of the
    Composite batch dispatch in potential_gpu.cpp). */
/** Resolve one term's modifier transform at time `t`.

    Three cases, in increasing cost, and the cheap ones stay cheap:
      - no modifiers            -> returns NULL, and the term takes exactly the
                                   arithmetic path it took before modifiers existed;
      - constant chain          -> returns the transform baked at build time;
      - time-varying chain      -> recomposed here from the term's stages, which
                                   costs a handful of spline lookups.
    `scratch` receives the recomposed transform in the last case; the returned
    pointer aliases it, so it must outlive the use of the return value. */
template<typename T>
AGAMA_DEVICE_INLINE const GpuPotXform<T>* gpu_term_xform(const GpuPotDesc<T>& d,
    const GpuPotTerm<T>& t, T time, /*scratch*/ GpuPotXform<T>& scratch)
{
    if(t.stageCount > 0) {
        gpu_stages_xform(d.stages + t.stageBegin, t.stageCount, d.splineData, time, scratch);
        return &scratch;
    }
    return t.hasXform ? &t.xf : (const GpuPotXform<T>*)NULL;
}

template<typename T>
AGAMA_DEVICE_INLINE void gpu_desc_phi_acc(const GpuPotDesc<T>& d,
    T x, T y, T z, /*nullable*/ T* phi, T acc[3], T time = 0)
{
    if(phi) *phi = 0;
    acc[0] = acc[1] = acc[2] = 0;
    // one scratch reused across terms: this runs inside the orbit kernel next to
    // DOP853's 120-element state/scratch arrays, where every extra live value
    // competes for registers and therefore for occupancy
    GpuPotXform<T> scratch;
    for(int c = 0; c < d.nterms; c++)
        gpu_term_phi_acc(d.terms[c], gpu_term_xform(d, d.terms[c], time, scratch),
            x, y, z, phi, acc);
}

/** Evaluate the mass density of the whole descriptor at one point: the sum of
    the members' densities, in member order (a composite's density is the sum of
    its components', same as its potential). */
template<typename T>
AGAMA_DEVICE_INLINE T gpu_desc_dens(const GpuPotDesc<T>& d, T x, T y, T z, T time = 0)
{
    T rho = 0;
    GpuPotXform<T> scratch;
    for(int c = 0; c < d.nterms; c++)
        gpu_term_dens(d.terms[c], gpu_term_xform(d, d.terms[c], time, scratch),
            x, y, z, &rho);
    return rho;
}

namespace detail {

/** Flush an accumulated constant run into a GPU_MOD_CONST stage, so that a
    time-varying stage emitted after it composes in the right order.

    Needed because composition does not commute: once any stage of a chain is
    time-varying, an OUTER constant run can no longer be folded into a single
    baked transform and applied separately -- it has to keep its position in the
    sequence. If there is no accumulated run (hasXform false) this is a no-op.
    \return false if the stage array is full. */
inline bool gpuDescFlushConstStage(GpuPotDesc<double>& desc,
    GpuPotXform<double>& xf, bool& hasXform, int& stageBegin, int& stageCount)
{
    if(!hasXform)
        return true;
    if(desc.nstages >= GPU_MOD_MAX_STAGES)
        return false;
    desc.stages[desc.nstages].kind = GPU_MOD_CONST;
    desc.stages[desc.nstages].xf   = xf;
    if(stageCount == 0)
        stageBegin = desc.nstages;
    desc.nstages++;
    stageCount++;
    // the run has been consumed; start accumulating a fresh one
    gpu_xform_identity(xf);
    hasXform = false;
    return true;
}

/** Recursive worker behind buildGpuPotDesc(): appends one or more terms for
    `pot`, each carrying the modifier transform `xf` accumulated by the wrappers
    already descended through (`hasXform` false means `xf` is still the
    identity and should not be stored). See buildGpuPotDesc() for the contract. */
inline bool gpuDescAddTerms(const BasePotential& pot, GpuPotDesc<double>& desc,
    double time, std::vector<double>* splineData,
    GpuPotXform<double> xf, bool hasXform, int stageBegin, int stageCount)
{
    // --- modifier wrappers: fold this stage into xf and descend into the wrapped
    // object. Because the transform rides on the term rather than on a separate
    // kernel, nesting depth is free: N wrappers still collapse to 13 numbers.
    // Two ways to absorb a modifier stage, chosen by whether the caller can
    // re-evaluate the descriptor at other times:
    //
    //  splineData == NULL (batch eval, one `time` shared by the whole batch):
    //      fold every stage into the accumulated `xf` at that time. Exact, not an
    //      approximation, because the answer is only ever wanted at this one time.
    //
    //  splineData != NULL (the orbit kernel, which sees a different t at every RK
    //      stage): fold CONSTANT stages as above -- they can never go stale -- but
    //      emit a GpuModStage for a time-varying one so the kernel recomposes it
    //      per step. A constant run that precedes a varying stage cannot simply be
    //      folded into `xf` and forgotten, because composition does not commute;
    //      it is flushed into a GPU_MOD_CONST stage so ordering is preserved.
    // Tilted is always time-independent (Euler angles fixed at construction), so it
    // is only ever folded -- it has no gpuEmitStage() and needs none.
    if(const Tilted<BasePotential>* m = dynamic_cast<const Tilted<BasePotential>*>(&pot)) {
        double A[9], b[3], k, s;
        m->gpuXformStage(time, A, b, k, s);
        gpuXformComposeInner(xf, A, b, k, s);
        return gpuDescAddTerms(*m->component(0), desc, time, splineData,
            xf, true, stageBegin, stageCount);
    }
    #define AGAMA_GPU_MOD_UNWRAP(ModClass) \
        if(const ModClass<BasePotential>* m =                                          \
            dynamic_cast<const ModClass<BasePotential>*>(&pot))                        \
        {                                                                              \
            if(!splineData || m->gpuXformConstant()) {                                  \
                double A[9], b[3], k, s;                                               \
                m->gpuXformStage(time, A, b, k, s);                                     \
                gpuXformComposeInner(xf, A, b, k, s);                                   \
                return gpuDescAddTerms(*m->component(0), desc, time, splineData,        \
                    xf, true, stageBegin, stageCount);                                  \
            }                                                                          \
            if(!gpuDescFlushConstStage(desc, xf, hasXform, stageBegin, stageCount))     \
                return false;                                                          \
            if(desc.nstages >= GPU_MOD_MAX_STAGES)                                     \
                return false;                                                          \
            if(!m->gpuEmitStage(desc.stages[desc.nstages], *splineData))                \
                return false;                                                          \
            if(stageCount == 0)                                                        \
                stageBegin = desc.nstages;                                             \
            desc.nstages++;                                                            \
            stageCount++;                                                              \
            return gpuDescAddTerms(*m->component(0), desc, time, splineData,            \
                xf, false, stageBegin, stageCount);                                     \
        }
    AGAMA_GPU_MOD_UNWRAP(Shifted)
    AGAMA_GPU_MOD_UNWRAP(Rotating)
    AGAMA_GPU_MOD_UNWRAP(Scaled)
    #undef AGAMA_GPU_MOD_UNWRAP

    // A modifier wrapping a Composite distributes over its members: each member
    // inherits the same accumulated transform, which is exactly right because
    // Phi is linear in the members.
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!gpuDescAddTerms(*comp->component(c), desc, time, splineData,
                xf, hasXform, stageBegin, stageCount))
                return false;
        return true;
    }
    if(desc.nterms >= GPU_POT_DESC_MAX_TERMS)
        return false;
    GpuPotTerm<double>& term = desc.terms[desc.nterms];
    term.aux = 0;
    for(int k = 0; k < 6; k++)
        term.p[k] = 0;
    term.stageBegin = stageBegin;
    term.stageCount = stageCount;
    if(stageCount > 0) {
        // A trailing constant run after the last varying stage still has to be
        // applied, and it is INNERMOST, so it must follow them as its own stage.
        if(hasXform) {
            if(desc.nstages >= GPU_MOD_MAX_STAGES)
                return false;
            desc.stages[desc.nstages].kind = GPU_MOD_CONST;
            desc.stages[desc.nstages].xf   = xf;
            desc.nstages++;
            term.stageCount++;
        }
        term.hasXform = 0;
        gpu_xform_identity(term.xf);
    } else {
        term.hasXform = hasXform ? 1 : 0;
        term.xf = xf;
    }
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
    folded into the terms. When `splineData` is NULL the resulting descriptor is
    valid AT THAT TIME ONLY; when it is given, only the constant parts are folded
    at `time` (which is immaterial for them) and the rest stays re-evaluable.
    \param[in,out]  splineData  when NULL (the default, and what batch evaluation
    wants), every modifier stage is folded at `time`. When non-NULL, time-varying
    stages are instead emitted into desc.stages and their spline node arrays are
    appended to this vector; the CALLER must then place that data where the kernel
    can read it (host memory for Serial/OpenMP, device memory for Cuda) and set
    desc.splineData to point at it -- see castGpuSplineData() and the orbit path
    in orbit_gpu.cpp. desc.splineData is deliberately left NULL here, because only
    the caller knows which memory space the kernel will run in.
    \return true on success; false if any member is not a representable type, a
    transform came out non-finite, a modifier spline could not be packed, or the
    member/stage count exceeds GPU_POT_DESC_MAX_TERMS / GPU_MOD_MAX_STAGES -- in
    which case the caller should fall back to the CPU path (desc is left partially
    filled and must not be used). Use potential::unsupportedGPUPotentialName() for
    the user-facing error message. */
inline bool buildGpuPotDesc(const BasePotential& pot, GpuPotDesc<double>& desc,
    double time = 0, std::vector<double>* splineData = NULL)
{
    desc.nterms  = 0;
    desc.nstages = 0;
    desc.splineData = NULL;
    if(splineData)
        splineData->clear();
    GpuPotXform<double> xf;
    gpu_xform_identity(xf);
    return detail::gpuDescAddTerms(pot, desc, time, splineData, xf, false, 0, 0);
}

/// element-wise cast of one transform; shared by the term and stage loops below
template<typename T>
inline void castGpuPotXform(const GpuPotXform<double>& in, /*out*/ GpuPotXform<T>& out)
{
    for(int k = 0; k < 9; k++)
        out.mat[k] = static_cast<T>(in.mat[k]);
    for(int k = 0; k < 3; k++)
        out.off[k] = static_cast<T>(in.off[k]);
    out.kphi = static_cast<T>(in.kphi);
    out.sc   = static_cast<T>(in.sc);
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
        out.terms[c].hasXform   = d.terms[c].hasXform;
        out.terms[c].stageBegin = d.terms[c].stageBegin;
        out.terms[c].stageCount = d.terms[c].stageCount;
        castGpuPotXform<T>(d.terms[c].xf, out.terms[c].xf);
    }
    out.nstages = d.nstages;
    for(int i = 0; i < d.nstages; i++) {
        out.stages[i].kind = d.stages[i].kind;
        castGpuPotXform<T>(d.stages[i].xf, out.stages[i].xf);
        for(int k = 0; k < 3; k++) {
            out.stages[i].s[k].off  = d.stages[i].s[k].off;
            out.stages[i].s[k].size = d.stages[i].s[k].size;
            out.stages[i].s[k].c    = static_cast<T>(d.stages[i].s[k].c);
        }
    }
    // set by the caller once the spline data is in the right memory space
    out.splineData = NULL;
    return out;
}

/** Convert the flat spline buffer that buildGpuPotDesc() filled to the kernel's
    working precision. Identity for T=double; a one-time host-side cast otherwise.
    Coefficients are stored fp64 and cast on load, as everywhere else in the
    descriptor (hard constraint #4). */
template<typename T>
inline std::vector<T> castGpuSplineData(const std::vector<double>& src)
{
    return std::vector<T>(src.begin(), src.end());
}

}  // namespace potential
