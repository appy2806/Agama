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
#include "potential_multipole.h"   // MultipoleDeviceDesc + multipoleEvalDevice (Tier 2)
#include "math_core.h"             // math::atan2 (device-callable)
#include "gpu_device.h"
#include <cmath>
#include <vector>

namespace potential {

/// concrete potential types representable in a GpuPotTerm;
/// must match the GPU-capable set in potential_gpu.cpp (AGAMA_GPU_POT_LIST),
/// EXCEPT GPU_POT_DEHNEN, GPU_POT_DISK_ANSATZ and GPU_POT_MULTIPOLE: Dehnen is
/// only representable for the spherical case (axisRatioY==axisRatioZ==1),
/// DiskAnsatz only for the 5 recognized closed-form radial/vertical functor
/// types, and a Multipole only for the shapes buildMultipoleDeviceDesc() accepts
/// (order <= math::LEGENDRE_MMAX, shared knot vectors, PowerLaw asymptotes, ...)
/// -- all three are special-cased in buildGpuPotDesc() below (and in
/// potential_gpu.cpp outside its AGAMA_GPU_POT_LIST macro; see the comment there)
/// rather than folded into the blanket per-type capability of the X-macro list.
enum GpuPotTag {
    GPU_POT_PLUMMER,
    GPU_POT_ISOCHRONE,
    GPU_POT_NFW,
    GPU_POT_MIYAMOTONAGAI,
    GPU_POT_LOGARITHMIC,
    GPU_POT_HARMONIC,
    GPU_POT_DEHNEN,
    GPU_POT_DISK_ANSATZ,
    GPU_POT_MULTIPOLE
};

/// maximum number of flattened members a descriptor can hold; a composite
/// with more members fails to build (falls back to the CPU path)
enum { GPU_POT_DESC_MAX_TERMS = 16 };

/** Maximum number of Multipole members a descriptor can hold. Unlike the analytic
    tags, a Multipole term needs a ~144-byte side entry (MultipoleDeviceDesc plus a
    payload offset), so one slot per GpuPotTerm would add ~2.5 KB to a by-value
    struct already at 4.3 KB against an 8 KB budget -- see the static_assert below.
    Four covers every realistic composite (a GalPot-style MW is one Multipole per
    spheroidal component: bulge + halo, occasionally a third), and a potential with
    more FAILS TO BUILD and falls back to the CPU rather than being truncated. */
enum { GPU_POT_MAX_MULTIPOLE = 4 };

/** Compile-time order cap of the Multipole scratch a descriptor kernel reserves.
    math::LEGENDRE_MMAX (32) and nothing lower, for the reason recorded in
    potential_multipole.h: MultipoleInterp2d -- the branch every realistic GalPot MW
    potential takes -- needs only 3.7 KB/thread and 155 registers with zero spills
    at order 32, because its scratch scales with mmax rather than lmax^2, and real
    use cases do reach lmax=32. buildGpuPotDesc() rejects anything above this so
    that a kernel's fixed-size scratch array can never be overrun. */
enum { GPU_POT_MULTIPOLE_ORDER = math::LEGENDRE_MMAX };

/* NOTE ON THE ORBIT KERNELS. They are instantiated at ORDER=0 only, and
   prepareOrbitBatch (orbit_gpu.cpp) rejects any Multipole-bearing potential so that
   an orbit falls back to the CPU instead of running against absent scratch. That is
   a COMPILE-TIME limit, measured, not a correctness or throughput one -- the full
   reason, the numbers, and the two candidate fixes are documented at that rejection
   site. Batch potential/force/density evaluation is unaffected and uses the full
   order-32 cap above: those kernels cost ptxas 0.4-4 s each. */

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
      MULTIPOLE                 : aux = index into GpuPotDesc::mp, p unused.
                All of this term's data lives in that side entry plus the shared
                payload buffer GpuPotDesc::splineData -- the same arrangement
                GpuModStage uses for its spline nodes, and for the same reason
                (~1 MB of coefficients cannot travel by value in kernel arguments).

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

/** One Multipole member: its by-value POD descriptor plus where its coefficient
    blob starts inside the descriptor's shared payload buffer. Held in a small side
    array indexed by GpuPotTerm::aux rather than inside the term, exactly as
    GpuModStage entries are indexed by stageBegin/stageCount -- see
    GPU_POT_MAX_MULTIPOLE for the size argument. */
template<typename T>
struct GpuMultipoleTerm {
    MultipoleDeviceDesc<T> d;
    /** element (not byte) index of this Multipole's blob within
        GpuPotDesc::splineData; the descriptor's own offXval/offCoefs/... are
        relative to that, so the evaluator is handed `splineData + blobOffset`. */
    int blobOffset;
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
    /// Multipole members, indexed by GpuPotTerm::aux; nmp of them are valid
    int nmp;
    GpuMultipoleTerm<T> mp[GPU_POT_MAX_MULTIPOLE];
    /** Flat payload buffer, in whichever memory space the kernel that reads it will
        run in -- host memory for the Serial/OpenMP policies, device memory for Cuda.
        Holds the node arrays of every spline referenced by `stages` AND the
        coefficient blob of every Multipole referenced by `mp`, appended in build
        order. buildGpuPotDesc() leaves this NULL and hands the host-side buffer back
        to the caller, because only the caller knows where the kernel will run and
        therefore where the data has to end up. NULL whenever nstages == 0 and
        nmp == 0. */
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
    `splineData` does, instead of widening the by-value struct.

    That is exactly what GPU_POT_MULTIPOLE does: the ~1 MB of spline coefficients
    goes into the shared `splineData` payload and the term carries only an index
    into `mp`, whose entries are ~144 bytes each (a MultipoleDeviceDesc is all ints
    and scalars). Four of them cost ~600 bytes, taking a double descriptor from
    4,312 to 4,920 bytes -- still comfortably inside this budget. */
static_assert(sizeof(GpuPotDesc<double>) <= 8192,
    "GpuPotDesc<double> no longer fits its CUDA kernel-argument budget; "
    "see the comment above before raising this limit");

/** Per-thread scratch a descriptor kernel needs, sized at COMPILE TIME by the
    order cap so that a kernel whose descriptor contains no Multipole term pays
    nothing at all: the ORDER==0 specialization is an empty struct and its
    mpScratch() is a null pointer, so no local array is emitted and the
    already-shipped analytic descriptor kernels keep their measured STACK:0.
    A kernel that may see a Multipole instantiates ORDER = GPU_POT_MULTIPOLE_ORDER
    and pays MultipoleDeviceScratchMax<32>::value = 580 T (4,640 B in fp64,
    2,320 B in fp32) per thread -- which is why this is a template parameter and
    not an unconditional local. The host picks the instantiation from
    gpuDescNeedsMultipole(). */
template<typename T, int ORDER>
struct GpuDescScratch {
    T mp[MultipoleDeviceScratchMax<ORDER>::value];
    AGAMA_DEVICE_INLINE T* mpScratch() { return mp; }
};
template<typename T>
struct GpuDescScratch<T, 0> {
    AGAMA_DEVICE_INLINE T* mpScratch() { return (T*)NULL; }
};

/// true iff evaluating this descriptor requires the Multipole scratch block, i.e.
/// iff the kernel must be instantiated at ORDER = GPU_POT_MULTIPOLE_ORDER rather
/// than 0. Host-side selector; also the guard that keeps a Multipole-carrying
/// descriptor away from a kernel compiled without room for it.
template<typename T>
inline bool gpuDescNeedsMultipole(const GpuPotDesc<T>& d) { return d.nmp > 0; }

/** Cartesian acceleration of a Multipole term, and optionally its potential.

    THE dPhi/dphi TRAP. cyl_acc_car() in potential_analytic.h takes only
    (dPhi/dR, dPhi/dz) and silently drops dPhi/dphi, which is correct for every
    analytic term that uses it because they are all axisymmetric in their own frame.
    A Multipole with mmax > 0 -- i.e. every triaxial or otherwise non-axisymmetric
    model, which is the entire reason one uses a Multipole -- has a NONZERO
    dPhi/dphi, so that helper would produce quietly wrong forces for exactly the
    interesting cases. It is left untouched (it is on hot analytic paths) and this
    term does the full 3-component conversion instead.

    It does it by calling coord::toGrad<Cyl,Car> rather than by hand-rolling the
    same three lines, because that IS the function the CPU path uses:
    Multipole::eval(PosCar) resolves to BasePotentialCyl::evalCar ->
    coord::evalAndConvert<Cyl,Car>, whose two steps are toPosDeriv<Car,Cyl> and
    toGrad<Cyl,Car>. Reusing the second one verbatim is what makes the descriptor
    path bit-for-bit identical to the virtual path instead of merely close (an
    independently written -dR*x/R - dphi*(-y/R^2) would fuse its multiply-adds
    differently and land 1 ULP away; see the FMA rule in potential_multipole.h).

    The FIRST step, toPosDeriv<Car,Cyl>, is defined in coord.cpp and is NOT
    device-callable, so its body is transcribed here -- the one place in this
    subsystem where an expression exists twice. Promoting it into coord.h the way
    5e04db5 promoted toGrad/toHess is the proper fix, but it changes the inlining
    context of every BasePotentialCyl::evalCar in the library and therefore has to
    be gated on local_notes/crosscheck_expansions.py as its own commit; the
    transcription below is byte-for-byte the coord.cpp body, including its R==0
    degenerate branch, and the round-trip gate in potential_multipole.cpp
    (multipoleDescBothPaths) is what proves the two agree.

    This conversion stage is fp64 even for T=float, which is the documented and
    measured-cheap (3-8%) fp64 residue of the Multipole device path: coord::GradCyl
    and PosDerivT are double-only structs, and templating them was measured not to
    be worth it (findings.md, "MEASURED: order cap ..."). */
/** coord::toPosDeriv<Car, Cyl>, transcribed from coord.cpp so it is device-callable
    (the original is an explicit specialization defined in that TU, untagged). Kept
    as its own named leaf rather than inlined into the caller for two reasons: it is
    the ONE expression in this subsystem that exists twice, so it should be findable;
    and the round-trip gate compares it BIT-FOR-BIT against the coord.cpp original
    over the same point sweep, which only works if the gate can call it.

    Byte-for-byte the coord.cpp body, degenerate branch included. */
AGAMA_DEVICE_INLINE void gpu_car_to_cyl_deriv(double px, double py,
    /*out*/ coord::PosDerivT<coord::Car, coord::Cyl>& deriv,
    /*out*/ double& R, /*out*/ double& phi)
{
    const double R2 = pow_2(px) + pow_2(py), Rd = std::sqrt(R2);
    if(Rd == 0) {
        // degenerate case, but provide something meaningful nevertheless,
        // assuming that these numbers will be multiplied by 0 anyway
        deriv.dRdx = deriv.dRdy = deriv.dphidx = deriv.dphidy = 1.;
        R   = 0;
        phi = 0;
        return;
    }
    const double cosphi = px/Rd, sinphi = py/Rd;
    deriv.dRdx   =  cosphi;
    deriv.dRdy   =  sinphi;
    deriv.dphidx = -sinphi/Rd;
    deriv.dphidy =  cosphi/Rd;
    R   = Rd;
    phi = math::atan2(py, px);
}

template<typename T>
AGAMA_DEVICE_INLINE void gpu_multipole_phi_acc(const GpuMultipoleTerm<T>& mt,
    const T* payload, T x, T y, T z,
    /*stored, nullable*/ T* phi, /*stored*/ T a[3], T* mpScratch)
{
    coord::PosDerivT<coord::Car, coord::Cyl> cd;
    double Rc, phic;
    gpu_car_to_cyl_deriv(static_cast<double>(x), static_cast<double>(y), cd, Rc, phic);
    // ---- the potential and its cylindrical gradient, from the device evaluator
    T pot = 0, gcyl[3];
    multipoleEvalDevice<T>(mt.d, payload + mt.blobOffset,
        static_cast<T>(Rc), z, static_cast<T>(phic),
        phi ? &pot : (T*)NULL, gcyl, (T*)NULL, mpScratch);
    // ---- coord::toGrad<Cyl, Car>, the real one
    coord::GradCyl gc;
    gc.dR   = gcyl[CYL_DR];
    gc.dz   = gcyl[CYL_DZ];
    gc.dphi = gcyl[CYL_DPHI];
    const coord::GradCar gcar = coord::toGrad<coord::Cyl, coord::Car>(gc, cd);
    a[0] = -static_cast<T>(gcar.dx);
    a[1] = -static_cast<T>(gcar.dy);
    a[2] = -static_cast<T>(gcar.dz);
    if(phi)
        *phi = pot;
}

/** STORE (not accumulate) Phi (optional) and the Cartesian acceleration
    a = -grad Phi of a single term's underlying concrete potential, ignoring any
    modifier transform the term carries -- the coordinates passed in are already
    in the leaf's own frame. The per-tag bodies are the same leaf-plus-glue
    sequences as the corresponding evalmanyPhiAccCarT lambdas in
    potential_analytic.h.

    `d` is needed only by GPU_POT_MULTIPOLE, which reads its side entry and the
    shared payload out of it; `mpScratch` likewise (NULL for a kernel instantiated
    without Multipole support, which such a descriptor can never reach -- see
    gpuDescNeedsMultipole).

    Split out from gpu_term_phi_acc() below so that the modified and unmodified
    paths share one copy of the per-tag switch. */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_term_leaf_phi_acc(const GpuPotDesc<T>& d, const GpuPotTerm<T>& t,
    T x, T y, T z, /*stored, nullable*/ T* phi, /*stored*/ T a[3], T* mpScratch)
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
    case GPU_POT_MULTIPOLE: {   // shape accepted by buildMultipoleDeviceDesc only
        if(mpScratch == (T*)NULL) {
            // Unreachable: buildGpuPotDesc only emits this tag into a descriptor
            // whose gpuDescNeedsMultipole() is true, and every caller uses that to
            // pick ORDER. Fail LOUD rather than reading past a null pointer if the
            // two ever drift apart.
            if(phi) *phi = T(NAN);
            a[0] = a[1] = a[2] = T(NAN);
            return;
        }
        gpu_multipole_phi_acc<T>(d.mp[t.aux], d.splineData, x, y, z, phi ? &pot : (T*)NULL,
            a, mpScratch);
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
AGAMA_DEVICE_INLINE void gpu_term_phi_acc(const GpuPotDesc<T>& d, const GpuPotTerm<T>& t,
    /*nullable: NULL means no modifier chain*/ const GpuPotXform<T>* xfp,
    T x, T y, T z, /*accumulated, nullable*/ T* phi, /*accumulated*/ T acc[3],
    T* mpScratch)
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
        gpu_term_leaf_phi_acc(d, t, q[0], q[1], q[2], phi ? &pot : (T*)NULL, ai, mpScratch);
        gpu_xform_vec(xf, ai, a);
        if(phi) *phi += xf.kphi * pot;
    } else {
        T pot = 0;
        gpu_term_leaf_phi_acc(d, t, x, y, z, phi ? &pot : (T*)NULL, a, mpScratch);
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
AGAMA_DEVICE_INLINE T gpu_term_leaf_rho(const GpuPotDesc<T>& d, const GpuPotTerm<T>& t,
    T x, T y, T z, T* mpScratch)
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
    case GPU_POT_MULTIPOLE: {
        if(mpScratch == (T*)NULL)
            return T(NAN);   // see the same guard in gpu_term_leaf_phi_acc
        // Multipole::densityCar routes through densityCyl(toPosCyl(pos)), and the
        // three-way radial dispatch inside it is NOT the Laplacian of the potential
        // everywhere -- see multipoleDensityDevice.
        //
        // NOTE the spelling: coord::toPos<Car,Cyl>, NOT the coord::toPosCyl<Car>
        // convenience wrapper. The wrapper is plain host-side `inline` (coord.h:767)
        // while the specialization it forwards to is AGAMA_DEVICE_INLINE, and nvcc
        // only WARNS (#20011 "calling a __host__ function from a __host__ __device__
        // function is not allowed") about the difference -- it then emits no code for
        // the branch. Measured with -Xptxas -v: the ORDER=32 density kernel had the
        // same 5,080-byte frame as the ORDER=0 one instead of 5,080+4,640, i.e. the
        // entire Multipole density path had been silently dropped from the kernel
        // while the build reported success. Any device-side use of a coord:: helper
        // must be the tagged specialization.
        const coord::PosCyl pc = coord::toPos<coord::Car, coord::Cyl>(coord::PosCar(
            static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)),
            coord::Cyl());
        return multipoleDensityDevice<T>(d.mp[t.aux].d, d.splineData + d.mp[t.aux].blobOffset,
            static_cast<T>(pc.R), static_cast<T>(pc.z), static_cast<T>(pc.phi), mpScratch);
    }
    default:  // unreachable if the descriptor was built by buildGpuPotDesc
        return 0;
    }
}

/** Accumulate the mass density of a single term into *rho, applying the term's
    modifier transform if it has one. A similarity transform multiplies the
    Laplacian by sc^2 on top of the potential's own kphi (see GpuPotXform). */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_term_dens(const GpuPotDesc<T>& d, const GpuPotTerm<T>& t,
    /*nullable: NULL means no modifier chain*/ const GpuPotXform<T>* xfp,
    T x, T y, T z, /*accumulated*/ T* rho, T* mpScratch)
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
        *rho += xf.kphi * xf.sc * xf.sc *
            gpu_term_leaf_rho(d, t, q[0], q[1], q[2], mpScratch);
    } else
        *rho += gpu_term_leaf_rho(d, t, x, y, z, mpScratch);
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

/** ORDER is the compile-time Multipole order cap this instantiation reserves
    per-thread scratch for; 0 (what every pre-Tier-2 call site gets through the
    convenience overload below) means "no Multipole term can appear here" and emits
    no scratch array at all, so the already-shipped analytic descriptor kernels are
    byte-for-byte the kernels they were. The host chooses between the two with
    gpuDescNeedsMultipole(). */
template<typename T, int ORDER>
AGAMA_DEVICE_INLINE void gpu_desc_phi_acc_ord(const GpuPotDesc<T>& d,
    T x, T y, T z, /*nullable*/ T* phi, T acc[3], T time)
{
    if(phi) *phi = 0;
    acc[0] = acc[1] = acc[2] = 0;
    // one scratch reused across terms: this runs inside the orbit kernel next to
    // DOP853's 120-element state/scratch arrays, where every extra live value
    // competes for registers and therefore for occupancy
    GpuPotXform<T> scratch;
    GpuDescScratch<T, ORDER> mps;   // empty struct, no local array, when ORDER == 0
    for(int c = 0; c < d.nterms; c++)
        gpu_term_phi_acc(d, d.terms[c], gpu_term_xform(d, d.terms[c], time, scratch),
            x, y, z, phi, acc, mps.mpScratch());
}

/// ORDER=0 spelling, preserving the pre-Tier-2 call signature
template<typename T>
AGAMA_DEVICE_INLINE void gpu_desc_phi_acc(const GpuPotDesc<T>& d,
    T x, T y, T z, /*nullable*/ T* phi, T acc[3], T time = 0)
{
    gpu_desc_phi_acc_ord<T, 0>(d, x, y, z, phi, acc, time);
}

/** Evaluate the mass density of the whole descriptor at one point: the sum of
    the members' densities, in member order (a composite's density is the sum of
    its components', same as its potential). ORDER as in gpu_desc_phi_acc_ord. */
template<typename T, int ORDER>
AGAMA_DEVICE_INLINE T gpu_desc_dens_ord(const GpuPotDesc<T>& d, T x, T y, T z, T time)
{
    T rho = 0;
    GpuPotXform<T> scratch;
    GpuDescScratch<T, ORDER> mps;   // empty struct, no local array, when ORDER == 0
    for(int c = 0; c < d.nterms; c++)
        gpu_term_dens(d, d.terms[c], gpu_term_xform(d, d.terms[c], time, scratch),
            x, y, z, &rho, mps.mpScratch());
    return rho;
}

/// ORDER=0 spelling, preserving the pre-Tier-2 call signature
template<typename T>
AGAMA_DEVICE_INLINE T gpu_desc_dens(const GpuPotDesc<T>& d, T x, T y, T z, T time = 0)
{
    return gpu_desc_dens_ord<T, 0>(d, x, y, z, time);
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
    double time, std::vector<double>* payload, bool foldTimeVarying,
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
        return gpuDescAddTerms(*m->component(0), desc, time, payload, foldTimeVarying,
            xf, true, stageBegin, stageCount);
    }
    #define AGAMA_GPU_MOD_UNWRAP(ModClass) \
        if(const ModClass<BasePotential>* m =                                          \
            dynamic_cast<const ModClass<BasePotential>*>(&pot))                        \
        {                                                                              \
            if(!payload || foldTimeVarying || m->gpuXformConstant()) {                  \
                double A[9], b[3], k, s;                                               \
                m->gpuXformStage(time, A, b, k, s);                                     \
                gpuXformComposeInner(xf, A, b, k, s);                                   \
                return gpuDescAddTerms(*m->component(0), desc, time, payload,           \
                    foldTimeVarying, xf, true, stageBegin, stageCount);                 \
            }                                                                          \
            if(!gpuDescFlushConstStage(desc, xf, hasXform, stageBegin, stageCount))     \
                return false;                                                          \
            if(desc.nstages >= GPU_MOD_MAX_STAGES)                                     \
                return false;                                                          \
            if(!m->gpuEmitStage(desc.stages[desc.nstages], *payload))                   \
                return false;                                                          \
            if(stageCount == 0)                                                        \
                stageBegin = desc.nstages;                                             \
            desc.nstages++;                                                            \
            stageCount++;                                                              \
            return gpuDescAddTerms(*m->component(0), desc, time, payload,               \
                foldTimeVarying, xf, false, stageBegin, stageCount);                    \
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
            if(!gpuDescAddTerms(*comp->component(c), desc, time, payload, foldTimeVarying,
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
    else if(const Multipole* p = dynamic_cast<const Multipole*>(&pot)) {
        // Multipole is likewise outside the blanket X-macro capability list: only
        // the shapes buildMultipoleDeviceDesc() validates are representable, and it
        // is the single authority on that question (order cap, shared knot vectors,
        // PowerLaw asymptotes, expected array sizes -- ~20 fail-closed conditions).
        if(!payload)
            return false;   // nowhere to put ~1 MB of coefficients; caller must pass one
        if(desc.nmp >= GPU_POT_MAX_MULTIPOLE)
            return false;   // more Multipoles than side-table slots: fall back to CPU
        GpuMultipoleTerm<double>& mt = desc.mp[desc.nmp];
        std::vector<double> blob;
        if(!buildMultipoleDeviceDesc<double>(*p, mt.d, blob))
            return false;
        // The kernel's scratch array is sized at compile time by
        // GPU_POT_MULTIPOLE_ORDER; anything needing more must not reach it. The
        // builder's own order cap already implies this, but assert it against the
        // number the kernel actually reserves rather than trusting two constants to
        // stay in step.
        if(multipoleDeviceScratchSize(mt.d) >
            MultipoleDeviceScratchMax<GPU_POT_MULTIPOLE_ORDER>::value)
            return false;
        // offsets into the payload are ints in the descriptor (a blob is ~1e5
        // elements, but check rather than assume)
        const std::size_t base = payload->size();
        if(base + blob.size() > static_cast<std::size_t>(2147483647))
            return false;
        mt.blobOffset = static_cast<int>(base);
        payload->insert(payload->end(), blob.begin(), blob.end());
        term.tag = GPU_POT_MULTIPOLE;
        term.aux = desc.nmp;
        desc.nmp++;
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
    folded into the terms. When they ARE folded (see `payload` /
    `foldTimeVaryingModifiers`) the resulting descriptor is valid AT THAT TIME ONLY;
    otherwise only the constant parts are folded at `time` (which is immaterial for
    them) and the rest stays re-evaluable.
    \param[in,out]  payload  the flat buffer that everything too large to travel by
    value goes into: modifier spline node arrays AND Multipole coefficient blobs.
    The CALLER must place its contents where the kernel can read them (host memory
    for Serial/OpenMP, device memory for Cuda) and set desc.splineData to point at
    them -- see castGpuSplineData() and the orbit path in orbit_gpu.cpp;
    desc.splineData is deliberately left NULL here, because only the caller knows
    which memory space the kernel will run in. Passing NULL means "I have no payload
    buffer": every modifier stage is then folded at `time`, and a Multipole member
    makes the build FAIL (there would be nowhere to put its coefficients).
    \param[in]  foldTimeVaryingModifiers  when true, time-VARYING modifier chains are
    folded at `time` as well, instead of being emitted as re-evaluable GpuModStage
    entries. That is what batch evaluation wants -- one `time` is shared by the whole
    batch, so folding is exact, not an approximation -- and it is a separate flag from
    `payload` because a batch evaluating a Multipole needs a payload buffer while
    still wanting its modifiers folded. The orbit kernel, which sees a different t at
    every RK stage, leaves it false.
    \return true on success; false if any member is not a representable type, a
    transform came out non-finite, a modifier spline could not be packed, a Multipole
    was not of a representable shape (or arrived with no payload buffer), or the
    member/stage/Multipole count exceeds GPU_POT_DESC_MAX_TERMS / GPU_MOD_MAX_STAGES
    / GPU_POT_MAX_MULTIPOLE -- in which case the caller should fall back to the CPU
    path (desc is left partially filled and must not be used). Use
    potential::unsupportedGPUPotentialName() for the user-facing error message. */
inline bool buildGpuPotDesc(const BasePotential& pot, GpuPotDesc<double>& desc,
    double time = 0, std::vector<double>* payload = NULL,
    bool foldTimeVaryingModifiers = false)
{
    desc.nterms  = 0;
    desc.nstages = 0;
    desc.nmp     = 0;
    desc.splineData = NULL;
    if(payload)
        payload->clear();
    GpuPotXform<double> xf;
    gpu_xform_identity(xf);
    return detail::gpuDescAddTerms(pot, desc, time, payload,
        foldTimeVaryingModifiers, xf, false, 0, 0);
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
    out.nmp = d.nmp;
    for(int i = 0; i < d.nmp; i++) {
        castMultipoleDeviceDesc<T>(d.mp[i].d, out.mp[i].d);
        out.mp[i].blobOffset = d.mp[i].blobOffset;
    }
    // set by the caller once the payload is in the right memory space
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

/** Round-trip gate engine for the Multipole descriptor path: evaluate `pot` at
    numPoints Cartesian points through BOTH the descriptor path
    (buildGpuPotDesc + gpu_desc_phi_acc_ord + gpu_desc_dens_ord, at ORDER =
    GPU_POT_MULTIPOLE_ORDER, in double) and the ordinary virtual path, and hand
    back both sets of numbers for a BIT-FOR-BIT comparison by the caller.

    Each output array holds 5*numPoints doubles per point: { Phi, ax, ay, az, rho }.

    IT LIVES IN potential_multipole.cpp ON PURPOSE, for the reason recorded at
    length on multipoleEvalBothPaths(): a bit-for-bit gate requires both sides
    compiled in the SAME translation unit, not merely built from the same source.
    Everything on the device side (multipoleEvalDevice, coord::toGrad) is
    header-inline, and the CPU reference's conversion stage
    (coord::evalAndConvert<Cyl,Car>) is header-inline too, so evaluating them from a
    test TU would compare two different FMA contractions of the same expression tree
    and the divergence count would track the test TU's flags rather than the code
    (measured 132 / 2,604 / 7,308 out of 50,688 for defaults / -fno-inline /
    -ffp-contract=off in the commit that built the evaluator). Compiled together in
    the TU that also defines Multipole::evalCyl, they move together under codegen
    drift and only a real arithmetic difference separates them.

    The CPU reference deliberately calls coord::evalAndConvert<Cyl,Car> and
    Multipole::densityCyl explicitly rather than going through the pot.eval(PosCar)
    /pot.density(PosCar) virtual entry points: BasePotentialCyl::evalCar is an
    in-class virtual, so its out-of-line copy is a weak symbol that the linker may
    take from any TU, which would put the reference's conversion arithmetic outside
    this TU again. What it computes is identical to what pot.eval(PosCar) computes --
    that IS the body of BasePotentialCyl::evalCar.

    \param[in]  pot  is the potential to check (a Multipole, or a composite/modifier
                chain containing one);
    \param[in]  numPoints, xyz  are N points packed as {x,y,z} triplets;
    \param[out] outCpu, outDev  must each hold 5*numPoints doubles;
    \param[out] nmp  if non-NULL, receives desc.nmp (so the caller can assert that
                the descriptor really did take the Multipole route);
    \param[out] exactRef  if non-NULL, receives true when the reference was built the
                exact way described above (which requires `pot` to be a BARE
                Multipole) and false when it came from the ordinary virtual Cartesian
                entry points, in which case the caller must use a tolerance rather
                than a bit comparison;
    \param[out] posDiffs  if non-NULL, receives the number of BITWISE differences
                between gpu_car_to_cyl_deriv (the transcribed Car->Cyl conversion the
                device path uses) and the coord.cpp toPosDeriv<Car,Cyl> it was copied
                from, counted over 6 quantities per point. This localizes any
                divergence: nonzero here means the duplicated expression, not the
                Multipole evaluator, is the source.
    \return  false if no descriptor could be built (nothing is written then). */
bool multipoleDescBothPaths(const BasePotential& pot, int numPoints, const double* xyz,
    double* outCpu, double* outDev, int* nmp, bool* exactRef, long* posDiffs);

/** The BIT-FOR-BIT half of the same gate: everything the Multipole term does after
    the Cartesian->cylindrical position conversion (multipoleEvalDevice, the
    coord::toGrad<Cyl,Car> that carries dPhi/dphi into Cartesian, and
    multipoleDensityDevice's three-way dispatch), with ONE conversion shared by both
    sides so the comparison is not polluted by it.

    The end-to-end comparison above cannot be bit-for-bit in principle: both paths
    must form x*x + y*y, and GCC may contract that as fma(x,x,y*y) at one inline site
    and fma(y,y,x*x) at the other, which round differently. So the claim is gated as a
    composition -- gpu_car_to_cyl_deriv == coord.cpp's toPosDeriv<Car,Cyl> exactly
    (multipoleDescBothPaths' posDiffs), and everything downstream exactly (here) --
    with the end-to-end number reported at a 1-ULP bound. See the long comment on the
    definition in potential_multipole.cpp.

    Same 5-doubles-per-point output layout as multipoleDescBothPaths.
    \return false if no device descriptor could be built. */
bool multipoleDescCylBothPaths(const Multipole& pot, int numPoints, const double* xyz,
    double* outCpu, double* outDev);

}  // namespace potential
