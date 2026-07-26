/** \file    potential_multipole.h
    \brief   density and potential approximations based on spherical-harmonic expansion
    \author  Eugene Vasiliev
    \date    2010-2026

    This module provides tools for representing arbitrary density and potential profiles
    in terms of spherical-harmonic (or multipole) expansion, with coefficients being
    interpolated in radius, and a related class for representing the potential in terms
    of a basis-set expansion (still spherical-harmonic in angles, but with a separate
    set of basis functions for the radial part).

    Mathematical background for spherical-harmonic transformation is provided in
    math_sphharm.h; in brief, the transformation is defined by two order parameters --
    lmax is the maximum index of Legendre polynomials in cos(theta), and mmax<=lmax is
    the maximum index of Fourier harmonics in phi. The choice of terms to be used in
    the expansion depends on symmetry properties of the model; for instance,
    in the triaxial case only terms with even and non-negative l,m are involved,
    and in the axisymmetric case only m=0 terms are non-zero.

    The density approximation uses cubic splines in log(r) for each sph.-harm. term,
    with power-law extrapolation to small and large radii.

    The potential approximation (Multipole) uses quintic splines in log(r), defined
    by their values and derivatives at grid nodes. Depending on the order of expansion,
    it employs either 1d splines for each term, or 2d splines in the (r,theta) plane
    for each azimuthal (m) Fourier harmonic, whichever is more efficient.
    In the second case, it is similar to CylSpline expansion, but the latter uses
    2d splines in scaled (R,z) coordinates for each m.
    Additionally, scaling transformations of coordinates and amplitudes are used
    to improve the accuracy of interpolation.
    The approximation is fairly accurate even with a rather sparse grid spacing:
    for instance, a grid with 20 nodes may cover a radial range spanning 6 orders
    of magnitude and still provide a good accuracy.
    Extrapolation of potential to small and large r (beyond the extent of the grid)
    is based on asymptotic power-law scaling of multipole coefficients, thus the density
    profile is generally well approximated even outside the grid.
    This asymptotic power-law extrapolation is also used in CylSpline.

    The potential expansion may be computed either from an existing potential
    (providing a computationally efficient approximation to a possibly expensive
    potential), from a density profile (thus solving the Poisson equation in spherical
    harmonics), or from an array of point masses (again solving the Poisson equation).
    Once constructed, the coefficients of expansion can be stored to and subsequently
    loaded from a text file, using routines `readPotential/writePotential` in
    potential_factory.h
*/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_sphharm.h"
#include "math_spline.h"   // evalQuinticSplineRaw / evalQuinticSpline2dRaw (device leaves)
#include "smart.h"

namespace potential {

// =====================================================================
// Tier 2 leaf math for the Multipole evaluator: AGAMA_DEVICE_INLINE and
// templated on the value type T, single-source with the CPU path.
//
// The internal helpers in potential_multipole.cpp that Multipole::evalCyl
// funnels through are being moved here one at a time as templated
// device-callable free functions, with the .cpp version reduced to a thin
// wrapper, so the virtual single-point path and any templated batch/device
// path stay bit-for-bit identical (CLAUDE.md constraint 5, refereed by
// crosscheck_expansions.py).
//
// Moved so far: fourierTransformAzimuth, the MultipoleInterp2d log-scaling
// un-transform, and both Sph->Cyl / tau->Cyl derivative transforms. All four
// gate at 0.000e+00 against a baseline recorded before this work.
//
// THE RULE THAT MAKES THESE MOVES EXACT -- learned the hard way, measured:
// relocate a COMPLETE unit, never split one. coord::toGrad/toHess became
// header-inline in 5e04db5, so anything that perturbs their call context
// perturbs their FMA contraction. Moving a whole block -- Jacobian
// construction AND the toGrad/toHess that consumes it, in one function, body
// character-identical, coord:: structs passed by reference -- is exact.
// Splitting that same block is NOT: passing grad/hess across a plain-T array
// boundary, or factoring out just the Jacobian and leaving toGrad/toHess at
// the call site, each moved 12 of the 57 BFE quantities and took 8 of them
// from EXACTLY 0.0 vs production agama to ~2e-16, losing the bit-for-bit
// agreement the axisymmetric Multipole cases had. Both were implemented and
// measured; see findings.md. Do not re-derive them, and do not "tidy" these
// functions by splitting them up.
//
// CONVENTIONS:
//
// (1) Scratch space is CALLER-PROVIDED. The CPU wrappers keep passing
//     alloca'd buffers, so they stay valid for arbitrary lmax/mmax exactly
//     as upstream; a device caller passes a fixed-size local array bounded
//     by its own compile-time cap. This is what lets the device path have a
//     cap without imposing one on the CPU path.
//
// (2) Coefficient arrays travel as plain T (fully honest at T=float, since
//     they touch no coord:: types) and the enums below fix the component
//     order -- do not reorder, call sites index positionally. The two
//     derivative TRANSFORMS are the exception: they take coord:: structs and
//     are fp64-only, because those structs hard-code `double`. That is the
//     fp64 residue of the Multipole device path; see tauToCylDerivs.
// =====================================================================

/// component order for the T grad[3] arrays below, in the SCALED spherical
/// coordinates the Multipole interpolators work in (for MultipoleInterp2d,
/// "r" means ln(r) and "theta" means tau -- see its evalCyl)
enum SphGradIndex { SPH_DR = 0, SPH_DTHETA = 1, SPH_DPHI = 2 };

/// component order for the T hess[6] arrays below
enum SphHessIndex {
    SPH_DR2 = 0, SPH_DRDTHETA = 1, SPH_DTHETA2 = 2,
    SPH_DRDPHI = 3, SPH_DTHETADPHI = 4, SPH_DPHI2 = 5
};

/** Fourier synthesis in the azimuthal angle: given the per-m harmonic
    coefficients C_m, produce the value and (optionally) the first and second
    derivatives in (r, theta, phi). Device-callable, templated restatement of
    fourierTransformAzimuth() in potential_multipole.cpp, which is now a thin
    wrapper over this.

    C_m holds nq * nm entries, nm = ind.mmax - ind.mmin() + 1, laid out as
    nm potential harmonics, then nm for dPhi/dr, and so on -- upstream's
    layout, unchanged. nq is implied by which outputs are requested: 6 if
    hess, else 3 if grad, else 1.

    \param[in]  ind   is the POD indexing scheme (see SphHarmIndices::pod()).
    \param[in]  phi   is the azimuthal angle.
    \param[in]  C_m   is the coefficient array described above.
    \param[in]  trig_m is caller-provided scratch of at least
                ind.mmax * (1 + useSine) entries, where useSine is
                (ind.mmin() < 0 || nq > 1). Unused when ind.mmax == 0, and
                may then be NULL.
    \param[out] val   receives the value      if != NULL.
    \param[out] grad  receives 3 components   if != NULL (SphGradIndex order).
    \param[out] hess  receives 6 components   if != NULL (SphHessIndex order).
*/
template<typename T>
AGAMA_DEVICE_INLINE void fourierTransformAzimuthT(
    const math::SphHarmIndicesPod& ind, const T phi, const T* C_m, T* trig_m,
    T* val, T* grad, T* hess)
{
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;
    const int mmin = ind.mmin();
    const int nm = ind.mmax - mmin + 1;   // number of azimuthal harmonics in C_m
    // first assign the m=0 harmonic, which is the only one in the axisymmetric case
    if(val)
        *val = C_m[-mmin];
    if(grad) {
        grad[SPH_DR]     = C_m[-mmin+nm];
        grad[SPH_DTHETA] = C_m[-mmin+nm*2];
        grad[SPH_DPHI]   = 0;
    }
    if(hess) {
        hess[SPH_DR2]      = C_m[-mmin+nm*3];
        hess[SPH_DRDTHETA] = C_m[-mmin+nm*4];
        hess[SPH_DTHETA2]  = C_m[-mmin+nm*5];
        hess[SPH_DRDPHI]   = hess[SPH_DTHETADPHI] = hess[SPH_DPHI2] = 0;
    }
    if(ind.mmax == 0)
        return;
    const bool useSine = mmin<0 || numQuantities>1;
    math::trigMultiAngle(phi, ind.mmax, useSine, trig_m);
    for(int mm=0; mm<nm; mm++) {
        int m = mm + mmin;
        if(m==0)
            continue;  // the m=0 terms were set at the beginning
        if(ind.lmin(m)>ind.lmax)
            continue;  // empty harmonic
        T trig  = m>0 ? trig_m[m-1] : trig_m[ind.mmax-m-1];  // cos or sin
        T dtrig = m>0 ? -m*trig_m[ind.mmax+m-1] : -m*trig_m[-m-1];
        T d2trig = -m*m*trig;
        if(val)
            *val += C_m[mm] * trig;
        if(grad) {
            grad[SPH_DR]     += C_m[mm+nm  ] *  trig;
            grad[SPH_DTHETA] += C_m[mm+nm*2] *  trig;
            grad[SPH_DPHI]   += C_m[mm]      * dtrig;
        }
        if(hess) {
            hess[SPH_DR2]       += C_m[mm+nm*3] *   trig;
            hess[SPH_DRDTHETA]  += C_m[mm+nm*4] *   trig;
            hess[SPH_DTHETA2]   += C_m[mm+nm*5] *   trig;
            hess[SPH_DRDPHI]    += C_m[mm+nm  ] *  dtrig;
            hess[SPH_DTHETADPHI]+= C_m[mm+nm*2] *  dtrig;
            hess[SPH_DPHI2]     += C_m[mm]      * d2trig;
        }
    }
}


/** Undo MultipoleInterp2d's amplitude log-scaling, in place on the C_m array.

    Device-callable, templated restatement of the `if(logScaling)` block inside
    MultipoleInterp2d::evalCyl, which is now a thin call to this. Two stages, in this
    order (which matters -- the second reads the first's output):

      1. invert the log-scaling of the m=0 term, which lives at array index -mmin:
         Phi = 1 / (invPhi0 - exp(X)) and the chain rule through it;
      2. multiply every other m term, which is stored as a RATIO to the m=0 term, by
         the now-unscaled m=0 value, with the product rule for its derivatives.

    Contains no coordinate transforms and no spline lookups -- pure arithmetic over the
    coefficient array -- so unlike the Sph->Cyl derivative stage it is fully honest at
    T=float, and unlike that stage relocating it does not disturb any FMA context in
    coord (measured: gate stays at 0.000e+00).

    \param[in]     ind is the POD indexing scheme; mmin/nm/lmin are read from it.
    \param[in]     numQuantities is 1, 3 or 6 -- how much of C_m is populated.
    \param[in]     invPhi0 is the inverse of the potential at origin (may be zero).
    \param[in,out] C_m is the nm*numQuantities coefficient array, laid out as
                   {Phi, dlnr, dtau, dlnr2, dlnrdtau, dtau2}, each of length nm.

    NOTE the sub-array pointers for the second-derivative blocks are formed
    unconditionally, exactly as the original body does, and are simply not dereferenced
    when numQuantities < 6. Forming a pointer one past the end of the array is what the
    original does too; do not "fix" it into a conditional, that changes nothing and
    diverges from the transcribed source.
*/
template<typename T>
AGAMA_DEVICE_INLINE void multipoleUnscaleLogT(const math::SphHarmIndicesPod& ind,
    const int numQuantities, const T invPhi0, T* C_m)
{
    const int mmin = ind.mmin(), nm = ind.mmax - mmin + 1;
    T *Phi   = C_m,
      *dlnr  = C_m+nm,
      *dtau  = C_m+nm*2,
      *dlnr2 = C_m+nm*3,
      *dlnrdtau = C_m+nm*4,
      *dtau2 = C_m+nm*5;
    // transform the amplitude: first perform the inverse log-scaling for the m=0 term,
    // which resides in the array elements with index mm = 0 - mmin
    T expX = std::exp(Phi[-mmin]), val = T(1) / (invPhi0 - expX);
    Phi[-mmin]  = val;
    if(numQuantities>=3) {
        T dPhidX = pow_2(val) * expX;
        if(numQuantities==6) {
            T d2PhidX2 = dPhidX * val * (invPhi0 + expX);
            dlnr2   [-mmin] = dPhidX * dlnr2   [-mmin] + d2PhidX2 * dlnr[-mmin] * dlnr[-mmin];
            dtau2   [-mmin] = dPhidX * dtau2   [-mmin] + d2PhidX2 * dtau[-mmin] * dtau[-mmin];
            dlnrdtau[-mmin] = dPhidX * dlnrdtau[-mmin] + d2PhidX2 * dlnr[-mmin] * dtau[-mmin];
        }
        dlnr[-mmin] *= dPhidX;
        dtau[-mmin] *= dPhidX;
    }

    // then multiply other terms by the value of the m=0 term, which resides in the [-mmin] element
    for(int mm=0; mm<nm; mm++) {
        int m = mm + mmin;
        if(m==0 || ind.lmin(m) > ind.lmax)
            continue;
        if(numQuantities==6) {
            dlnr2[mm] = dlnr2[mm] * Phi[-mmin] + Phi[mm] * dlnr2[-mmin] + 2 * dlnr[mm] * dlnr[-mmin];
            dtau2[mm] = dtau2[mm] * Phi[-mmin] + Phi[mm] * dtau2[-mmin] + 2 * dtau[mm] * dtau[-mmin];
            dlnrdtau[mm] = dlnrdtau[mm] * Phi[-mmin] + Phi[mm] * dlnrdtau[-mmin] +
                dlnr[mm] * dtau[-mmin] + dtau[mm] * dlnr[-mmin];
        }
        if(numQuantities>=3) {
            dlnr[mm] = dlnr[mm] * Phi[-mmin] + Phi[mm] * dlnr[-mmin];
            dtau[mm] = dtau[mm] * Phi[-mmin] + Phi[mm] * dtau[-mmin];
        }
        Phi[mm] *= Phi[-mmin];
    }
}


/** Transform potential derivatives from scaled spherical {ln(r), theta} to cylindrical
    {R, z}. Device-callable; the whole body moved here VERBATIM from
    potential_multipole.cpp, and PowerLawMultipole/MultipoleInterp1d call it unchanged.

    Distinct from tauToCylDerivs() below -- there the second scaled coordinate is
    tau = z/(r+|R|), here it is the polar angle theta. Different Jacobian, different
    second-derivative block. Do not merge them.

    fp64 only, and the same residue tauToCylDerivs documents: coord::GradT / HessT /
    PosDerivT hard-code `double`, so this stage cannot be honestly templated on a value
    type. See tauToCylDerivs for the cost estimate and where the fix belongs.
*/
AGAMA_DEVICE_INLINE void sphToCylDerivs(const coord::PosCyl& pos,
    const coord::GradSph &gradSph, const coord::HessSph &hessSph,
    coord::GradCyl *gradCyl, coord::HessCyl *hessCyl)
{
    // abuse the coordinate transformation framework (Sph -> Cyl), where actually
    // in the source grad/hess we have derivs w.r.t. ln(r) instead of r
    const double r2inv = 1 / (pow_2(pos.R) + pow_2(pos.z));
    coord::PosDerivT<coord::Cyl, coord::Sph> der;
    der.drdR = pos.R * r2inv;
    der.drdz = pos.z * r2inv;
    der.dthetadR =  der.drdz;
    der.dthetadz = -der.drdR;
    if(gradCyl)
        *gradCyl = coord::toGrad(gradSph, der);
    if(hessCyl) {
        coord::PosDeriv2T<coord::Cyl, coord::Sph> der2;
        der2.d2rdR2      = pow_2(der.drdz) - pow_2(der.drdR);
        der2.d2rdRdz     = -2 * der.drdR * der.drdz;
        der2.d2rdz2      = -der2.d2rdR2;
        der2.d2thetadR2  =  der2.d2rdRdz;
        der2.d2thetadRdz = -der2.d2rdR2;
        der2.d2thetadz2  = -der2.d2rdRdz;
        *hessCyl = coord::toHess(gradSph, hessSph, der, der2);
    }
}


/** Transform MultipoleInterp2d's derivatives from its scaled coordinates {ln r, tau} to
    cylindrical {R, z}. Device-callable, and the single source of this arithmetic: the tail
    of MultipoleInterp2d::evalCyl is now one call to this.

    This is NOT the same transform as transformDerivsSphToCyl() in potential_multipole.cpp,
    and the two must not be merged: there the second scaled coordinate is the polar angle
    theta, here it is tau = z / (r + |R|), which gives different dthetadR / dthetadz and a
    completely different second-derivative block. Upstream keeps them separate for the same
    reason.

    ### WHY THIS TAKES coord:: STRUCTS AND IS NOT TEMPLATED ON A VALUE TYPE

    Both alternatives were tried and both cost bit-for-bit agreement with upstream. Passing
    grad/hess across a plain-T array boundary, and separately factoring out just the
    Jacobian while leaving toGrad/toHess at the call site, each moved 12 of the 57 BFE
    quantities -- and, the number that settled it, took 8 of them from EXACTLY 0.0 vs
    production agama to ~2e-16, losing the bit-for-bit agreement the axisymmetric Multipole
    cases had. Cause: coord::toGrad/toHess became header-inline in 5e04db5, so perturbing
    their call context perturbs their FMA contraction. Keeping the Jacobian construction and
    its consumption together in one function, body character-identical, structs by
    reference, avoids that entirely -- 0.000e+00.

    Consequence, which is a real limitation and not an oversight: this stage runs in fp64
    whatever precision the rest of the kernel uses, because coord::GradT / HessT /
    PosDerivT hard-code `double`. It is the fp64 residue of the Multipole device path.
    ~40 fp64 flops against roughly 3400 fp32 flops for the 17 azimuthal 2D-quintic
    evaluations of an lmax=mmax=8 model -- bounded, and NOT the ~200x an fp64 spline lookup
    would have cost. The fix belongs in coord (defaulted value-type parameter on the
    structs; note toGrad/toHess are function templates with explicit specializations, which
    C++ cannot partially specialize, so they must become overloads or a class template).

    \param[in]  R, z      is the position in cylindrical coordinates.
    \param[in]  r         is sqrt(R^2+z^2); passed in because the caller already has it,
                          and recomputing it here could round differently.
    \param[in]  rplusRinv is 1/(r+|R|), likewise already available in the caller.
    \param[in]  tau       is the scaled polar coordinate z*rplusRinv (or sign(z) at R==0).
    \param[in]  trGrad    is the gradient w.r.t. {ln r, tau, phi}.
    \param[in]  trHess    is the hessian in the same scaled coordinates; read only when
                          hess != NULL (callers may leave it uninitialized otherwise).
    \param[out] grad      receives the cylindrical gradient if != NULL.
    \param[out] hess      receives the cylindrical hessian if != NULL.
*/
AGAMA_DEVICE_INLINE void tauToCylDerivs(const double R, const double z, const double r,
    const double rplusRinv, const double tau,
    const coord::GradSph& trGrad, const coord::HessSph& trHess,
    coord::GradCyl* grad, coord::HessCyl* hess)
{
    // abuse the coordinate transformation framework (Sph -> Cyl), where actually
    // our source Sph coords are not (r, theta, phi), but (ln r, tau, phi)
    const double
        rinv  = 1/r,
        r2inv = pow_2(rinv);
    coord::PosDerivT<coord::Cyl, coord::Sph> der;
    der.drdR = R * r2inv;
    der.drdz = z * r2inv;
    der.dthetadR = -tau * rinv;
    der.dthetadz = rinv - rplusRinv;
    if(grad)
        *grad = coord::toGrad(trGrad, der);
    if(hess) {
        coord::PosDeriv2T<coord::Cyl, coord::Sph> der2;
        der2.d2rdR2  = pow_2(der.drdz) - pow_2(der.drdR);
        der2.d2rdRdz = -2 * der.drdR * der.drdz;
        der2.d2rdz2  = -der2.d2rdR2;
        der2.d2thetadR2  = z * r2inv * rinv;
        der2.d2thetadRdz = pow_2(der.dthetadR) - pow_2(der.drdR) * r * rplusRinv;
        der2.d2thetadz2  = -der2.d2thetadR2 - der.dthetadR * rplusRinv;
        *hess = coord::toHess(trGrad, trHess, der, der2);
    }
}


// =====================================================================
// Tier 2 commit 5: the whole Multipole evaluator, device-callable.
//
// Two pieces:
//   MultipoleDeviceDesc<T> + buildMultipoleDeviceDesc<T>()  -- a by-value POD
//       descriptor plus ONE flat blob of T holding every knot vector and every
//       spline/power-law coefficient of a Multipole object. Built on the host
//       from the live object; the blob is what a kernel would upload once and
//       keep resident.
//   multipoleEvalDevice<T>()  -- reproduces Multipole::evalCyl's 4-branch
//       dispatch (inner PowerLaw / outer PowerLaw / MultipoleInterp1d /
//       MultipoleInterp2d) reading only the descriptor + blob, calling the
//       device leaves that already exist (fourierTransformAzimuthT,
//       multipoleUnscaleLogT, sphToCylDerivs, tauToCylDerivs,
//       math::evalQuinticSplineRaw / evalQuinticSpline2dRaw,
//       math::sphHarmArray / trigMultiAngle).
//
// WHAT IS AND IS NOT SINGLE-SOURCED WITH THE CPU PATH, AND WHY
//
// All the *arithmetic kernels* are shared with the virtual CPU path: the two
// spline evaluators, the harmonics, the azimuthal Fourier synthesis, the
// Interp2d log-unscaling and both derivative transforms are literally the same
// functions the .cpp calls. What is restated here is the LOOP STRUCTURE around
// them (the per-(l,m) / per-m gather, the Interp1d log-scaling chain rule, and
// the PowerLaw coefficient expression) -- for one deliberate reason:
// PowerLawMultipole's device shape is FUSED. Upstream materializes
// Phi_lm[3*(lmax+1)^2] in one loop and consumes it in another; at order 32 that
// is 30,112 B/thread of local memory (measured, -Xptxas -v), versus 3,968 B when
// each (l,m) coefficient is computed inside the harmonic loop that consumes it.
// Every element's defining expression is unchanged -- the single copy of it is
// multipolePowerLawTermT() below, called by both the fused loop and the lmax==0
// fast track -- so fusing cannot move a bit, and the CPU-side round-trip gate in
// tests/test_gpu_policy.cpp proves it did not: the device path is compared
// BIT-FOR-BIT (doubles as integers) against Multipole::eval on the same points,
// across all four branches, all three symmetry classes and all output
// combinations. A tolerance would be worthless here; identity is the property.
//
// Folding the .cpp's own eval bodies onto these functions (making
// PowerLawMultipole::evalCyl etc. thin wrappers) is the natural follow-up and
// would remove the restated loops entirely, but it perturbs the call context of
// coord::toGrad/toHess inside the CPU path and therefore MUST be gated on
// local_notes/crosscheck_expansions.py -- see the FMA rule at the top of this
// file. It is deliberately not done in this commit, so that this commit cannot
// change a single CPU-path bit.
//
// SCRATCH is caller-provided, as everywhere else in this file: pass
// multipoleDeviceScratchSize(desc) elements of T, or the compile-time worst case
// MultipoleDeviceScratchMax<ORDER>::value for a kernel with a fixed order cap.
//
// ORDER CAP is math::LEGENDRE_MMAX (32) and nothing lower: the builder rejects
// anything above it (keeping legendrePmm's NAN branch unreachable on device),
// and MultipoleInterp2d -- the branch every realistic GalPot MW potential takes
// -- was measured at only 3.7 KB/thread and 155 registers with ZERO spills at
// order 32, because its scratch scales with mmax, not lmax^2.
// =====================================================================

/// component order for the T gradCyl[3] arrays below: coord::GradCyl's member
/// order (coord.h), so the conversion is a field-by-field copy with no reordering
enum CylGradIndex { CYL_DR = 0, CYL_DZ = 1, CYL_DPHI = 2 };

/// component order for the T hessCyl[6] arrays below: coord::HessCyl's member order
enum CylHessIndex {
    CYL_DR2 = 0, CYL_DZ2 = 1, CYL_DPHI2 = 2,
    CYL_DRDZ = 3, CYL_DZDPHI = 4, CYL_DRDPHI = 5
};

/// which interpolator Multipole::impl is; mirrors the LMAX_1D_SPLINE choice made
/// in the Multipole constructor (1d splines per (l,m) iff lmax <= 2, else one 2d
/// spline per azimuthal m)
enum MultipoleImplKind { MULTIPOLE_IMPL_INTERP1D = 0, MULTIPOLE_IMPL_INTERP2D = 1 };

/** By-value POD description of a Multipole potential for the device path: fixed
    size, no pointers, no std:: containers, trivially copyable into a kernel
    argument or into a descriptor table. All bulk data lives in a separate flat
    array of T ("the blob") and is addressed by the integer element offsets below,
    which is the shape potential_descriptor.h prescribes for coefficient-carrying
    potentials (one payload pointer for the whole object, offsets in the term).

    Build one ONLY via buildMultipoleDeviceDesc(); it is the only place that can
    see MultipoleInterp1d/MultipoleInterp2d (declared inside
    potential_multipole.cpp) and the only place that validates the assumptions the
    layout below depends on. It FAILS CLOSED -- returns false, caller falls back
    to the CPU -- for any order above math::LEGENDRE_MMAX, any unrecognised branch
    class, any non-shared knot vector, or any unexpected array size. A wrong
    descriptor is far worse than a CPU fallback.

    ### BLOB LAYOUT

    All offsets are ELEMENT indices into the blob (not bytes), all entries are T.
    The blob has exactly the size buildMultipoleDeviceDesc() gave the vector, and
    every region below is contiguous, in this order:

      [offXval, offXval+nx)     the ln(r) knot vector. SHARED by every spline in
                                the object -- MultipoleInterp1d builds all its
                                (l,m) splines on one gridR, MultipoleInterp2d all
                                its m splines on one (gridR, gridT) -- so it is
                                stored ONCE, not per harmonic. The builder
                                verifies that against every present spline and
                                returns false if any disagrees. offXval == 0 and
                                nx == the number of radial grid nodes.

      [offYval, offYval+ny)     the tau = cos(theta)/(sin(theta)+1) knot vector.
                                MULTIPOLE_IMPL_INTERP2D only; for INTERP1D
                                ny == 0 and this region is empty.

      [offCoefs, offCoefs + nslot*slotStride)    the coefficient block:

        MULTIPOLE_IMPL_INTERP2D:
            nslot      = 2*ind.mmax + 1          (mirrors MultipoleInterp2d::spl)
            slotStride = 9*nx*ny
            slot index = m + ind.mmax   for ind.mmin() <= m <= ind.mmax
            slot content: nine contiguous nx*ny arrays, each row-major with the
            ln(r) index major (exactly as math::Matrix / BaseInterpolator2d store
            them: element (i,j) at i*ny+j), in precisely the argument order of
            math::evalQuinticSpline2dRaw:
                +0*nx*ny fval   +1*nx*ny fx     +2*nx*ny fy
                +3*nx*ny fxx    +4*nx*ny fxy    +5*nx*ny fyy
                +6*nx*ny fxxy   +7*nx*ny fxyy   +8*nx*ny fxxyy

        MULTIPOLE_IMPL_INTERP1D:
            nslot      = ind.size() = (ind.lmax+1)^2  (mirrors MultipoleInterp1d::spl)
            slotStride = 3*nx
            slot index = c = math::SphHarmIndicesPod::index(l,m)
            slot content: three contiguous nx arrays, in evalQuinticSplineRaw order:
                +0*nx fval      +1*nx fder      +2*nx fder2

        Slots belonging to an m (or an (l,m)) that the indexing scheme does not
        use -- ind.lmin(m) > ind.lmax -- are PRESENT but zero-filled and never
        read: multipoleEvalDevice skips exactly the harmonics the CPU loop skips.
        Keeping the full 2*mmax+1 / (lmax+1)^2 stride makes the slot index
        identical to the CPU container index, at the cost of leaving the m<0 half
        unused when the model is y-reflection symmetric (mmin()==0). That waste is
        bounded by 2x on a ~1 MB payload and buys an indexing rule a kernel author
        can apply from this comment alone.

      [offInnerSUW, +3*ind.size())   the inner PowerLawMultipole's coefficients as
                                     three contiguous ind.size() arrays: S, U, W.
      [offOuterSUW, +3*ind.size())   the same three arrays for the outer asymptote.

    Note ind.size() == indInner.size() == indOuter.size(): PowerLawMultipole
    derives its own indexing scheme from the U array via
    math::getIndicesFromCoefs, which fixes lmax = sqrt(U.size())-1, so the three
    schemes always share lmax and can differ only in mmax and symmetry.

    Coefficients are stored in the blob as T. For T=float that is a narrowing cast
    AT UPLOAD, which is the established pattern for the device fp32 path; the host
    classes keep their fp64 storage untouched.
*/
template<typename T>
struct MultipoleDeviceDesc {
    math::SphHarmIndicesPod ind;       ///< indexing scheme of the interpolated (impl) expansion
    math::SphHarmIndicesPod indInner;  ///< ... of the inner power-law asymptote
    math::SphHarmIndicesPod indOuter;  ///< ... of the outer power-law asymptote
    T rminSq;        ///< inner branch boundary, ALREADY squared and safety-factored
    T rmaxSq;        ///< outer branch boundary, likewise
    int implKind;    ///< one of MultipoleImplKind
    int logScaling;  ///< impl's logScaling flag (int, to keep this POD trivially printable)
    T invPhi0;       ///< impl's inverse potential at origin; may be zero
    int offXval, nx; ///< ln(r) knot vector: shared across all m / all (l,m)
    int offYval, ny; ///< tau knot vector (MULTIPOLE_IMPL_INTERP2D only; ny==0 otherwise)
    int offCoefs;    ///< start of the coefficient block
    T r0sqInner, r0sqOuter;   ///< PowerLaw reference radii, squared
    T qInner, qOuter;         ///< PowerLaw Q coefficient (the r^2 term of the inner monopole)
    int offInnerSUW, offOuterSUW;  ///< start of each asymptote's {S,U,W} block
};

class Multipole;

/** Extract a device-ready descriptor + coefficient blob from a live Multipole.
    Host-only (it needs RTTI and std::vector). Returns false, having left `desc`
    unspecified and `blob` empty, if anything about the object is not exactly one
    of the shapes the layout above describes -- see the fail-closed list on
    MultipoleDeviceDesc. Instantiated for T = double and T = float.
    \param[in]  pot   is the potential to describe.
    \param[out] desc  receives the POD descriptor.
    \param[out] blob  receives the flat coefficient payload, indexed by desc.
    \return  true on success; false means "no device path for this object".
*/
template<typename T>
bool buildMultipoleDeviceDesc(const Multipole& pot, MultipoleDeviceDesc<T>& desc,
    std::vector<T>& blob);

/** Narrow a descriptor to the kernel's working precision. The counterpart of
    castGpuPotDesc/castGpuSplineData in potential_descriptor.h and used the same
    way: the DOUBLE descriptor and blob are built once from the live object (so the
    host classes' fp64 storage is the single source), and both are cast together
    just before a kernel is instantiated. Cheaper and, more importantly, safer than
    calling buildMultipoleDeviceDesc<float> separately -- two independent builds
    could in principle disagree about which shapes they accept. */
template<typename T>
inline void castMultipoleDeviceDesc(const MultipoleDeviceDesc<double>& in,
    /*out*/ MultipoleDeviceDesc<T>& out)
{
    out.ind         = in.ind;
    out.indInner    = in.indInner;
    out.indOuter    = in.indOuter;
    out.rminSq      = static_cast<T>(in.rminSq);
    out.rmaxSq      = static_cast<T>(in.rmaxSq);
    out.implKind    = in.implKind;
    out.logScaling  = in.logScaling;
    out.invPhi0     = static_cast<T>(in.invPhi0);
    out.offXval     = in.offXval;
    out.nx          = in.nx;
    out.offYval     = in.offYval;
    out.ny          = in.ny;
    out.offCoefs    = in.offCoefs;
    out.r0sqInner   = static_cast<T>(in.r0sqInner);
    out.r0sqOuter   = static_cast<T>(in.r0sqOuter);
    out.qInner      = static_cast<T>(in.qInner);
    out.qOuter      = static_cast<T>(in.qOuter);
    out.offInnerSUW = in.offInnerSUW;
    out.offOuterSUW = in.offOuterSUW;
}

/** Evaluate a Multipole at numPoints points (packed {R,z,phi} triplets) through
    BOTH the virtual CPU path and multipoleEvalDevice<double>, and return both sets
    of numbers for a bit-for-bit comparison by the caller. Each output array holds
    10*numPoints doubles: {Phi, grad[3] in CylGradIndex order, hess[6] in
    CylHessIndex order} per point, with unrequested slots zeroed on both sides.

    This is the round-trip gate's engine, and it lives in potential_multipole.cpp
    ON PURPOSE: comparing a header-inlined device evaluator in the CALLER's
    translation unit against the CPU path compiled in the library's is not a
    bit-for-bit test at all -- coord::toGrad/toHess are header-inline, so the two
    contexts fuse their multiply-adds differently and the observed divergence count
    tracks the caller's compiler flags rather than the code (132 / 2,604 / 7,308
    values out of 50,688 for project defaults / -fno-inline / -ffp-contract=off,
    with the library binary unchanged). Compiled together here, both sides move
    together under codegen drift and only real arithmetic changes separate them.

    \return false if no device descriptor could be built for this object, in which
    case nothing is written to the output arrays. */
bool multipoleEvalBothPaths(const Multipole& pot, MultipoleDeviceDesc<double>& desc,
    int numPoints, const double* Rzphi, bool wantGrad, bool wantHess,
    double* outCpu, double* outDev);

/** Does this indexing scheme hit sphHarmTransformInverseDeriv's optimized
    lmax==2 special case (potential_multipole.cpp: the {0,0},{2,0},{2,2} shortcut)?
    The device path must take the same fork, or it would compute different numbers
    for the very common triaxial lmax=2 model. */
AGAMA_DEVICE_INLINE bool multipoleDeriv2Shape(const math::SphHarmIndicesPod& ind)
{
    return ind.lmax==2 && ind.mmin()==0 && ind.step==2;
}

/** Scratch, in elements of T, for one branch of the dispatch with the given
    indexing scheme. The four terms are, in blob-free order:
      6*(2*mmax+1)      C_m, sized exactly as sphHarmTransformInverseDeriv's sizeC
      3*(lmax+1)        P_lm, dP_lm, d2P_lm for math::sphHarmArray
      2*mmax            trig_m for fourierTransformAzimuthT (cos and sin halves)
      3*(lmax+1)^2      Phi_lm, dPhi_lm, d2Phi_lm -- ONLY when the branch has to
                        materialize the (l,m) coefficient array, i.e. the
                        MultipoleInterp1d branch (which always does) and the
                        PowerLaw branch when it takes the lmax==2 shortcut (where
                        the term is at most 3*9 = 27). The fused PowerLaw loop
                        never needs it, which is the whole point of fusing. */
AGAMA_DEVICE_INLINE int multipoleBranchScratch(const math::SphHarmIndicesPod& ind,
    const bool needCoefArray)
{
    return 6*(2*ind.mmax+1) + 3*(ind.lmax+1) + 2*ind.mmax
        + (needCoefArray ? 3*(ind.lmax+1)*(ind.lmax+1) : 0);
}

/** Number of T elements of scratch multipoleEvalDevice needs for this descriptor:
    the worst case over the four branches, since a single point takes exactly one
    of them and the caller cannot know which in advance. */
template<typename T>
AGAMA_DEVICE_INLINE int multipoleDeviceScratchSize(const MultipoleDeviceDesc<T>& d)
{
    int n = multipoleBranchScratch(d.ind, d.implKind == MULTIPOLE_IMPL_INTERP1D);
    const int i = multipoleBranchScratch(d.indInner, multipoleDeriv2Shape(d.indInner));
    const int o = multipoleBranchScratch(d.indOuter, multipoleDeriv2Shape(d.indOuter));
    if(i > n) n = i;
    if(o > n) n = o;
    return n;
}

/** Compile-time worst-case scratch size (in elements of T) for ANY Multipole with
    lmax, mmax <= ORDER -- what a kernel with a fixed order cap declares as a local
    array. Dominant term is C_m; the +27 is the materialized coefficient array of
    the two branches that need one, both of which are pinned to lmax<=2 (the
    Interp1d branch by LMAX_1D_SPLINE, the PowerLaw shortcut by its own shape
    test), so it does NOT grow as 3*(ORDER+1)^2.

    At the shipping cap ORDER = math::LEGENDRE_MMAX = 32 this is 580 elements:
    4,640 B/thread in fp64, 2,320 B in fp32. */
template<int ORDER>
struct MultipoleDeviceScratchMax {
    static const int value = 6*(2*ORDER+1) + 3*(ORDER+1) + 2*ORDER + 27;
};

/** One {l,m} power-law coefficient of a PowerLawMultipole, and its first two
    derivatives w.r.t. ln(r). The single copy of this expression on the device
    path: called from the fused harmonic loop AND from the lmax==0 fast track, so
    the fused and un-fused shapes cannot drift apart.

    \param[in]  s, u, w   are S[c], U[c], W[c] for this harmonic;
    \param[in]  v         is l for the inward and -l-1 for the outward extrapolation;
    \param[in]  Q         is the extra r^2 coefficient (used only when v==0);
    \param[in]  rsq       is R^2+z^2, and r0sq the squared reference radius;
    \param[in]  dlogr     is ln(r/r0) = log(rsq/r0sq)*0.5, computed once by the caller;
    \param[in]  needD1, needD2  whether the derivative outputs are wanted;
    \param[out] p0, p1, p2  receive Phi_lm, dPhi_lm/dlnr, d2Phi_lm/dlnr^2.
*/
template<typename T>
AGAMA_DEVICE_INLINE void multipolePowerLawTermT(const T s, const T u, const T w, const T v,
    const T Q, const T rsq, const T r0sq, const T dlogr,
    const bool needD1, const bool needD2, T* p0, T* p1, T* p2)
{
    const T rv  = v!=0 ? std::exp( dlogr * v ) : T(1);                    // (r/r0)^v
    const T rs  = s!=v ? (s!=0 ? std::exp( dlogr * s ) : T(1)) : rv;     // (r/r0)^s
    const T urs = u * rs * (s!=v || u==0 ? T(1) : dlogr);  // if s==v, multiply by ln(r/r0)
    const T wrv = w * rv;
    const T qr2 = v==0 ? Q * rsq / r0sq : T(0);  // Q * (r/r0)^2, only for the inner monopole
    *p0 = urs + wrv + qr2;
    if(needD1)
        *p1 = urs*s + wrv*v + (s!=v ? T(0) : u*rs) + qr2*2;
    if(needD2)
        *p2 = urs*s*s + wrv*v*v + (s!=v ? T(0) : 2*s*u*rs) + qr2*4;
}

/** One {l,m} DENSITY coefficient of a PowerLawMultipole -- the body of the (m,l)
    loop in PowerLawMultipole::densityCyl, factored out so that the fast track and
    the fused transform below share one copy of the expression (the same discipline
    multipolePowerLawTermT above follows for the potential). Only the U term has a
    nonzero Laplacian, which is why W does not appear.

    \param[in]  s, u    are S[c], U[c] for this harmonic;
    \param[in]  v       is l for the inward and -l-1 for the outward extrapolation;
    \param[in]  Q       is the extra r^2 coefficient (contributes only when v==0);
    \param[in]  dlogr   is ln(r/r0), computed once by the caller;
    \param[in]  l       is the harmonic's degree (enters as the integer l*(l+1)).
    \return  rho_lm, still to be multiplied by 0.25/pi/r0^2 by the caller. */
template<typename T>
AGAMA_DEVICE_INLINE T multipolePowerLawDensityTermT(const T s, const T u, const T v,
    const T Q, const T dlogr, const int l)
{
    const T ursm2 = s!=2 ? u * std::exp( dlogr * (s-2) ) : u;   // u * (r/r0)^(s-2)
    if(s!=v)
        return ursm2 * (s*(s+1) - T(l*(l+1))) + (v==0 ? 6*Q : T(0));
    else
        return ursm2 * (s*(s+1) * dlogr - s*(s-1) + 1);
}

/** Device-callable restatement of sphHarmTransformInverseDeriv2 -- the optimized
    lmax==2, mmin==0, step==2 shortcut, processing only {0,0}, {2,0} and (if
    mmax==2) {2,2}. Outputs are plain T arrays in SphGradIndex / SphHessIndex
    order, the same convention fourierTransformAzimuthT uses.

    NOTE the sine/cosine pair is taken from math::sincos, which is fp64-only; at
    T=float this costs one double sincos. That is deliberate and cheap: this
    function only ever runs for lmax==2 models, and using a float sincos here
    would make the fp32 path disagree with the fp64 reference for no benefit. */
template<typename T>
AGAMA_DEVICE_INLINE void sphHarmTransformInverseDeriv2T(
    const math::SphHarmIndicesPod& ind, const T R, const T z, const T phi,
    const T* C_lm, const T* dC_lm, const T* d2C_lm,
    T* val, T* grad, T* hess)
{
    const int i00 = math::SphHarmIndicesPod::index(0,0),
              i20 = math::SphHarmIndicesPod::index(2,0),
              i22 = math::SphHarmIndicesPod::index(2,2);
    const T C2 = std::sqrt(T(1.25)), D2 = std::sqrt(T(3.75));
    const T
    tau = z == 0 ? T(0) : z / (std::sqrt(pow_2(R) + pow_2(z)) + R),
    ct  =      2 * tau  / (1 + tau*tau),  // cos(theta)
    st  = (1 - tau*tau) / (1 + tau*tau),  // sin(theta)
    cc  = ct * ct, cs = ct * st, ss = st * st,
    Y20       = (3*C2 * cc - C2),
    dY20      = -6*C2 * cs,
    d2Y20     =-12*C2 * cc + 6*C2;
    if(val)
        *val          =   Y20 *   C_lm[i20] +   C_lm[i00];
    if(grad) {
        grad[SPH_DR]     =   Y20 *  dC_lm[i20] +  dC_lm[i00];
        grad[SPH_DTHETA] =  dY20 *   C_lm[i20];
        grad[SPH_DPHI]   = 0;
    }
    if(hess) {
        hess[SPH_DR2]      =   Y20 * d2C_lm[i20] + d2C_lm[i00];
        hess[SPH_DRDTHETA] =  dY20 *  dC_lm[i20];
        hess[SPH_DTHETA2]  = d2Y20 *   C_lm[i20];
        hess[SPH_DRDPHI]   = hess[SPH_DTHETADPHI] = hess[SPH_DPHI2] = 0;
    }
    if(ind.mmax == 2) {
        double sp_d, cp_d;
        math::sincos(2 * static_cast<double>(phi), sp_d, cp_d);
        const T sp = static_cast<T>(sp_d), cp = static_cast<T>(cp_d);
        const T
        Y22       =    D2 * ss,
        dY22      =  2*D2 * cs,
        d2Y22     =  4*D2 * cc - 2*D2;
        if(val)
            *val += Y22 * C_lm[i22] * cp;
        if(grad) {
            grad[SPH_DR]     +=  Y22 *  dC_lm[i22] *    cp;
            grad[SPH_DTHETA] += dY22 *   C_lm[i22] *    cp;
            grad[SPH_DPHI]   +=  Y22 *   C_lm[i22] * -2*sp;
        }
        if(hess) {
            hess[SPH_DR2]        +=   Y22 * d2C_lm[i22] *    cp;
            hess[SPH_DRDTHETA]   +=  dY22 *  dC_lm[i22] *    cp;
            hess[SPH_DTHETA2]    += d2Y22 *   C_lm[i22] *    cp;
            hess[SPH_DRDPHI]     +=   Y22 *  dC_lm[i22] * -2*sp;
            hess[SPH_DTHETADPHI] +=  dY22 *   C_lm[i22] * -2*sp;
            hess[SPH_DPHI2]      +=   Y22 *   C_lm[i22] * -4*cp;
        }
    }
}

/** Device-callable restatement of sphHarmTransformInverseDeriv: the inverse
    spherical-harmonic transform of the coefficient arrays C_lm and their first
    two derivatives w.r.t. an arbitrary function of radius, into value / gradient /
    hessian in (that function of) spherical coordinates. Dispatches to
    sphHarmTransformInverseDeriv2T for the optimized shape, exactly as the CPU
    version does, and finishes through the existing fourierTransformAzimuthT leaf.

    \param[in]  ind       is the POD indexing scheme;
    \param[in]  R, z, phi is the position in cylindrical coordinates;
    \param[in]  C_lm, dC_lm, d2C_lm  are the ind.size()-long coefficient arrays
                (dC_lm/d2C_lm are read only when grad/hess are requested);
    \param[in]  scratch   is at least multipoleBranchScratch(ind, false) elements,
                and is not touched at all on the sphHarmTransformInverseDeriv2T fork;
    \param[out] val, grad, hess  as in fourierTransformAzimuthT.
*/
template<typename T>
AGAMA_DEVICE_INLINE void sphHarmTransformInverseDerivT(
    const math::SphHarmIndicesPod& ind, const T R, const T z, const T phi,
    const T* C_lm, const T* dC_lm, const T* d2C_lm, T* scratch,
    T* val, T* grad, T* hess)
{
    if(multipoleDeriv2Shape(ind)) {   // an optimized special case
        sphHarmTransformInverseDeriv2T<T>(ind, R, z, phi, C_lm, dC_lm, d2C_lm, val, grad, hess);
        return;
    }
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;  // number of quantities in C_m
    const int sizeC = 6 * (2*ind.mmax+1), sizeP = ind.lmax+1;
    T*   C_m  = scratch;
    T*   P_lm = C_m + sizeC;
    T*  dP_lm = numQuantities>=3 ? P_lm + sizeP   : NULL;
    T* d2P_lm = numQuantities==6 ? P_lm + sizeP*2 : NULL;
    T* trig_m = P_lm + sizeP*3;
    const T tau = z == 0 ? T(0) : z / (std::sqrt(pow_2(R) + pow_2(z)) + R);
    const int nm = ind.mmax - ind.mmin() + 1;  // number of azimuthal harmonics in C_m array
    for(int mm=0; mm<nm; mm++) {
        const int m = mm + ind.mmin();
        const int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        // extra factor sqrt{2} for m!=0 trig fncs
        const T mul = m==0 ? T(2*M_SQRTPI) : T(2*M_SQRTPI*M_SQRT2);
        for(int q=0; q<numQuantities; q++)
            C_m[mm + q*nm] = 0;
        const int absm = m<0 ? -m : m;
        math::sphHarmArray<T>(ind.lmax, absm, tau, P_lm, dP_lm, d2P_lm);
        for(int l=lmin; l<=ind.lmax; l+=ind.step) {
            const int c = math::SphHarmIndicesPod::index(l, m), p = l-absm;
            C_m[mm] += P_lm[p] * C_lm[c] * mul;
            if(numQuantities>=3) {
                C_m[mm + nm  ] +=  P_lm[p] * dC_lm[c] * mul;   // dPhi_m/dr
                C_m[mm + nm*2] += dP_lm[p] *  C_lm[c] * mul;   // dPhi_m/dtheta
            }
            if(numQuantities==6) {
                C_m[mm + nm*3] +=   P_lm[p] * d2C_lm[c] * mul; // d2Phi_m/dr2
                C_m[mm + nm*4] +=  dP_lm[p] *  dC_lm[c] * mul; // d2Phi_m/drdtheta
                C_m[mm + nm*5] += d2P_lm[p] *   C_lm[c] * mul; // d2Phi_m/dtheta2
            }
        }
    }
    fourierTransformAzimuthT<T>(ind, phi, C_m, trig_m, val, grad, hess);
}

/// copy a plain-T spherical grad/hess pair into the coord:: structs the two
/// fp64-only derivative transforms take (a pure copy -- no arithmetic, hence no
/// bit movement; the CPU's fourierTransformAzimuth wrapper does exactly this)
template<typename T>
AGAMA_DEVICE_INLINE void multipoleUnpackSph(const T* grad, const T* hess,
    coord::GradSph* gradSph, coord::HessSph* hessSph)
{
    if(grad) {
        gradSph->dr     = grad[SPH_DR];
        gradSph->dtheta = grad[SPH_DTHETA];
        gradSph->dphi   = grad[SPH_DPHI];
    }
    if(hess) {
        hessSph->dr2        = hess[SPH_DR2];
        hessSph->drdtheta   = hess[SPH_DRDTHETA];
        hessSph->dtheta2    = hess[SPH_DTHETA2];
        hessSph->drdphi     = hess[SPH_DRDPHI];
        hessSph->dthetadphi = hess[SPH_DTHETADPHI];
        hessSph->dphi2      = hess[SPH_DPHI2];
    }
}

/// copy the cylindrical grad/hess out of the coord:: structs into the plain-T
/// output arrays (again a pure copy, in CylGradIndex / CylHessIndex order)
template<typename T>
AGAMA_DEVICE_INLINE void multipolePackCyl(const coord::GradCyl& gc, const coord::HessCyl& hc,
    T* gradCyl, T* hessCyl)
{
    if(gradCyl) {
        gradCyl[CYL_DR]   = static_cast<T>(gc.dR);
        gradCyl[CYL_DZ]   = static_cast<T>(gc.dz);
        gradCyl[CYL_DPHI] = static_cast<T>(gc.dphi);
    }
    if(hessCyl) {
        hessCyl[CYL_DR2]    = static_cast<T>(hc.dR2);
        hessCyl[CYL_DZ2]    = static_cast<T>(hc.dz2);
        hessCyl[CYL_DPHI2]  = static_cast<T>(hc.dphi2);
        hessCyl[CYL_DRDZ]   = static_cast<T>(hc.dRdz);
        hessCyl[CYL_DZDPHI] = static_cast<T>(hc.dzdphi);
        hessCyl[CYL_DRDPHI] = static_cast<T>(hc.dRdphi);
    }
}

/** The PowerLawMultipole branch (both asymptotes: `inner` selects the exponent
    convention v = l vs v = -l-1 and the extreme-regime test), in its FUSED form:
    no Phi_lm[3*(lmax+1)^2] is materialized in the general case; each harmonic's
    coefficient triple is produced by multipolePowerLawTermT() inside the loop
    that consumes it. The one exception is the lmax==2 shortcut, where the
    consumer indexes three fixed harmonics instead of looping, so the small
    (<= 27 element) array is materialized and handed to
    sphHarmTransformInverseDerivT, which then forks to the same shortcut the CPU
    takes.

    \param[in]  ind     is the asymptote's own indexing scheme (NOT the Multipole's:
                PowerLawMultipole derives it from the nonzero pattern of U);
    \param[in]  inner   selects inward vs outward extrapolation;
    \param[in]  r0sq, Q are the squared reference radius and the extra r^2 coefficient;
    \param[in]  S, U, W are ind.size()-long coefficient arrays inside the blob;
    \param[in]  R, z, phi is the position; scratch as documented above;
    \param[out] potential, gradCyl, hessCyl  are the T outputs (any may be NULL).
*/
template<typename T>
AGAMA_DEVICE_INLINE void multipolePowerLawEvalDeviceT(
    const math::SphHarmIndicesPod& ind, const bool inner,
    const T r0sq, const T Q, const T* S, const T* U, const T* W,
    const T R, const T z, const T phi, T* scratch,
    T* potential, T* gradCyl, T* hessCyl)
{
    const bool needGrad = gradCyl!=NULL || hessCyl!=NULL;
    const bool needHess = hessCyl!=NULL;
    const int ncoefs = ind.size();
    const T rsq   = pow_2(R) + pow_2(z);
    const T dlogr = std::log(rsq / r0sq) * T(0.5);
    // simplified treatment in strongly asymptotic regime - retain only l==0 term
    const int lmax = (inner && rsq < r0sq*T(1e-16)) || (!inner && rsq > r0sq*T(1e16)) ? 0 : ind.lmax;

    if(lmax == 0) {  // fast track
        T p0 = 0, p1 = 0, p2 = 0;
        multipolePowerLawTermT<T>(S[0], U[0], W[0], inner ? T(0) : T(-1), Q, rsq, r0sq, dlogr,
            needGrad, needHess, &p0, &p1, &p2);
        if(potential)
            *potential = p0;
        const T rsqinv = rsq>0 ? T(1)/rsq : T(0),
            Rr2 = rsq<INFINITY ? R * rsqinv : T(0),
            zr2 = rsq<INFINITY ? z * rsqinv : T(0);
        if(gradCyl) {
            gradCyl[CYL_DR]   = p1 * Rr2;
            gradCyl[CYL_DZ]   = p1 * zr2;
            gradCyl[CYL_DPHI] = 0;
        }
        if(hessCyl) {
            const T d2 = p2 - 2 * p1;
            hessCyl[CYL_DR2]  = d2 * pow_2(Rr2) + p1 * rsqinv;
            hessCyl[CYL_DZ2]  = d2 * pow_2(zr2) + p1 * rsqinv;
            hessCyl[CYL_DRDZ] = d2 * Rr2 * zr2;
            hessCyl[CYL_DRDPHI] = hessCyl[CYL_DZDPHI] = hessCyl[CYL_DPHI2] = 0;
        }
        return;
    }

    T g[3], h[6];
    if(multipoleDeriv2Shape(ind)) {
        // the consumer is the {0,0}/{2,0}/{2,2} shortcut, which random-accesses three
        // harmonics rather than looping, so fusing does not apply: materialize the
        // (at most 27-element) coefficient array, exactly as the CPU body does.
        T*   Phi_lm = scratch;
        T*  dPhi_lm = Phi_lm + ncoefs;
        T* d2Phi_lm = Phi_lm + ncoefs*2;
        for(int m=ind.mmin(); m<=ind.mmax; m++)
            for(int l=ind.lmin(m); l<=lmax; l+=ind.step) {
                const int c = math::SphHarmIndicesPod::index(l, m);
                multipolePowerLawTermT<T>(S[c], U[c], W[c], inner ? T(l) : T(-l-1),
                    Q, rsq, r0sq, dlogr, needGrad, needHess,
                    &Phi_lm[c], &dPhi_lm[c], &d2Phi_lm[c]);
            }
        sphHarmTransformInverseDerivT<T>(ind, R, z, phi, Phi_lm, dPhi_lm, d2Phi_lm,
            scratch + ncoefs*3, potential, needGrad ? g : NULL, needHess ? h : NULL);
    } else {
        // the fused general case: sphHarmTransformInverseDerivT's loop with the
        // power-law coefficients computed where they are used
        const int numQuantities = needHess ? 6 : needGrad ? 3 : 1;
        const int sizeC = 6 * (2*ind.mmax+1), sizeP = ind.lmax+1;
        T*   C_m  = scratch;
        T*   P_lm = C_m + sizeC;
        T*  dP_lm = numQuantities>=3 ? P_lm + sizeP   : NULL;
        T* d2P_lm = numQuantities==6 ? P_lm + sizeP*2 : NULL;
        T* trig_m = P_lm + sizeP*3;
        const T tau = z == 0 ? T(0) : z / (std::sqrt(pow_2(R) + pow_2(z)) + R);
        const int nm = ind.mmax - ind.mmin() + 1;
        for(int mm=0; mm<nm; mm++) {
            const int m = mm + ind.mmin();
            const int lmin = ind.lmin(m);
            if(lmin > ind.lmax)
                continue;
            const T mul = m==0 ? T(2*M_SQRTPI) : T(2*M_SQRTPI*M_SQRT2);
            for(int q=0; q<numQuantities; q++)
                C_m[mm + q*nm] = 0;
            const int absm = m<0 ? -m : m;
            math::sphHarmArray<T>(ind.lmax, absm, tau, P_lm, dP_lm, d2P_lm);
            for(int l=lmin; l<=ind.lmax; l+=ind.step) {
                const int c = math::SphHarmIndicesPod::index(l, m), p = l-absm;
                T p0 = 0, p1 = 0, p2 = 0;
                multipolePowerLawTermT<T>(S[c], U[c], W[c], inner ? T(l) : T(-l-1),
                    Q, rsq, r0sq, dlogr, numQuantities>=3, numQuantities==6, &p0, &p1, &p2);
                C_m[mm] += P_lm[p] * p0 * mul;
                if(numQuantities>=3) {
                    C_m[mm + nm  ] +=  P_lm[p] * p1 * mul;
                    C_m[mm + nm*2] += dP_lm[p] * p0 * mul;
                }
                if(numQuantities==6) {
                    C_m[mm + nm*3] +=   P_lm[p] * p2 * mul;
                    C_m[mm + nm*4] +=  dP_lm[p] * p1 * mul;
                    C_m[mm + nm*5] += d2P_lm[p] * p0 * mul;
                }
            }
        }
        fourierTransformAzimuthT<T>(ind, phi, C_m, trig_m, potential,
            needGrad ? g : NULL, needHess ? h : NULL);
    }
    if(needGrad) {
        coord::GradSph gradSph = {0, 0, 0};
        coord::HessSph hessSph = {0, 0, 0, 0, 0, 0};
        multipoleUnpackSph<T>(g, needHess ? h : NULL, &gradSph, &hessSph);
        coord::GradCyl gc = {0, 0, 0};
        coord::HessCyl hc = {0, 0, 0, 0, 0, 0};
        sphToCylDerivs(coord::PosCyl(R, z, phi), gradSph, hessSph,
            gradCyl ? &gc : NULL, hessCyl ? &hc : NULL);
        multipolePackCyl<T>(gc, hc, gradCyl, hessCyl);
    }
}

/** The MultipoleInterp1d branch (taken when lmax <= LMAX_1D_SPLINE = 2): one 1d
    quintic spline in ln(r) per {l,m}, optional log-scaling of the l=0 term with
    the other terms stored as ratios to it, then the inverse harmonic transform.

    \param[in]  ind, logScaling, invPhi0  are the interpolator's own parameters;
    \param[in]  xval, nx   is the shared ln(r) knot vector;
    \param[in]  coefs      points at the descriptor's coefficient block: ind.size()
                slots of 3*nx, slot c = index(l,m) holding {fval, fder, fder2};
    \param[in]  R, z, phi, scratch, and the outputs, as in multipolePowerLawEvalDeviceT.
*/
template<typename T>
AGAMA_DEVICE_INLINE void multipoleInterp1dEvalDeviceT(
    const math::SphHarmIndicesPod& ind, const int logScaling, const T invPhi0,
    const T* xval, const int nx, const T* coefs,
    const T R, const T z, const T phi, T* scratch,
    T* potential, T* gradCyl, T* hessCyl)
{
    const bool needGrad = gradCyl!=NULL || hessCyl!=NULL;
    const bool needHess = hessCyl!=NULL;
    const T r = std::sqrt(pow_2(R) + pow_2(z)), logr = std::log(r);
    const int ncoefs = (ind.lmax+1) * (ind.lmax+1);
    T*   Phi_lm = scratch;
    T*  dPhi_lm = Phi_lm + ncoefs;
    T* d2Phi_lm = Phi_lm + ncoefs*2;
    coord::GradSph gradSph = {0, 0, 0};
    coord::HessSph hessSph = {0, 0, 0, 0, 0, 0};

    // first compute the l=0 coefficient, possibly log-unscaled
    math::evalQuinticSplineRaw<T>(logr, xval, coefs, coefs+nx, coefs+nx*2, nx,
        Phi_lm, needGrad ? dPhi_lm : NULL, needHess ? d2Phi_lm : NULL);
    if(logScaling) {
        const T expX = std::exp(Phi_lm[0]), Phi = T(1) / (invPhi0 - expX);
        Phi_lm[0] = Phi;
        if(needGrad) {
            const T dPhidX = pow_2(Phi) * expX;
            if(needHess)
                d2Phi_lm[0] = dPhidX * (d2Phi_lm[0] + pow_2(dPhi_lm[0]) * Phi * (invPhi0 + expX));
            dPhi_lm[0] *= dPhidX;
        }
    }
    if(ind.lmax == 0) {   // fast track in the spherical case
        if(potential)
            *potential = Phi_lm[0];
        if(needGrad) {
            gradSph.dr = dPhi_lm[0];
            gradSph.dtheta = gradSph.dphi = 0;
        }
        if(needHess) {
            hessSph.dr2 = d2Phi_lm[0];
            hessSph.dtheta2 = hessSph.dphi2 = hessSph.drdtheta =
                hessSph.drdphi = hessSph.dthetadphi = 0;
        }
    } else {
        // compute spherical-harmonic coefs
        for(int m=ind.mmin(); m<=ind.mmax; m++)
            for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
                const int c = math::SphHarmIndicesPod::index(l, m);
                if(c==0)
                    continue;
                const T* sc = coefs + c*3*nx;
                math::evalQuinticSplineRaw<T>(logr, xval, sc, sc+nx, sc+nx*2, nx,
                    &Phi_lm[c], needGrad ? &dPhi_lm[c] : NULL, needHess ? &d2Phi_lm[c] : NULL);
                // if necessary, scale by the value of l=0 coef
                if(logScaling) {
                    if(needHess)
                        d2Phi_lm[c] = d2Phi_lm[c] * Phi_lm[0] + 2 * dPhi_lm[c] * dPhi_lm[0] +
                            Phi_lm[c] * d2Phi_lm[0];
                    if(needGrad)
                        dPhi_lm[c] = dPhi_lm[c] * Phi_lm[0] + Phi_lm[c] * dPhi_lm[0];
                    Phi_lm[c] *= Phi_lm[0];
                }
            }
        T g[3], h[6];
        sphHarmTransformInverseDerivT<T>(ind, R, z, phi, Phi_lm, dPhi_lm, d2Phi_lm,
            scratch + ncoefs*3, potential, needGrad ? g : NULL, needHess ? h : NULL);
        multipoleUnpackSph<T>(needGrad ? g : NULL, needHess ? h : NULL, &gradSph, &hessSph);
    }
    if(needGrad) {
        coord::GradCyl gc = {0, 0, 0};
        coord::HessCyl hc = {0, 0, 0, 0, 0, 0};
        sphToCylDerivs(coord::PosCyl(R, z, phi), gradSph, hessSph,
            gradCyl ? &gc : NULL, hessCyl ? &hc : NULL);
        multipolePackCyl<T>(gc, hc, gradCyl, hessCyl);
    }
}

/** The MultipoleInterp2d branch (lmax > 2, i.e. every realistic GalPot MW model):
    one 2d quintic spline in (ln r, tau) per azimuthal harmonic m, with all l
    folded into the tau grid -- so this branch never calls sphHarmArray at all,
    and its scratch scales with mmax rather than lmax^2.

    \param[in]  xval, nx / yval, ny  are the shared ln(r) and tau knot vectors;
    \param[in]  coefs  points at the coefficient block: 2*ind.mmax+1 slots of
                9*nx*ny, slot m+ind.mmax holding the nine node arrays in
                evalQuinticSpline2dRaw order (see MultipoleDeviceDesc);
    \param[in]  everything else as in multipoleInterp1dEvalDeviceT.
*/
template<typename T>
AGAMA_DEVICE_INLINE void multipoleInterp2dEvalDeviceT(
    const math::SphHarmIndicesPod& ind, const int logScaling, const T invPhi0,
    const T* xval, const int nx, const T* yval, const int ny, const T* coefs,
    const T R, const T z, const T phi, T* scratch,
    T* potential, T* gradCyl, T* hessCyl)
{
    const T
        r         = std::sqrt(pow_2(R) + pow_2(z)),
        logr      = std::log(r),
        rplusRinv = T(1) / (r + std::fabs(R)),
        tau       = R==0 ? math::sign(z) : z * rplusRinv;

    // number of azimuthal harmonics to compute
    const int mmin = ind.mmin(), nm = ind.mmax - mmin + 1;

    // only compute those quantities that will be needed in output
    const int numQuantities = hessCyl!=NULL ? 6 : gradCyl!=NULL ? 3 : 1;

    // Phi, two first and three second derivs for each m -- the same C_m layout the
    // CPU body allocates on the stack, here carved out of the caller's scratch. Six
    // rows are always reserved (not numQuantities rows) because multipoleUnscaleLogT
    // forms all six row pointers unconditionally, as the original body does.
    T* C_m = scratch;
    T* trig_m = C_m + nm*6;
    const int npt = nx*ny, stride = npt*9;

    // compute azimuthal harmonics
    for(int mm=0; mm<nm; mm++) {
        const int m = mm + mmin;
        if(ind.lmin(m) > ind.lmax)
            continue;
        const T* c = coefs + (m + ind.mmax) * stride;
        math::evalQuinticSpline2dRaw<T>(logr, tau, xval, yval, nx, ny,
            c, c+npt, c+npt*2, c+npt*3, c+npt*4, c+npt*5, c+npt*6, c+npt*7, c+npt*8,
            &C_m[mm],
            numQuantities>=3 ? &C_m[mm+nm  ] : NULL,
            numQuantities>=3 ? &C_m[mm+nm*2] : NULL,
            numQuantities==6 ? &C_m[mm+nm*3] : NULL,
            numQuantities==6 ? &C_m[mm+nm*4] : NULL,
            numQuantities==6 ? &C_m[mm+nm*5] : NULL);
    }

    if(logScaling)
        multipoleUnscaleLogT<T>(ind, numQuantities, invPhi0, C_m);

    // Fourier synthesis from azimuthal harmonics to actual quantities, still in scaled coords
    T trPot, g[3], h[6];
    fourierTransformAzimuthT<T>(ind, phi, C_m, trig_m, &trPot,
        numQuantities>=3 ? g : NULL, numQuantities==6 ? h : NULL);

    if(potential)
        *potential = trPot;
    if(numQuantities==1)
        return;   // nothing else needed

    coord::GradSph trGrad = {0, 0, 0};
    coord::HessSph trHess = {0, 0, 0, 0, 0, 0};
    multipoleUnpackSph<T>(g, numQuantities==6 ? h : NULL, &trGrad, &trHess);
    coord::GradCyl gc = {0, 0, 0};
    coord::HessCyl hc = {0, 0, 0, 0, 0, 0};
    // deliberately NOT sphToCylDerivs -- that one's second scaled coordinate is theta,
    // this one's is tau
    tauToCylDerivs(R, z, r, rplusRinv, tau, trGrad, trHess,
        gradCyl ? &gc : NULL, hessCyl ? &hc : NULL);
    multipolePackCyl<T>(gc, hc, gradCyl, hessCyl);
}

/** Evaluate a Multipole potential on the device from its descriptor and blob.
    Reproduces Multipole::evalCyl's four-branch dispatch on radius: inner
    power-law asymptote below rminSq, outer above rmaxSq, and the interpolator in
    between (1d or 2d splines per implKind). The branch boundaries in the
    descriptor are already squared and safety-factored, so the test here is the
    same comparison the CPU makes on the same rsq.

    \param[in]  d       is the descriptor from buildMultipoleDeviceDesc();
    \param[in]  blob    is its coefficient payload (host or device pointer);
    \param[in]  R, z, phi  is the position in cylindrical coordinates;
    \param[out] Phi     receives the potential   if != NULL;
    \param[out] gradCyl receives 3 components    if != NULL (CylGradIndex order);
    \param[out] hessCyl receives 6 components    if != NULL (CylHessIndex order);
    \param[in]  scratch is caller-provided, at least multipoleDeviceScratchSize(d)
                elements of T (or MultipoleDeviceScratchMax<ORDER>::value for a
                kernel with a compile-time order cap).

    As on the CPU path, asking for the hessian without the gradient still computes
    the gradient internally; and asking for neither takes the cheap value-only
    route through every branch.
*/
/** The `impl` (interpolated) branch of a Multipole ALONE, without the radial
    dispatch: 1d or 2d splines per implKind. Split out of multipoleEvalDevice
    below because the density path needs exactly this and not the dispatch --
    Multipole::densityCyl calls `impl->density(pos)`, which routes through
    BasePotential::densityCyl's Laplacian and therefore evaluates the INTERPOLATOR
    at a slightly displaced point when close to the z axis; that displaced point
    must stay on the interpolator even if it were to cross a branch boundary,
    exactly as it does on the CPU. Arguments and outputs as in
    multipoleEvalDevice. */
template<typename T>
AGAMA_DEVICE_INLINE void multipoleImplEvalDeviceT(const MultipoleDeviceDesc<T>& d,
    const T* blob, T R, T z, T phi, T* Phi, T* gradCyl, T* hessCyl, T* scratch)
{
    if(d.implKind == MULTIPOLE_IMPL_INTERP1D)
        multipoleInterp1dEvalDeviceT<T>(d.ind, d.logScaling, d.invPhi0,
            blob + d.offXval, d.nx, blob + d.offCoefs,
            R, z, phi, scratch, Phi, gradCyl, hessCyl);
    else
        multipoleInterp2dEvalDeviceT<T>(d.ind, d.logScaling, d.invPhi0,
            blob + d.offXval, d.nx, blob + d.offYval, d.ny, blob + d.offCoefs,
            R, z, phi, scratch, Phi, gradCyl, hessCyl);
}

template<typename T>
AGAMA_DEVICE_INLINE void multipoleEvalDevice(const MultipoleDeviceDesc<T>& d, const T* blob,
    T R, T z, T phi, T* Phi, T* gradCyl, T* hessCyl, T* scratch)
{
    const T rsq = pow_2(R) + pow_2(z);
    if(rsq < d.rminSq) {
        const int n = d.indInner.size();
        const T* SUW = blob + d.offInnerSUW;
        multipolePowerLawEvalDeviceT<T>(d.indInner, true, d.r0sqInner, d.qInner,
            SUW, SUW+n, SUW+n*2, R, z, phi, scratch, Phi, gradCyl, hessCyl);
    } else if(rsq > d.rmaxSq) {
        const int n = d.indOuter.size();
        const T* SUW = blob + d.offOuterSUW;
        multipolePowerLawEvalDeviceT<T>(d.indOuter, false, d.r0sqOuter, d.qOuter,
            SUW, SUW+n, SUW+n*2, R, z, phi, scratch, Phi, gradCyl, hessCyl);
    } else
        multipoleImplEvalDeviceT<T>(d, blob, R, z, phi, Phi, gradCyl, hessCyl, scratch);
}


// ---------------------------------------------------------------------
// Device-callable DENSITY of a Multipole. Mirrors Multipole::densityCyl's own
// three-way dispatch, which is NOT the same thing as the Laplacian of
// multipoleEvalDevice: outside the radial grid, PowerLawMultipole overrides
// densityCyl with a closed form built from the U coefficients only, precisely to
// avoid the cancellation that the Laplacian route suffers there. Reproducing the
// dispatch is therefore mandatory, not an optimization.
// ---------------------------------------------------------------------

/** The two fp64-tuned thresholds BasePotential::densityCyl uses, re-derived per
    value type (CLAUDE.md recipe item 6: a threshold inherited from upstream is
    fp64-tuned and is wrong in fp32).
      sqrtEps  -- "close to or exactly on the z axis" test, R <= |z| * sqrtEps,
                  and the size of the sideways step taken to get d2Phi/dphi2 there;
      epsRel   -- eps^(2/3), the relative size below which the sum of four
                  cancelling second derivatives is declared roundoff and the
                  density is reported as exactly zero.
    The double values are the SQRT_DBL_EPSILON / DBL_EPSILON/ROOT3_DBL_EPSILON of
    math_base.h and potential_base.cpp, restated here as literals because those
    are macros/TU-locals; tests/test_gpu_policy.cpp pins them against the
    originals so a change upstream cannot silently desynchronize this. */
template<typename T> struct MultipoleDensityEps;  // no default: unsupported T won't compile
template<> struct MultipoleDensityEps<double> {
    /// sqrt(DBL_EPSILON)
    static AGAMA_DEVICE_INLINE double sqrtEps() { return 1.4901161193847656e-08; }
    /// DBL_EPSILON / ROOT3_DBL_EPSILON = eps^(2/3) ~ 4e-11
    static AGAMA_DEVICE_INLINE double epsRel()  { return 2.2204460492503131e-16 /
                                                         6.0554544523933429e-06; }
};
template<> struct MultipoleDensityEps<float> {
    /// sqrt(FLT_EPSILON)
    static AGAMA_DEVICE_INLINE float sqrtEps() { return 3.4526698e-04f; }
    /// FLT_EPSILON / cbrt(FLT_EPSILON) = eps^(2/3) ~ 2.4e-05
    static AGAMA_DEVICE_INLINE float epsRel()  { return 1.1920929e-07f / 4.9215667e-03f; }
};

/** Device-callable form of PowerLawMultipole::densityCyl, FUSED: upstream fills
    rho_lm[ind.size()] in one (m,l) loop and consumes it in the (m,l) loop inside
    math::sphHarmTransformInverse; both loops visit the harmonics in the same
    order and each element is read exactly once, so computing it where it is used
    changes no element's defining expression and no accumulation order -- the same
    argument (and the same measured outcome: zero bitwise differences) as the
    fusion already done for the potential in multipolePowerLawEvalDeviceT. What it
    buys is scratch: the un-fused shape would need 1,089 extra T at order 32,
    nearly tripling the per-thread scratch of the whole descriptor kernel.

    \param[in]  ind    is the asymptote's own indexing scheme;
    \param[in]  inner  selects the inward (v=l) vs outward (v=-l-1) convention and
                       which extreme-regime test drops to the monopole;
    \param[in]  r0sq, Q  are the squared reference radius and the extra r^2 coefficient;
    \param[in]  S, U   are ind.size()-long coefficient arrays inside the blob (W is
                       deliberately not used: it has zero Laplacian);
    \param[in]  R, z, phi  is the position;
    \param[in]  scratch  needs 2*ind.mmax + ind.lmax + 1 elements (well inside
                       multipoleBranchScratch, so the caller's existing block serves).
    \return the mass density. */
template<typename T>
AGAMA_DEVICE_INLINE T multipolePowerLawDensityDeviceT(
    const math::SphHarmIndicesPod& ind, const bool inner,
    const T r0sq, const T Q, const T* S, const T* U,
    const T R, const T z, const T phi, T* scratch)
{
    const T rsq   = pow_2(R) + pow_2(z);
    const T dlogr = std::log(rsq / r0sq) * T(0.5);
    // simplified treatment in strongly asymptotic regime - retain only l==0 term
    const int lmax = (inner && rsq < r0sq*T(1e-16)) || (!inner && rsq > r0sq*T(1e16))
        ? 0 : ind.lmax;

    if(lmax == 0) {   // fast track - just the l=0 coef
        const T rho0 = multipolePowerLawDensityTermT<T>(S[0], U[0],
            inner ? T(0) : T(-1), Q, dlogr, 0);
        return T(0.25/M_PI) / r0sq * rho0;
    }

    // --- fused inverse spherical-harmonic transform (math::sphHarmTransformInverse)
    T* trig_m = scratch;                  // 2*mmax elements
    T* P_lm   = scratch + 2*ind.mmax;     // lmax+1 elements
    const bool useSine = ind.mmin() < 0;
    if(ind.mmax > 0)
        math::trigMultiAngle<T>(phi, ind.mmax, useSine, trig_m);
    const T tau = z == 0 ? T(0) : z / (std::sqrt(pow_2(R) + pow_2(z)) + R);
    T result = 0;
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        const int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;   // empty m-harmonic
        const int absm = m<0 ? -m : m;
        // extra numerical factors from the definition of sph.harm.
        const T trig = m==0 ? T(2*M_SQRTPI) :
            m>0 ? trig_m[m-1]            * T(2*M_SQRTPI * M_SQRT2) :
                  trig_m[ind.mmax-m-1]   * T(2*M_SQRTPI * M_SQRT2);
        math::sphHarmArray<T>(ind.lmax, absm, tau, P_lm, (T*)NULL, (T*)NULL);
        for(int l=lmin; l<=ind.lmax; l+=ind.step) {
            const int c = math::SphHarmIndicesPod::index(l, m);
            const T rho = multipolePowerLawDensityTermT<T>(S[c], U[c],
                inner ? T(l) : T(-l-1), Q, dlogr, l);
            const T leg = P_lm[l-absm];
            result += rho * leg * trig;
        }
    }
    return T(0.25/M_PI) / r0sq * result;
}

/** Device-callable form of Multipole::densityCyl: the same three-way radial
    dispatch, with the interpolated branch going through
    BasePotential::densityCyl's cylindrical Laplacian of the INTERPOLATOR (which is
    what `impl->density(pos)` resolves to -- MultipoleInterp1d/2d do not override
    densityCyl) and the two asymptotes through PowerLawMultipole's closed form.

    \param[in]  d, blob  as in multipoleEvalDevice;
    \param[in]  R, z, phi  is the position in cylindrical coordinates;
    \param[in]  scratch  as in multipoleEvalDevice (the Laplacian route needs the
                full grad+hess scratch, so the same bound applies).
    \return the mass density. */
template<typename T>
AGAMA_DEVICE_INLINE T multipoleDensityDevice(const MultipoleDeviceDesc<T>& d,
    const T* blob, T R, T z, T phi, T* scratch)
{
    const T rsq = pow_2(R) + pow_2(z);
    if(rsq < d.rminSq) {
        const int n = d.indInner.size();
        const T* SUW = blob + d.offInnerSUW;
        return multipolePowerLawDensityDeviceT<T>(d.indInner, true, d.r0sqInner,
            d.qInner, SUW, SUW+n, R, z, phi, scratch);
    }
    if(rsq > d.rmaxSq) {
        const int n = d.indOuter.size();
        const T* SUW = blob + d.offOuterSUW;
        return multipolePowerLawDensityDeviceT<T>(d.indOuter, false, d.r0sqOuter,
            d.qOuter, SUW, SUW+n, R, z, phi, scratch);
    }
    // --- BasePotential::densityCyl on the interpolator, transcribed
    T grad[3], hess[6];
    multipoleImplEvalDeviceT<T>(d, blob, R, z, phi, (T*)NULL, grad, hess, scratch);
    const T eps = MultipoleDensityEps<T>::sqrtEps();
    T derivR_over_R     = grad[CYL_DR]    / R;
    T deriv2phi_over_R2 = hess[CYL_DPHI2] / pow_2(R);
    if(R <= std::fabs(z) * eps) {   // close to or exactly on the z axis
        derivR_over_R = hess[CYL_DR2];
        if(isZRotSymmetric(static_cast<coord::SymmetryType>(d.ind.sym)))
            deriv2phi_over_R2 = 0;   // d2Phi/dphi2 is always zero in this case
        else {
            // to compute d2Phi/dphi2, we need to step out of z axis just a tiny bit
            T hessoff[6];
            const T Roff = std::fabs(z) * eps;
            multipoleImplEvalDeviceT<T>(d, blob, Roff, z, phi,
                (T*)NULL, (T*)NULL, hessoff, scratch);
            deriv2phi_over_R2 = hessoff[CYL_DPHI2] / pow_2(Roff);
        }
    }
    const T result = hess[CYL_DR2] + derivR_over_R + hess[CYL_DZ2] + deriv2phi_over_R2;
    if(!(std::fabs(result) > MultipoleDensityEps<T>::epsRel() *
        (std::fabs(hess[CYL_DR2]) + std::fabs(derivR_over_R) +
         std::fabs(hess[CYL_DZ2]) + std::fabs(deriv2phi_over_R2))))
        return 0;   // dominated by roundoff errors
    return result / T(4*M_PI);
}


/** Spherical-harmonic expansion of density with coefficients being spline functions of radius */
class DensitySphericalHarmonic: public BaseDensity {
public:
    /** Construct the density interpolator from the provided density profile and grid parameters.
        This is not a constructor, but a static method returning a shared pointer to
        the newly created density object.
        \param[in]  src        is the input density model;
        \param[in]  sym        is the required symmetry of the density expansion
        (if set to ST_UNKNOWN, will be taken from the input density model);
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect;
        \param[in]  fixOrder   whether to limit the order of the internal sph.-harm. expansion
        of the input density to the output order; if false (default), it may be higher
        (use a larger number of grid points in angles) to improve accuracy.
    */
    static shared_ptr<const DensitySphericalHarmonic> create(
        const BaseDensity& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0, double rmax = 0,
        bool fixOrder = false);

    /** Construct the density interpolator from an N-body snapshot.
        This is not a constructor, but a static method returning a shared pointer to
        the newly created density object.
        \param[in]  particles  is the array of particles.
        \param[in]  sym        is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect.
        \param[in]  smoothing  is the amount of smoothing applied during penalized spline fitting.
        \note OpenMP-parallelized loops over particles and over expansion coefficients.
    */
    static shared_ptr<const DensitySphericalHarmonic> create(
        const particles::ParticleArray<coord::PosCyl> &particles,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0., double smoothing = 1.);

    /** Construct the object from previously computed coefficients.
        \param[in]  gridRadii  is the grid in radius (sorted in order of increase, first node > 0).
        \param[in]  coefs  is the 2d array of sph.-harm. coefficients:
        the first dimension of the array is the number of spherical harmonics (lmax+1)^2,
        and the second dimension is the number of radial grid points.
        If all coefficients of the l=0 harmonic are positive, a logarithmic scaling
        for this harmonic will be employed, and all l!=0 terms will be scaled relative to the l=0 term.
        After this optional scaling, all harmonic coefficients are spline-interpolated in log radius.
    */
    DensitySphericalHarmonic(const std::vector<double> &gridRadii,
        const std::vector< std::vector<double> > &coefs);

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "DensitySphericalHarmonic"; }

    /** return the radii of spline nodes */
    const std::vector<double>& getRadii() const { return gridRadii; }

    /** return the radii of spline nodes and the array of density expansion coefficients.
        \param[out]  radii  will contain the grid radii.
        \param[out]  coefsArray  will be filled with the array of coefficients at these radii.
    */
    void getCoefs(std::vector<double> &radii, std::vector< std::vector<double> > &coefsArray) const;

    /** fill the array of density expansion coefficients at a given radius.
        \param[out]  coefs  is a pre-allocated array of size ind.size(), which will be filled
        by this routine, but its elements corresponding to empty harmonics are left uninitialized.
    */
    void getCoefsAtRadius(double radius, double coefs[]) const;

    /** return the size of coefficients array, to be used for pre-allocating the output storage in
        getCoefsAtRadii() */
    inline unsigned int getCoefsSize() const { return ind.size(); }

private:
    /// radial grid
    const std::vector<double> gridRadii;

    /// indexing scheme for sph.-harm. coefficients
    const math::SphHarmIndices ind;

    /// radial dependence of each sph.-harm. expansion term
    std::vector<math::PtrFunction> spl;

    /// logarithmic density slope at small and large radii (rho ~ r^s)
    /// and the extrapolated value at origin
    double innerSlope, outerSlope, centralValue;

    /// whether the l=0 term is interpolated using log-scaling
    bool logScaling;

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    // the actual implementation
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;

};  // class DensitySphericalHarmonic


/** Auxiliary potential class that represents an array of multipole terms,
    with each term being a sum of two power-law profiles.
    It is internally used by spherical-harmonic and azimuthal Fourier expansion potential classes,
    as an extrapolation to small or large radii beyond the definition region of the main potential.
    Each term with index {l,m} is given by
    \f$  \Phi_{l,m}(r) = W_{l,m} * (r/r0)^v + U_{l,m} * (r/r0)^{s_{l,m}} + Q * (r/r0)^2 \f$  if s!=v,
    \f$  \Phi_{l,m}(r) = W_{l,m} * (r/r0)^v + U_{l,m} * (r/r0)^{s_{l,m}} * ln(r/r0)     \f$  if s==v.
    Here v=l for the inward extrapolation and v=-1-l for the outward extrapolation,
    so that the term W r^v represents the 'main' multipole component corresponding 
    to the Laplace equation, i.e. with zero density. The other term U r^s corresponds
    to a power-law density profile of the given harmonic component (rho ~ r^{s-2}),
    and the term Q r^2, optionally present only for v=0 and s>2 (i.e. for monopole at small radii),
    allows one to represent a density profile with a finite central limit and radial gradient.
*/
class PowerLawMultipole: public BasePotentialCyl {
public:
    /** Create the potential from the three arrays: amplitudes of harmonic coefficients (U, W)
        the power-law slope of the coefficient U with nonzero Laplacian (S),
        the optional extra coefficient Q describing a profile with a finite central density,
        the reference radius r0 and the flag choosing between inward and outward extrapolation */
    PowerLawMultipole(double r0, bool inner,
        const std::vector<double>& S,
        const std::vector<double>& U,
        const std::vector<double>& W,
        const double Q = 0);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual std::string name() const { return "PowerLaw"; }
private:
    const math::SphHarmIndices ind;    ///< indexing scheme for sph.-harm.coefficients
    const double r0sq;                 ///< reference radius, squared
    const bool inner;                  ///< whether this is an inward or outward extrapolation
    const std::vector<double> S, U, W; ///< sph.-harm.coefficients for extrapolation
    const double Q;                    ///< extra coefficient for the inward extrapolation of l=0 term

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const;

    /// re-implement the density computation to avoid cancellation errors at large radii,
    /// by using only the U-terms which have non-zero Laplacian
    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const;

    /// the device-descriptor builder reads ind/r0sq/inner/S/U/W/Q directly rather than
    /// widening this class's public surface with accessors that nothing else would use
    template<typename T> friend bool buildMultipoleDeviceDesc(const Multipole& pot,
        MultipoleDeviceDesc<T>& desc, std::vector<T>& blob);
};


/// Multipole expansion for potentials
class Multipole: public BasePotentialCyl{
public:
    /** create the potential from the analytic density or potential model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential.
        It exists in two variants: the first one takes a density model as input
        and solves Poisson equation to find the potential sph.-harm. coefficients;
        the second one takes a potential model and computes these coefs directly.
        \param[in]  src        is the input density or potential model;
        \param[in]  sym        is the required symmetry of the potential
        (if set to ST_UNKNOWN, will be taken from the input density model);
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect;
        \param[in]  fixOrder   whether to limit the order of the internal sph.-harm. expansion
        of the input density to the output order; if false (default), it may be higher
        (use a larger number of grid points in angles) to improve accuracy.
    */
    static shared_ptr<const Multipole> create(
        const BaseDensity& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0, double rmax = 0,
        bool fixOrder = false);

    /** same as above, but takes a potential model as an input */
    static shared_ptr<const Multipole> create(
        const BasePotential& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0, double rmax = 0,
        bool fixOrder = false);

    /** create the potential from an N-body snapshot.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential.
        \param[in]  particles  is the array of particles.
        \param[in]  sym  is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect.
        \param[in]  smoothing  is the amount of smoothing applied during penalized spline fitting.
        \note OpenMP-parallelized loops over particles and over expansion coefficients.
    */
    static shared_ptr<const Multipole> create(
        const particles::ParticleArray<coord::PosCyl> &particles,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0., double smoothing = 1.);

    /** construct the potential from the set of spherical-harmonic coefficients.
        \param[in]  radii  is the grid in radius;
        \param[in]  Phi  is the matrix of harmonic coefficients for the potential;
                    its first dimension is the number of coefficients (lmax+1)^2,
                    and the second is the number of radial grid points;
        \param[in]  dPhi  is the matrix of radial derivatives of harmonic coefs
                    (same size as Phi, each element is  d Phi_{l,m}(r) / dr ).
    */
    Multipole(const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);

    /** return the radii of spline nodes */
    const std::vector<double>& getRadii() const { return gridRadii; }

    /** return the array of spherical-harmonic expansion coefficients.
        \param[out] radii will contain the radii of grid nodes;
        \param[out] Phi   will contain the spherical-harmonic expansion coefficients
                    for the potential at the given radii;
        \param[out] dPhi  will contain the radial derivatives of these coefs.
    */
    void getCoefs(std::vector<double> &radii,
        std::vector<std::vector<double> > &Phi,
        std::vector<std::vector<double> > &dPhi) const;

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Multipole"; }
    virtual double enclosedMass(const double radius) const;

private:
    /// radial grid
    const std::vector<double> gridRadii;

    /// indexing scheme for sph.-harm. coefficients
    const math::SphHarmIndices ind;

    /// actual potential implementation (based either on 1d or 2d interpolating splines)
    PtrPotential impl;

    /// asymptotic behaviour at small and large radii described by `PowerLawMultipole`
    PtrPotential asymptInner, asymptOuter;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const;

    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const;

    /// the device-descriptor builder walks gridRadii / ind / impl / asymptInner / asymptOuter;
    /// it is defined in potential_multipole.cpp, the only place MultipoleInterp1d and
    /// MultipoleInterp2d are visible
    template<typename T> friend bool buildMultipoleDeviceDesc(const Multipole& pot,
        MultipoleDeviceDesc<T>& desc, std::vector<T>& blob);
};


/// Basis-set expansion for potentials using the Zhao(1996) basis set, possibly time-dependent
class BasisSet: public BasePotentialSph{
public:
    /** create the potential from the analytic density or potential model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential.
        \param[in]  src   is the input density or potential model;
        \param[in]  lmax  is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax  is the order of expansion in azimuth (phi);
        \param[in]  nmax  is the order of radial expansion (number of terms is nmax+1);
        \param[in]  eta   is the shape parameter of basis functions;
        \param[in]  r0    is the scale radius of basis functions (0 means auto-detect);
        \param[in]  fixOrder  whether to limit the order of the internal sph.-harm. expansion
        of the input density to the output order; if false (default), it may be higher
        (use a larger number of grid points in angles) to improve accuracy.
    */
    static shared_ptr<const BasisSet> create(
        const BaseDensity& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int nmax, double eta=1.0, double r0=0.0,
        bool fixOrder=false);

    /** create the potential from an N-body snapshot.
        \param[in]  particles  is the array of particles.
        \param[in]  sym  is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax  is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax  is the order of expansion in azimuth (phi);
        \param[in]  nmax  is the order of radial expansion (number of terms is nmax+1);
        \param[in]  eta   is the shape parameter of basis functions;
        \param[in]  r0    is the scale radius of basis functions (0 means auto-detect).
        \note OpenMP-parallelized loop over particles.
    */
    static shared_ptr<const BasisSet> create(
        const particles::ParticleArray<coord::PosCyl> &particles,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int nmax, double eta=1.0, double r0=0.0);

    /** construct a possibly time-dependent potential from the set of basis-set expansion coefficients.
        If only one block of coefficients is provided, the potential is time-independent, otherwise
        it is linearly interpolated in time and extrapolated as a constant.
        \param[in]  eta  is the shape parameter of basis functions
        (0.5 for Clutton-Brock, 1 for Hernquist-Ostriker, values between 1 and 2 provide best results),
        the 0th order function has the 'Spheroid' (Zhao) double-power-law density profile with
        transition steepness alpha=1/eta, outer slope beta=3+1/eta, and inner slope gamma=2-1/eta.
        \param[in]  r0   is the scale radius of basis functions
        (typically should be comparable to half-mass radius).
        \param[in] coefs is the pointer to the first element of an array of time-dependent coefficients,
        their number specified by the length of the timestamps vector, or if the latter is empty,
        this implies that only one block of coefficients is used and the potential is time-independent.
        Each block corresponding to one timestamp (or the only block for a time-independent potential)
        is itself a nested 2d array with the first dimension being the number of spherical-harmonic
        coefficiets (lmax+1)^2, and the second dimension being the number of radial basis functions
        nmax+1.
        \param[in] timestamps  is the array of timestamps, which may be empty.
    */
    BasisSet(double eta, double r0, const std::vector<std::vector<double> > coefs[],
        const std::vector<double> timestamps = std::vector<double>());

    /** return the array of basis-set expansion coefficients.
        \param[out] eta   will contain the shape parameter of basis functions;
        \param[out] r0    will contain the scale radius of basis functions;
        \param[out] coefs will contain the coefficients (one nested 2d array for each timestamp,
        or just one 2d array if the potential is not time-dependent).
        \param[out] timestamps  will contain corresponding timestamps
        (empty if the potential is not time-dependent).
    */
    void getCoefs(double& eta, double& r0,
        std::vector< std::vector<std::vector<double> > > &coefs,
        std::vector<double> &timestamps) const;

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "BasisSet"; }

private:
    const std::vector<double> timestamps;

    const std::vector<std::vector<std::vector<double> > > coefs;  ///< arrays of expansion coefficients

    const math::SphHarmIndices ind;  ///< indexing scheme for sph.-harm. coefficients

    const double eta;  ///< shape parameter of basis functions

    const double r0;   ///< scale radius of basis functions

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double /*time*/) const;

    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const;
};

}  // namespace
