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
