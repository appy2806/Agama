/** \file    potential_cylspline.h
    \brief   density and potential approximations based on 2d spline in cylindrical coordinates
    \author  Eugene Vasiliev
    \date    2014-2024

    The classes and routines in this file deal with a generic way of representing
    an arbitrary density or potential profile as a 1d expansion + 2d interpolation:
    the azimuthal dependence for a non-axisymmetric profile is represented by 
    a Fourier expansion in phi angle, with the coefficients of expansion being
    2d interpolated functions in meridional plane (R,z).

    The coefficients of expansion are stored as arrays of m-th harmonic terms
    (cos(m phi), sin(m phi), 0 <= m <= mmax), where each term is a 2d matrix
    of coefficients, with two additional arrays defining grids in R and z directions.
    The density interpolator uses one set of coefficients (the value of density),
    while the potential uses three - the value of potential and its R- and z-derivatives.

    Standalone routines compute the coefficients of expansion for density or potential,
    the latter case - in three variants: from the provided potential, from a density
    model via the solution of Poisson equation in cylindrical coordinates, or from
    an array of point masses (again solving the Poisson equation); these two last
    options are quite expensive computationally.

    Once computed, the coefficients may be used to construct instances of interpolator
    classes, and stored/loaded by `writePotential`/`readPotential` routines from
    potential_factory.h
*/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_base.h"     // pow_2 / pow_3, and AGAMA_DEVICE_INLINE via gpu_device.h
#include "math_linalg.h"
#include "smart.h"
#include <cmath>

namespace potential {

/** \name  Device-callable leaf math for CylSpline evaluation
    \{
    The arithmetic half of CylSpline::evalCyl(), extracted so that the CPU virtual method
    and (later) the GPU device path call ONE transcription of the formulas -- CLAUDE.md
    constraint 5, refereed by crosscheck_expansions.py.

    THE SHAPE IS DICTATED BY potential_multipole.h:71-83 -- relocate a COMPLETE unit,
    never split one -- so ONE leaf holds the whole chain from the Jacobian to the final
    store, grad/hess never leave its frame, and the caller supplies the per-harmonic
    spline results through scratch (convention (1) there).

    MEASURED CAVEAT, and it is a real deviation from this project's usual standard: this
    extraction does NOT reach bit-for-bit parity on the CPU path. crosscheck_expansions.py's
    fork-to-fork arm moves from 0.000e+00 to 2.715e-16 -- CylSpline density on every model,
    plus force on the triaxial one, at ~1 ULP (worst absolute 1.1e-13 on a density of order
    1e+2). The fork-vs-production arm is unchanged at 1.113e-07.

    It is FMA re-association, confirmed in the assembly: same 27 fused multiply-adds before
    and after, with one vmulsd+vfmadd231sd becoming a vfmadd213sd. It is not fixable by
    arranging the leaves differently -- three structures were built and measured (six small
    leaves; leaves templated on the output struct so the host writes its coord:: structs
    directly; and this single complete unit) and all three give the SAME differing arrays at
    the SAME magnitudes. Forcing inlining does not change it either.

    The reason it is unavoidable here, and the reason Multipole's equivalent moves WERE
    exact: Multipole's leaves were relocations of helpers that were already separate
    functions in potential_multipole.cpp, so the frame boundary existed before the move and
    the move preserved it. CylSpline::evalCyl is monolithic -- there is no internal boundary
    to preserve -- so any extraction at all introduces one, and gcc then contracts one
    hessian expression differently. The alternative is two transcriptions of the formulas,
    which CLAUDE.md constraint 5 forbids outright.

    What stays with the caller is exactly what cannot be device code: the asinh bounds
    test and the container/virtual spline lookups.

    Precision note: these leaves carry no fp64 residue -- the in-grid arithmetic touches
    no coord:: conversion machinery, only plain scalars. The surrounding path still does,
    twice over, and both must be dealt with before T=float is meaningful for CylSpline:
    the 2d interpolator lookups are fp64 (math::Interpolator2d), and the OUT-OF-GRID
    branch delegates to PowerLawMultipole, whose sphToCylDerivs is fp64 by construction
    (potential_multipole.h:271-273). The raw templated 2d evaluators needed to fix the
    first are already in place -- evalCubicSpline2dRaw (math_spline.h:341) and
    evalQuinticSpline2dRaw (math_spline.h:488), cubic and quintic respectively, since
    CylSpline picks between them by the object-wide `haveDerivs` flag.
*/

/// grad of a scalar field in cylindrical coordinates, templated on the value type.
/// Member ORDER deliberately matches coord::GradCyl so the two are interchangeable.
template<typename T> struct CylSplineGrad { T dR, dz, dphi; };

/// hessian of a scalar field in cylindrical coordinates, templated on the value type.
/// Member ORDER deliberately matches coord::HessCyl (dRdz, dzdphi, dRdphi -- note the
/// last two, which are easy to transpose) so the two are interchangeable.
template<typename T> struct CylSplineHess { T dR2, dz2, dphi2, dRdz, dzdphi, dRdphi; };

/// component order of the per-harmonic scratch handed to cylsplineEvalInGrid(): for each
/// of the 2*mmax+1 slots, six consecutive entries in this order. Call sites index
/// positionally -- do not reorder.
enum CylSplineHarmIndex {
    CYLSPL_PHI = 0, CYLSPL_DR = 1, CYLSPL_DZ = 2,
    CYLSPL_DR2 = 3, CYLSPL_DRDZ = 4, CYLSPL_DZ2 = 5,
    CYLSPL_NHARM = 6
};

/** The whole in-grid evaluation, from the coordinate Jacobian to the final store.

    This is one unit on purpose; see the note above. It reproduces everything
    CylSpline::evalCyl does after its spline lookups: the asinh Jacobian, the optional
    log-scaling un-transform of the m=0 term, the axisymmetric short-circuit, the
    azimuthal Fourier synthesis, and the final un-scaling -- with the expressions
    character-identical to upstream and in upstream's order.

    \param[in]  R, z, Rscale  are the unscaled position and the scaling radius;
    \param[in]  mmax          is (spl.size()-1)/2, the achieved azimuthal order;
    \param[in]  logScaling    selects the m=0 log un-transform and the fused final block;
    \param[in]  phi           is the azimuthal angle;
    \param[in]  present       is a bitmask over slots mm=0..2*mmax: bit set iff that
                harmonic has a spline. Slot mmax (m=0) is handled separately and its bit
                is ignored. A mask rather than zero-filling, so that an absent harmonic
                contributes nothing at all -- zero-filling would still add +-0 into
                grad.dphi and can flip the sign of a zero result;
    \param[in]  m0            holds the SIX m=0 quantities in CylSplineHarmIndex order,
                straight from the spline, BEFORE any log un-scaling (this leaf does it);
    \param[in]  harm          holds CYLSPL_NHARM entries per slot, slot mm at
                harm[mm*CYLSPL_NHARM + ...], again straight from the spline;
    \param[in]  trig          holds trigMultiAngle's output for this phi: cos(m phi) at
                [m-1] for m=1..mmax, then sin(m phi) at [mmax+m-1]. Caller-provided
                scratch of 2*mmax entries (see below); may be NULL when mmax==0;
    \param[out] val, der, der2  may each be NULL; GradT/HessT are coord::GradCyl and
                coord::HessCyl on the host, CylSplineGrad<T>/CylSplineHess<T> on device.

    SCRATCH SIZE -- 2*mmax, unconditionally, NOT mmax*(1+needSine). The dtrig expression
    below is evaluated without a needGrad guard (upstream does the same) and for m>0 it
    indexes trig[mmax+m-1], i.e. up to 2*mmax-1. Upstream's narrower alloca was therefore
    read past its end on any value-only query of a y-reflection-symmetric model -- the
    ordinary `potential(x,y,z)` call on a bar. The value is discarded when needGrad is
    false, so nothing computed ever changed; only the out-of-bounds access is removed.
    (trigMultiAngle still leaves the upper half uninitialised when needSine is false, so
    that discarded read is of an indeterminate value -- in bounds, but not yet clean.) */
template<typename T, typename GradT, typename HessT>
AGAMA_DEVICE_INLINE void cylsplineEvalInGrid(
    const T R, const T z, const T Rscale, const int mmax, const bool logScaling,
    const unsigned int* present,
    const T* m0, const T* harm, const T* trig,
    T* val, GradT* der, HessT* der2)
{
    const bool needGrad = der !=NULL || der2!=NULL;
    const bool needHess = der2!=NULL;

    T dRscaleddR   = T(1) / std::sqrt(pow_2(R) + pow_2(Rscale));
    T dzscaleddz   = T(1) / std::sqrt(pow_2(z) + pow_2(Rscale));
    T d2RscaleddR2 = -R * pow_3(dRscaleddR);
    T d2zscaleddz2 = -z * pow_3(dzscaleddz);

    // Read only what the caller actually filled. evalDeriv is passed NULL for the slots
    // that are not needed, so the rest are indeterminate; upstream never read them
    // because every use sat inside the same if(der)/if(der2) guards. Reading them
    // unconditionally -- e.g. into by-value parameters -- would be UB on the commonest
    // call of all, a value-only potential(x,y,z).
    T Phi0       = m0[CYLSPL_PHI];
    T dPhi0dR    = needGrad ? m0[CYLSPL_DR]   : T(0);
    T dPhi0dz    = needGrad ? m0[CYLSPL_DZ]   : T(0);
    T d2Phi0dR2  = needHess ? m0[CYLSPL_DR2]  : T(0);
    T d2Phi0dRdz = needHess ? m0[CYLSPL_DRDZ] : T(0);
    T d2Phi0dz2  = needHess ? m0[CYLSPL_DZ2]  : T(0);

    if(logScaling) {
        Phi0 = -std::exp(Phi0);
        if(needHess) {
            d2Phi0dR2  = Phi0 * (d2Phi0dR2  + pow_2(dPhi0dR));
            d2Phi0dRdz = Phi0 * (d2Phi0dRdz + dPhi0dR * dPhi0dz);
            d2Phi0dz2  = Phi0 * (d2Phi0dz2  + pow_2(dPhi0dz));
        }
        if(needGrad) {
            dPhi0dR *= Phi0;
            dPhi0dz *= Phi0;
        }
    }

    // if the potential is axisymmetric, skip the Fourier transform and amplitude scaling
    if(mmax==0) {
        if(val)
            *val = Phi0;
        if(der) {
            der->dR   = dPhi0dR * dRscaleddR;
            der->dz   = dPhi0dz * dzscaleddz;
            der->dphi = 0;
        }
        if(der2) {
            der2->dR2 = d2Phi0dR2 * pow_2(dRscaleddR) + dPhi0dR * d2RscaleddR2;
            der2->dz2 = d2Phi0dz2 * pow_2(dzscaleddz) + dPhi0dz * d2zscaleddz2;
            der2->dRdz= d2Phi0dRdz * dRscaleddR * dzscaleddz;
            der2->dRdphi = der2->dzdphi = der2->dphi2 = 0;
        }
        return;
    }

    // total scaled potential, gradient and hessian in scaled coordinates:
    // if using log-scaling, the values of m!=0 coefs are multiplied by the value of the
    // m=0 term, which we do at the very end, so initialize the sum with the value of the
    // m=0 term scaled by itself, i.e. unity; otherwise sum all m terms without scaling
    T Phi = logScaling ? 1 : Phi0;
    GradT grad;
    HessT hess;
    grad.dR  = grad.dz  = grad.dphi  = 0;
    hess.dR2 = hess.dz2 = hess.dphi2 = hess.dRdz = hess.dRdphi = hess.dzdphi = 0;

    // loop over other (m!=0) azimuthal harmonics and compute the temporary (scaled) values
    for(int mm=0; mm<=2*mmax; mm++) {
        int m = mm-mmax;
        if(!((present[mm>>5] >> (mm&31)) & 1u) || m==0)  // empty harmonic or the m=0 one
            continue;
        const T* h = harm + mm*CYLSPL_NHARM;
        T Phi_m = h[CYLSPL_PHI];
        T trigv = m>0 ? trig[m-1] : trig[mmax-1-m];  // cos or sin
        T dtrig = m>0 ? -m*trig[mmax+m-1] : -m*trig[-m-1];
        T d2trig = -m*m*trigv;
        Phi += Phi_m * trigv;
        if(needGrad) {
            grad.dR   += h[CYLSPL_DR] *  trigv;
            grad.dz   += h[CYLSPL_DZ] *  trigv;
            grad.dphi +=  Phi_m       * dtrig;
        }
        if(needHess) {
            hess.dR2    += h[CYLSPL_DR2]  *   trigv;
            hess.dz2    += h[CYLSPL_DZ2]  *   trigv;
            hess.dRdz   += h[CYLSPL_DRDZ] *   trigv;
            hess.dRdphi += h[CYLSPL_DR]   *  dtrig;
            hess.dzdphi += h[CYLSPL_DZ]   *  dtrig;
            hess.dphi2  +=  Phi_m         * d2trig;
        }
    }

    if(logScaling) {
        // unscale both amplitude of all quantities and their coordinate derivatives
        if(val)
            *val = Phi0 * Phi;
        if(der) {
            der->dR   = (Phi0 * grad.dR + dPhi0dR * Phi) * dRscaleddR;
            der->dz   = (Phi0 * grad.dz + dPhi0dz * Phi) * dzscaleddz;
            der->dphi =  Phi0 * grad.dphi;
        }
        if(der2) {
            der2->dR2 = (Phi0 * hess.dR2 + 2 * dPhi0dR * grad.dR + d2Phi0dR2 * Phi) *
                pow_2(dRscaleddR)  +  (Phi0 * grad.dR + dPhi0dR * Phi) * d2RscaleddR2;
            der2->dz2 = (Phi0 * hess.dz2 + 2 * dPhi0dz * grad.dz + d2Phi0dz2 * Phi) *
                pow_2(dzscaleddz)  +  (Phi0 * grad.dz + dPhi0dz * Phi) * d2zscaleddz2;
            der2->dRdz = (Phi0 * hess.dRdz + dPhi0dR * grad.dz + dPhi0dz * grad.dR + d2Phi0dRdz * Phi) *
                dRscaleddR * dzscaleddz;
            der2->dRdphi = (Phi0 * hess.dRdphi + dPhi0dR * grad.dphi) * dRscaleddR;
            der2->dzdphi = (Phi0 * hess.dzdphi + dPhi0dz * grad.dphi) * dzscaleddz;
            der2->dphi2  =  Phi0 * hess.dphi2;
        }
    } else {
        // unscale just the derivatives according to the coordinate transformation
        if(val)
            *val = Phi;
        if(der) {
            der->dR   = (grad.dR + dPhi0dR) * dRscaleddR;
            der->dz   = (grad.dz + dPhi0dz) * dzscaleddz;
            der->dphi = grad.dphi;
        }
        if(der2) {
            der2->dR2 = (hess.dR2 + d2Phi0dR2) * pow_2(dRscaleddR) + (grad.dR + dPhi0dR) * d2RscaleddR2;
            der2->dz2 = (hess.dz2 + d2Phi0dz2) * pow_2(dzscaleddz) + (grad.dz + dPhi0dz) * d2zscaleddz2;
            der2->dRdz = (hess.dRdz + d2Phi0dRdz) * dRscaleddR * dzscaleddz;
            der2->dRdphi = hess.dRdphi * dRscaleddR;
            der2->dzdphi = hess.dzdphi * dzscaleddz;
            der2->dphi2  = hess.dphi2;
        }
    }
}

/// \}

/** Density profile expressed as a Fourier expansion in azimuthal angle (phi)
    with coefficients interpolated on a 2d grid in meridional plane (R,z).
*/
class DensityAzimuthalHarmonic: public BaseDensity {
public:
    /** Construct the density interpolator from the input density profile.
        This is a static member function returning a pointer to a newly created object.
        The arguments have the same meaning as for `CylSpline::create`, but the grid extent
        (min/max) must be provided explicitly, i.e. is not determined automatically.
        The input density values are taken at the nodes of 2d grid in (R,z) and
        nphi distinct values of phi, where nphi=mmaxFourier+1 if the density is
        reflection-symmetric in y, or nphi=2*mmaxFourier+1 otherwise.
        The order of this internal Fourier expansion mmaxFourier will be fixed to mmax
        if fixOrder==true, otherwise will be higher than mmax to improve accuracy.
    */
    static shared_ptr<const DensityAzimuthalHarmonic> create(const BaseDensity& src,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax,
        bool fixOrder=false);

    /** construct the object from the array of coefficients */
    DensityAzimuthalHarmonic(
        const std::vector<double> &gridR,
        const std::vector<double> &gridz,
        const std::vector< math::Matrix<double> > &coefs);
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "DensityAzimuthalHarmonic"; }

    /** return the grid parameters */
    void getGridExtent(double &Rmin, double &Rmax, double &zmin, double &zmax) const;

    /** retrieve the values of density expansion coefficients 
        and the nodes of 2d grid used for interpolation */
    void getCoefs(std::vector<double> &gridR, std::vector<double> &gridz, 
        std::vector< math::Matrix<double> > &coefs) const;

    /** return the value of m-th Fourier harmonic at the given point in R,z plane */
    double rho_m(int m, double R, double z) const;

private:
    std::vector<math::PtrInterpolator2d> spl;  ///< spline for rho_m(R,z)
    coord::SymmetryType sym;  ///< type of symmetry deduced from coefficients
    double Rscale;            ///< radial scaling factor

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    /** Return density interpolated on the 2d grid, or zero if the point lies outside the grid */
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
};

/** Generic potential approximation based on Fourier expansion in azimuthal angle (phi)
    with coefficients being 2d spline functions of R,z.
    It is suitable for strongly flattened, possibly non-axisymmetric models with
    non-singular density profiles at origin. The potential and its derivatives are accurately
    interpolated within the grid definition region (0<=R<=Rmax, -zmax<=z<=zmax),
    and extrapolated outside this region using asymptotic spherical-harmonic expansion,
    which approximates the potential and forces rather well at all radii, but corresponds
    to zero density outside the grid.
    The cost of evaluating the potential is roughly the same as for the spherical-harmonic
    potential approximation, although the cost of computing the coefficients (via non-member
    functions `computePotentialCoefsCyl` or static factory functions `create`) is higher.
*/
class CylSpline: public BasePotentialCyl
{
public:

    /** Create the potential from the provided density model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential: it creates the grids, computes the coefficients
        and calls the actual class constructor.
        It exists in several variants: the first one takes a density model as input
        and solves Poisson equation to find the potential azimuthal harmonic coefficients;
        the second one takes a potential model and computes these coefs directly.
        In both cases, if the input model is not axisymmetric, its angular Fourier expansion
        with order mmaxFourier will be used to compute the harmonics, where mmaxFourier is
        fixed to mmax if fixOrder==true, otherwise will be higher to improve accuracy.
        A third variant takes an N-body snapshot as input (discussed separately).
        If the grid extent (R/z min/max) is not specified (left as zeros), it is determined
        automatically from the requirement to enclose almost all of the model mass and have
        a sufficient resolution at origin.
        \param[in]  src        is the input density or potential model;
        \param[in]  sym        is the required symmetry of the potential
        (if set to ST_UNKNOWN, will be taken from the input density model);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the number of grid nodes in cylindrical radius (semi-logarithmic);
        \param[in]  Rmin, Rmax give the radial grid extent (first non-zero node and
                    the outermost node); zero values mean auto-detect;
        \param[in]  gridSizez  is the number of grid nodes in vertical direction;
        \param[in]  zmin, zmax give the vertical grid extent (first non-zero positive node
                    and the outermost node; if the source model is not symmetric w.r.t.
                    z-reflection, a mirrored extension of the grid to negative z will be created);
                    zero values mean auto-detect;
        \param[in]  fixOrder   determines whether to limit the order of the internal Fourier
                    expansion of the input density to mmax; if false (default), use a larger
                    number of points for the integration in phi to improve the accuracy,
                    and then truncate the result back to mmax;
        \param[in]  useDerivs  specifies whether to compute potential derivatives from density.
        \note OpenMP-parallelized loop over nodes of a 2d grid in R,z when integrating the density
        (the latter is the input density when it is axisymmetric, otherwise an internally created
        instance of DensityAzimuthalHarmonic).
    */
    static shared_ptr<const CylSpline> create(const BaseDensity& src,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax,
        bool fixOrder=false, bool useDerivs=true);

    /** Same as above, but taking a potential model as an input. */
    static shared_ptr<const CylSpline> create(const BasePotential& src,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax,
        bool fixOrder=false);

    /** Create the potential from an N-body snapshot.
        \param[in] particles  is the array of particles.
        \param[in] sym  is the assumed symmetry of the input snapshot,
        which defines the list of angular harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd m terms are zeros).
        \param[in] mmax  is the order of angular expansion (if the symmetry includes
        coord::ST_ZROTATION flag, mmax will be set to zero).
        \param[in]  gridSizeR  is the number of grid nodes in cylindrical radius (semi-logarithmic);
        \param[in]  Rmin, Rmax give the radial grid extent (first non-zero node and
        the outermost node); zero values mean that they will be determined automatically from
        the requirement to enclose almost all particles and provide a good resolution.
        \param[in]  gridSizez  is the number of grid nodes in vertical direction;
        \param[in]  zmin, zmax give the vertical grid extent (first non-zero positive node
        and the outermost node); if the requested symmetry type does not include
        z-reflection, a mirrored extension of the grid to negative z will be created;
        zero values mean that they will be assigned automatically.
        \param[in]  useDerivs  specifies whether to compute potential derivatives
        and construct a quintic spline, or skip it and construct a cubic spline
        (due to noisy nature of N-body models, higher order does not necessarily imply more accuracy).
        \note OpenMP-parallelized loop over nodes of a 2d grid in R,z.
    */
    static shared_ptr<const CylSpline> create(
        const particles::ParticleArray<coord::PosCyl>& particles,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax, bool useDerivs=false);

    /** Construct the potential from previously computed coefficients.
        \param[in]  gridR  is the grid in cylindrical radius
        (nodes must start at 0 and be increasing with R);
        \param[in]  gridz  is the grid in the vertical direction:
        if it starts at 0 and covers the positive half-space, then the potential
        is assumed to be symmetric w.r.t. z-reflection, so that the internal grid
        will be extended to the negative half-space; in the opposite case it
        is assumed to be asymmetric and the grid must cover negative z too.
        \param[in]  Phi  is the 3d array of harmonic coefficients:
        the outermost dimension determines the order of expansion - the number of terms
        is 2*mmax+1, so that m runs from -mmax to mmax inclusive (i.e. the m=0 harmonic
        is contained in the element with index mmax).
        Each element of this array (apart from the one at mmax, i.e. for m=0) may be empty,
        in which case the corresponding harmonic is taken to be identically zero.
        Non-empty elements are matrices with dimension gridR.size() * gridz.size(),
        regardless of whether gridz covers only z>=0 or both positive and negative z.
        The indexing scheme is Phi[m+mmax](iR,iz) = Phi_m(gridR[iR], gridz[iz]).
        \param[in]  dPhidR  is the array of radial derivatives of the potential,
        with the same shape as Phi, and containing the same number of non-empty terms.
        \param[in]  dPhidz  is the array of vertical derivatives.
        If both dPhidR and dPhidz are empty arrays (not arrays with empty elements),
        as specified by default, then the potential is constructed using only the values
        of Phi at grid nodes, employing 2d cubic spline interpolation for each m term.
        If derivatives are provided, then the interpolation is based on quintic splines,
        improving the accuracy.
    */
    CylSpline(
        const std::vector<double> &gridR,
        const std::vector<double> &gridz,
        const std::vector< math::Matrix<double> > &Phi,
        const std::vector< math::Matrix<double> > &dPhidR = std::vector< math::Matrix<double> >(),
        const std::vector< math::Matrix<double> > &dPhidz = std::vector< math::Matrix<double> >() );

    virtual std::string name() const { return myName(); }
    static std::string myName() { return "CylSpline"; }
    virtual coord::SymmetryType symmetry() const { return sym; };
    virtual double enclosedMass(const double radius) const;

    /** return the grid parameters */
    void getGridExtent(double &Rmin, double &Rmax, double &zmin, double &zmax) const;

    /** retrieve coefficients of potential approximation.
        \param[out] gridR will be filled with the array of R-values of grid nodes;
        \param[out] gridz will be filled with the array of z-values of grid nodes:
        if the potential is symmetric w.r.t. z-reflection, then only the half-space with
        non-negative z is returned both in this array and in the coefficients.
        \param[out] Phi will contain array of sequentially stored 2d arrays
        (the size of the outer array equals the number of terms in azimuthal expansion (2*mmax+1),
        inner arrays contain gridR.size()*gridz.size() values).
        \param[out] dPhidR will contain the array of derivatives in the radial direction,
        with the same shape as Phi, if they were provided at construction (i.e., when using
        quintic interpolation internally), otherwise this will be an empty array.
        \param[out] dPhidz will contain the array of derivatives in the vertical direction,
        or an empty array if the derivatives were not provided at construction
    */
    void getCoefs(
        std::vector<double> &gridR,
        std::vector<double> &gridz,
        std::vector< math::Matrix<double> > &Phi,
        std::vector< math::Matrix<double> > &dPhidR,
        std::vector< math::Matrix<double> > &dPhidz) const;

private:
    /// array of 2d splines (for each m-component in the expansion in azimuthal angle)
    std::vector<math::PtrInterpolator2d> spl;
    coord::SymmetryType sym;  ///< type of symmetry deduced from coefficients
    double Rscale;            ///< radial scaling factor for coordinate transformation
    bool logScaling;          ///< flag for optional log-transformation of the m=0 term
    bool haveDerivs;          ///< whether the potential derivatives were provided at construction

    /// asymptotic behaviour at large radii described by `PowerLawMultipole`
    PtrPotential asymptOuter;

    /// compute potential and its derivatives
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const;
};

}  // namespace
