/** \file    potential_analytic.h
    \brief   Several common analytic potential models
    \author  Eugene Vasiliev
    \date    2009-2025
*/
#pragma once
#include "potential_base.h"
#include "gpu_policy.h"   // agama::forall for templated batch evaluators (Tier 1)
#include <cmath>          // std::sqrt, std::log

namespace potential{

// =====================================================================
// Tier 1 leaf math: Phi-only, AGAMA_DEVICE_INLINE, single-source per class.
// Each leaf is the same body that the corresponding evalDeriv/evalCar/evalCyl
// uses for the *potential* output, extracted as a free function so it can be
// called from both CPU and CUDA TUs. POD-only inputs by value -> safe inside
// a captured-by-copy device lambda.
// =====================================================================

/** Plummer:  Phi(r) = -M / sqrt(r^2 + b^2).  Returns 0 when mass=0 (matches CPU). */
template<typename T>
AGAMA_DEVICE_INLINE T plummer_phi(T mass, T scaleRadius, T r) {
    if(mass == T(0)) return T(0);
    return -mass / std::sqrt(r*r + scaleRadius*scaleRadius);
}

/** Isochrone:  Phi(r) = -M / (b + sqrt(r^2 + b^2)). */
template<typename T>
AGAMA_DEVICE_INLINE T isochrone_phi(T mass, T scaleRadius, T r) {
    return -mass / (scaleRadius + std::sqrt(r*r + scaleRadius*scaleRadius));
}

/** NFW:  Phi(r) = -M ln(1+r/r_s) / r, with a Pade(2,3) expansion at r->0. */
template<typename T>
AGAMA_DEVICE_INLINE T nfw_phi(T mass, T scaleRadius, T r) {
    T rrel = r / scaleRadius;
    T ln_over_r = r == T(INFINITY) ? T(0) :
        rrel > T(0.016) ? std::log(T(1) + rrel) / r :
        // accurate (14 digits) asymptotic Pade(2,3) expansion at r->0
        (T(1) + rrel * (T(1) + T(11.0/60.0) * rrel)) /
        (T(1) + rrel * (T(1.5) + rrel * (T(0.6) + rrel * T(0.05)))) / scaleRadius;
    return -mass * ln_over_r;
}

/** MiyamotoNagai:  Phi(R,z) = -M / sqrt(R^2 + (a + sqrt(z^2+b^2))^2). */
template<typename T>
AGAMA_DEVICE_INLINE T miyamoto_nagai_phi(T mass, T scaleRadius, T scaleHeight, T R, T z) {
    T zb   = std::sqrt(z*z + scaleHeight*scaleHeight);
    T azb  = scaleRadius + zb;
    return -mass / std::sqrt(R*R + azb*azb);
}

/** Logarithmic:  Phi(x,y,z) = 0.5 v0^2 ln( (r_c^2 + x^2 + (y/p)^2 + (z/q)^2) / L^2 ).
    Parameters are passed already-squared to match the class member layout. */
template<typename T>
AGAMA_DEVICE_INLINE T logarithmic_phi(T v0squared, T coreRadius2, T p2, T q2, T lengthUnit2,
                                      T x, T y, T z)
{
    T m2 = coreRadius2 + x*x + (y*y)/p2 + (z*z)/q2;
    return T(0.5) * v0squared * std::log(m2 / lengthUnit2);
}

/** Harmonic:  Phi(x,y,z) = 0.5 Omega^2 ( x^2 + (y/p)^2 + (z/q)^2 ).
    Parameters are passed already-squared to match the class member layout. */
template<typename T>
AGAMA_DEVICE_INLINE T harmonic_phi(T Omega2, T p2, T q2, T x, T y, T z) {
    return T(0.5) * Omega2 * (x*x + (y*y)/p2 + (z*z)/q2);
}

/** Spherical Plummer potential:
    \f$  \Phi(r) = - M / \sqrt{r^2 + b^2}  \f$. */
class Plummer: public BasePotentialSphericallySymmetric{
public:
    Plummer(double _mass, double _scaleRadius) :
        mass(_mass), scaleRadius(_scaleRadius) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Plummer"; }
    virtual double enclosedMass(const double radius) const;
    virtual double totalMass() const { return mass; }

    /** Tier 1 batch evaluator: Phi at N Cartesian positions via plummer_phi leaf.
        xyz: packed input length 3*N (x0,y0,z0, x1,y1,z1, ...); phi: output length N.
        Templated on precision T (float or double) and execution policy.
        add=true accumulates into phi[] instead of overwriting (composite support);
        the math is unchanged, only the final store differs (uniform branch). */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = plummer_phi(m, b, r);
            phi[i] = add ? phi[i] + v : v;
        });
    }
private:
    const double mass;         ///< total mass  (M)
    const double scaleRadius;  ///< scale radius of the Plummer model  (b)

    /** Evaluate potential and up to two its derivatives by spherical radius. */
    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;

    /** explicitly define the density function, instead of relying on the potential derivatives
        (they suffer from cancellation errors already at r>1e5) */
    virtual double densitySph(const coord::PosSph &pos, double time) const;
};

/** Spherical Isochrone potential:
    \f$  \Phi(r) = - M / (b + \sqrt{r^2 + b^2})  \f$. */
class Isochrone: public BasePotentialSphericallySymmetric{
public:
    Isochrone(double _mass, double _scaleRadius) :
        mass(_mass), scaleRadius(_scaleRadius) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Isochrone"; }
    virtual double totalMass() const { return mass; }
    double getRadius() const { return scaleRadius; }

    /** Tier 1 batch evaluator: Phi at N Cartesian positions via isochrone_phi leaf.
        add=true accumulates into phi[] instead of overwriting (composite support). */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = isochrone_phi(m, b, r);
            phi[i] = add ? phi[i] + v : v;
        });
    }
private:
    const double mass;         ///< total mass  (M)
    const double scaleRadius;  ///< scale radius of the Isochrone model  (b)
    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;
    virtual double densitySph(const coord::PosSph &pos, double time) const;
};

/** Spherical Navarro-Frenk-White potential:
    \f$  \Phi(r) = - M \ln{1 + (r/r_s)} / r  \f$  (note that total mass is infinite and not M). */
class NFW: public BasePotentialSphericallySymmetric{
public:
    NFW(double _mass, double _scaleRadius) :
        mass(_mass), scaleRadius(_scaleRadius) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "NFW"; }
    virtual double totalMass() const { return INFINITY; }

    /** Tier 1 batch evaluator: NFW potential at N Cartesian positions.
        Single source: the lambda body calls the inline `nfw_phi` leaf, which is
        the same body that `evalDeriv` uses for the potential output. Policy may be
        agama::Serial, agama::OpenMP, or (with HAVE_CUDA=1 + nvcc TU) agama::Cuda.
        For the Cuda policy, `xyz[]` and `phi[]` must be device pointers.

        xyz: packed input length 3*N (x0,y0,z0, x1,y1,z1, ...); phi: output length N.
        Templated on precision T (float or double) and execution policy.
        add=true accumulates into phi[] instead of overwriting (composite support). */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, bool add = false) const
    {
        const T m = static_cast<T>(mass), rs = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = nfw_phi(m, rs, r);
            phi[i] = add ? phi[i] + v : v;
        });
    }
private:
    const double mass;         ///< normalization factor  (M);  equals to mass enclosed within ~5.3r_s
    const double scaleRadius;  ///< scale radius of the NFW model  (r_s)

    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const
    { return (1./4/M_PI) * mass / pos.r / pow_2(pos.r + scaleRadius); }
};

/** Axisymmetric Miyamoto-Nagai potential:
    \f$  \Phi(r) = - M / \sqrt{ R^2 + (a + \sqrt{z^2+b^2})^2 }  \f$.
    When a=0, this potential is equivalent to a spherical Plummer model with scale radius b.
*/
class MiyamotoNagai: public BasePotentialCyl{
public:
    MiyamotoNagai(double _mass, double _scaleRadius, double _scaleHeight) :
        mass(_mass), scaleRadius(_scaleRadius), scaleHeight(_scaleHeight) {}
    virtual coord::SymmetryType symmetry() const {
        return scaleRadius==0 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "MiyamotoNagai"; }
    virtual double totalMass() const { return mass; }

    /** Tier 1 batch evaluator: Phi at N Cartesian positions via miyamoto_nagai_phi leaf.
        Computes R = sqrt(x^2+y^2) directly (axisymmetric Phi has no phi dependence).
        add=true accumulates into phi[] instead of overwriting (composite support). */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, bool add = false) const
    {
        const T m = static_cast<T>(mass), a = static_cast<T>(scaleRadius), b = static_cast<T>(scaleHeight);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T R = std::sqrt(x*x + y*y);
            const T v = miyamoto_nagai_phi(m, a, b, R, z);
            phi[i] = add ? phi[i] + v : v;
        });
    }
private:
    const double mass;        ///< total mass  (M)
    const double scaleRadius; ///< scale radius (a),  determines the extent in the disk plane
    const double scaleHeight; ///< scale height (b),  determines the vertical extent

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
};

/** Triaxial Long-Murali bar potential, obtained by convolving the Miyamoto-Nagai potential
    \f$ \Phi_{MN} \f$  with a needle extending from x=-l to x=+l:
    \f$  \Phi(x,y,z) = \frac{1}{2l} \int_{-l}^{l} dl\, \Phi_{MN}(x-l, y, z)  \f$.
    It looks like a Miyamoto-Nagai disk with an embedded bar along the x axis; 
    when l=0 and a!=0, this potential is equivalent to an axisymmetric Miyamoto-Nagai disk;
    when a=0 and l!=0, the bar is symmetric w.r.t. rotation about the x axis, and there is no disk.
*/
class LongMurali: public BasePotentialCar{
public:
    LongMurali(double _mass, double _scaleRadius, double _scaleHeight, double _barLength) :
        mass(_mass), scaleRadius(_scaleRadius), scaleHeight(_scaleHeight), barLength(_barLength) {}
    virtual coord::SymmetryType symmetry() const {
        return barLength==0 ?
            (scaleRadius==0 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) :
            coord::ST_TRIAXIAL; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "LongMurali"; }
    virtual double totalMass() const { return mass; }
private:
    const double mass, scaleRadius, scaleHeight, barLength;
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
};

/** Triaxial logarithmic potential:
    \f$  \Phi(r) = (1/2) v_0^2 \ln[ (r_c^2 + x^2 + (y/p)^2 + (z/q)^2) / L^2 ]  \f$,
    where  v_0  is the asymptotic circular velocity,
    r_c  is the core radius,  p and q  are the axis ratios,
    and L  is the length unit that makes the expression under the logarithm dimensionless.
*/
class Logarithmic: public BasePotentialCar{
public:
    Logarithmic(double v0, double coreRadius=0, double axisRatioYtoX=1, double axisRatioZtoX=1,
        double lengthUnit=1) :
        v0squared(pow_2(v0)), coreRadius2(pow_2(coreRadius)),
        p2(pow_2(axisRatioYtoX)), q2(pow_2(axisRatioZtoX)), lengthUnit2(pow_2(lengthUnit))
    {}
    virtual coord::SymmetryType symmetry() const {
        return p2==1 ? (q2==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Logarithmic"; }
    virtual double totalMass() const { return INFINITY; }

    /** Tier 1 batch evaluator: Phi at N Cartesian positions via logarithmic_phi leaf.
        add=true accumulates into phi[] instead of overwriting (composite support). */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, bool add = false) const
    {
        const T v2 = static_cast<T>(v0squared), c2 = static_cast<T>(coreRadius2);
        const T pp = static_cast<T>(p2), qq = static_cast<T>(q2), L2 = static_cast<T>(lengthUnit2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T v = logarithmic_phi(v2, c2, pp, qq, L2, x, y, z);
            phi[i] = add ? phi[i] + v : v;
        });
    }
private:
    const double v0squared;    ///< squared asymptotic circular velocity (v_0)
    const double coreRadius2;  ///< squared core radius (r_c)
    const double p2;           ///< squared y/x axis ratio (p)
    const double q2;           ///< squared z/x axis ratio (q)
    const double lengthUnit2;  ///< squared length unit (L)

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
};

/** Triaxial harmonic potential:
    \f$  \Phi(r) = (1/2) \Omega^2 [ x^2 + (y/p)^2 + (z/q)^2 ]  \f$. */
class Harmonic: public BasePotentialCar{
public:
    Harmonic(double Omega, double axisRatioYtoX=1, double axisRatioZtoX=1) :
        Omega2(pow_2(Omega)), p2(pow_2(axisRatioYtoX)), q2(pow_2(axisRatioZtoX)) {}
    virtual coord::SymmetryType symmetry() const {
        return p2==1 ? (q2==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Harmonic"; }
    virtual double totalMass() const { return INFINITY; }

    /** Tier 1 batch evaluator: Phi at N Cartesian positions via harmonic_phi leaf.
        add=true accumulates into phi[] instead of overwriting (composite support). */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, bool add = false) const
    {
        const T w2 = static_cast<T>(Omega2), pp = static_cast<T>(p2), qq = static_cast<T>(q2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T v = harmonic_phi(w2, pp, qq, x, y, z);
            phi[i] = add ? phi[i] + v : v;
        });
    }
private:
    const double Omega2;       ///< squared oscillation frequency (Omega)
    const double p2;           ///< squared y/x axis ratio (p)
    const double q2;           ///< squared z/x axis ratio (q)

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
};


//------ Potential of a binary black hole ------//

/** Parameters describing the central single or binary supermassive black hole (BH) */
struct KeplerBinaryParams {
    double mass;  ///< mass of the central black hole or total mass of the binary
    double q;     ///< binary BH mass ratio (0<=q<=1)
    double sma;   ///< binary BH semimajor axis
    double ecc;   ///< binary BH eccentricity (0<=ecc<1)
    double phase; ///< binary BH orbital phase (0<=phase<2*pi)

    /// set defaults
    KeplerBinaryParams(double _mass=0, double _q=0, double _sma=0, double _ecc=0, double _phase=0) :
        mass(_mass), q(_q), sma(_sma), ecc(_ecc), phase(_phase) {}

    /** Compute the position and velocity of the two components of the binary black hole
        at the time 't'.
        if this is a single black hole, it stays at origin with zero velocity,
        and in case of a binary, its center of mass is fixed at origin.
        Output arrays of length 2 each will contain the x- and y-coordinates and
        corresponding velocities of both components of the binary at time 't';
        its orbit is assumed to lie in the x-y plane and directed along the x axis.
    */
    void keplerOrbit(double t, double bhX[], double bhY[], double bhVX[], double bhVY[]) const;

    /** Compute the time-dependent potential of the two black holes at the given point */
    double potential(const coord::PosCar& point, double time) const;
};

/** Time-dependent potential of a binary BH on a Kepler orbit in the x-y plane, centered at origin */
class KeplerBinary: public BasePotentialCar {
public:
    KeplerBinary(const KeplerBinaryParams& _params) : params(_params) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "KeplerBinary"; }
    virtual coord::SymmetryType symmetry() const
    { return (params.sma==0 || params.q==0 ? coord::ST_SPHERICAL : coord::ST_NONE); }
    virtual double totalMass() const { return params.mass; }
    virtual double enclosedMass(const double radius) const
    { return radius>=params.sma ? params.mass : 0; }
private:
    const KeplerBinaryParams params;   ///< parameters of the binary
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &, double) const { return 0; }
};

}