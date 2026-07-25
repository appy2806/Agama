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
// Tier 1 leaf math: AGAMA_DEVICE_INLINE, single-source per class.
// Each <name>_eval leaf is the SAME math as the corresponding CPU
// evalDeriv/evalCyl/evalCar in potential_analytic.cpp -- the expressions
// (including Pade/Taylor guard branches) are copied verbatim, and the CPU
// virtual methods are now thin calls into these leaves, so the values are
// bit-for-bit identical between the virtual single-point path and the
// templated batch path. Outputs are nullable pointers, AGAMA's own
// evalDeriv convention: pass NULL to skip an output (uniform branch, safe
// on device). POD-only inputs by value -> safe inside a captured-by-copy
// device lambda.
//
// Each <name>_rho leaf mirrors the class's EXISTING density path: the
// explicit closed-form density override where one exists (Plummer,
// Isochrone, NFW, MiyamotoNagai), or the Laplacian/Poisson route
// (sum of Cartesian second derivatives) / (4 pi G) used by the generic
// BasePotential::densityCar for Logarithmic and Harmonic.
// =====================================================================

/** Plummer:  Phi(r) = -M / sqrt(r^2 + b^2), plus dPhi/dr and d2Phi/dr2.
    Outputs 0 when mass=0 (matches CPU). Null output pointers are skipped. */
template<typename T>
AGAMA_DEVICE_INLINE void plummer_eval(T mass, T scaleRadius, T r,
    T* potential, T* deriv, T* deriv2)
{
    T invrsq = mass != T(0) ?  T(1) / (pow_2(r) + pow_2(scaleRadius)) : T(0);  // if mass=0, output 0
    T pot = -mass * std::sqrt(invrsq);
    if(potential)
        *potential = pot;
    if(deriv)
        *deriv = -pot * r * invrsq;
    if(deriv2)
        *deriv2 = pot * (T(2) * pow_2(r * invrsq) - pow_2(scaleRadius * invrsq));
}

/** Plummer density (explicit closed form, same as Plummer::densitySph). */
template<typename T>
AGAMA_DEVICE_INLINE T plummer_rho(T mass, T scaleRadius, T r) {
    T invrsq = T(1) / (pow_2(r) + pow_2(scaleRadius));
    return T(3./4/M_PI) * mass * pow_2(scaleRadius * invrsq) * std::sqrt(invrsq);
}

/** Plummer:  Phi only (thin wrapper over plummer_eval). */
template<typename T>
AGAMA_DEVICE_INLINE T plummer_phi(T mass, T scaleRadius, T r) {
    T phi;
    plummer_eval(mass, scaleRadius, r, &phi, (T*)NULL, (T*)NULL);
    return phi;
}

/** Isochrone:  Phi(r) = -M / (b + sqrt(r^2 + b^2)), plus dPhi/dr and d2Phi/dr2. */
template<typename T>
AGAMA_DEVICE_INLINE void isochrone_eval(T mass, T scaleRadius, T r,
    T* potential, T* deriv, T* deriv2)
{
    T rb  = std::sqrt(pow_2(r) + pow_2(scaleRadius));
    T brb = scaleRadius + rb;
    T pot = -mass / brb;
    if(potential)
        *potential = pot;
    if(deriv)
        *deriv = -pot * r / (rb * brb);
    if(deriv2)
        *deriv2 = pot * (T(2)*pow_2(r / (rb * brb)) -
            pow_2(scaleRadius / (rb * brb)) * (T(1) + scaleRadius / rb));
}

/** Isochrone density (explicit closed form, same as Isochrone::densitySph). */
template<typename T>
AGAMA_DEVICE_INLINE T isochrone_rho(T mass, T scaleRadius, T r) {
    T rb  = std::sqrt(pow_2(r) + pow_2(scaleRadius));
    T brb = scaleRadius + rb;
    return T(1./4/M_PI) * mass * scaleRadius *
        (T(3) * scaleRadius * brb + T(2) * pow_2(r)) / pow_3(rb * brb);
}

/** Isochrone:  Phi only (thin wrapper over isochrone_eval). */
template<typename T>
AGAMA_DEVICE_INLINE T isochrone_phi(T mass, T scaleRadius, T r) {
    T phi;
    isochrone_eval(mass, scaleRadius, r, &phi, (T*)NULL, (T*)NULL);
    return phi;
}

/** Crossover r/r_s between the closed-form NFW expressions and their Pade expansions at
    r->0. This has to depend on the precision: above the crossover, dPhi/dr and d2Phi/dr2
    subtract two nearly-equal O(1/r_s) terms, so the closed form loses ~eps/rrel^2 relative
    accuracy to cancellation, while the Pade truncation error grows with rrel. The optimum
    therefore scales as sqrt(eps) and sits ~20x higher in single precision. Measured
    worst-case relative error over rrel in [1e-5, 100] against an 80-bit reference:

                       fp64 thresholds     fp32 threshold
                       used in fp32        0.25
        Phi              3.7e-6              3.1e-7
        dPhi/dr          4.7e-4              3.1e-6
        d2Phi/dr2        4.5e-2              3.3e-5

    The fp64 values are upstream's tuned constants and MUST NOT change -- fp64 is
    bit-for-bit legacy behaviour (hard constraint #3). In fp32 one crossover is within 40%
    of the per-quantity optimum for all three quantities, so unlike fp64 there is nothing
    to gain from three separate constants. */
template<typename T> struct nfw_pade_guard;   // no default: an unsupported T won't compile
template<> struct nfw_pade_guard<double> {
    static AGAMA_DEVICE_INLINE double potential() { return 0.016; }
    static AGAMA_DEVICE_INLINE double deriv()     { return 0.013; }
    static AGAMA_DEVICE_INLINE double deriv2()    { return 0.010; }
};
template<> struct nfw_pade_guard<float> {
    static AGAMA_DEVICE_INLINE float potential() { return 0.25f; }
    static AGAMA_DEVICE_INLINE float deriv()     { return 0.25f; }
    static AGAMA_DEVICE_INLINE float deriv2()    { return 0.25f; }
};

/** NFW:  Phi(r) = -M ln(1+r/r_s) / r, plus dPhi/dr and d2Phi/dr2, each with
    its own accurate Pade expansion at r->0 (see nfw_pade_guard for the thresholds). */
template<typename T>
AGAMA_DEVICE_INLINE void nfw_eval(T mass, T scaleRadius, T r,
    T* potential, T* deriv, T* deriv2)
{
    T rrel = r / scaleRadius;
    T ln_over_r = r == T(INFINITY) ? T(0) :
        rrel > nfw_pade_guard<T>::potential() ? std::log(T(1) + rrel) / r :
        // accurate (14 digits) asymptotic Pade(2,3) expansion at r->0
        (T(1) + rrel * (T(1) + T(11./60) * rrel)) /
        (T(1) + rrel * (T(1.5) + rrel * (T(0.6) + rrel * T(0.05)))) / scaleRadius;
    if(potential)
        *potential = -mass * ln_over_r;
    if(deriv)
        *deriv = mass * (rrel > nfw_pade_guard<T>::deriv() ?
            (ln_over_r - T(1)/(r+scaleRadius)) / r :
            // accurate (12 digits) asymptotic Pade(1,3) expansion at r->0
            (T(0.5) + T(17./96) * rrel) /
            (T(1) + rrel * (T(27./16) + rrel * (T(0.75) + T(11./160) * rrel))) /
            pow_2(scaleRadius));
    if(deriv2)
        *deriv2 = -mass * (rrel > nfw_pade_guard<T>::deriv2() ?
            (T(2)*ln_over_r - (T(2)*scaleRadius + T(3)*r) / pow_2(scaleRadius+r) ) / pow_2(r) :
            // accurate (10 digits) asymptotic Pade(2,3) expansion at r->0
            T(1) / (T(1.5) + rrel * (T(27./8) + rrel * (T(351./160) + T(183./640) * rrel))) /
            pow_3(scaleRadius) );
}

/** NFW density (explicit closed form, same as NFW::densitySph). */
template<typename T>
AGAMA_DEVICE_INLINE T nfw_rho(T mass, T scaleRadius, T r) {
    return T(1./4/M_PI) * mass / r / pow_2(r + scaleRadius);
}

/** NFW:  Phi only (thin wrapper over nfw_eval). */
template<typename T>
AGAMA_DEVICE_INLINE T nfw_phi(T mass, T scaleRadius, T r) {
    T phi;
    nfw_eval(mass, scaleRadius, r, &phi, (T*)NULL, (T*)NULL);
    return phi;
}

/** MiyamotoNagai:  Phi(R,z) = -M / sqrt(R^2 + (a + sqrt(z^2+b^2))^2),
    plus first (dR, dz) and second (dR2, dz2, dRdz) cylindrical derivatives.
    dphi-direction derivatives are identically zero (axisymmetric) and not output. */
template<typename T>
AGAMA_DEVICE_INLINE void miyamoto_nagai_eval(T mass, T scaleRadius, T scaleHeight, T R, T z,
    T* potential, T* dR, T* dz, T* dR2, T* dz2, T* dRdz)
{
    T zb    = std::sqrt(pow_2(z) + pow_2(scaleHeight));
    T azb2  = pow_2(scaleRadius + zb);
    T den2  = T(1) / (pow_2(R) + azb2);
    T denom = std::sqrt(den2);
    T Rsc   = R * denom;
    T zsc   = z * denom;
    if(potential)
        *potential = -mass * denom;
    if(dR)
        *dR = mass * den2 * Rsc;
    if(dz)
        *dz = mass * den2 * zsc * (T(1) + scaleRadius / zb);
    if(dR2 || dz2 || dRdz) {
        T mden3 = mass * denom * den2;
        if(dR2)
            *dR2  = mden3 * (azb2 * den2 - T(2)*pow_2(Rsc));
        if(dz2)
            *dz2  = mden3 * ( (pow_2(Rsc) - T(2) * azb2 * den2) * pow_2(z / zb) +
                pow_2(scaleHeight) * (T(1) + scaleRadius / zb) * (pow_2(Rsc) + azb2 * den2) / pow_2(zb) );
        if(dRdz)
            *dRdz = mden3 * T(-3) * Rsc * zsc * (T(1) + scaleRadius / zb);
    }
}

/** MiyamotoNagai density (explicit closed form, same as MiyamotoNagai::densityCyl). */
template<typename T>
AGAMA_DEVICE_INLINE T miyamoto_nagai_rho(T mass, T scaleRadius, T scaleHeight, T R, T z) {
    T zb   = std::sqrt(pow_2(z) + pow_2(scaleHeight));
    T azb2 = pow_2(scaleRadius + zb), R2azb2 = pow_2(R) + azb2;
    return T(1./4/M_PI) * mass * pow_2(scaleHeight) *
        (scaleRadius + T(3)*zb * azb2 / R2azb2) / (pow_3(zb) * R2azb2 * std::sqrt(R2azb2));
}

/** MiyamotoNagai:  Phi only (thin wrapper over miyamoto_nagai_eval). */
template<typename T>
AGAMA_DEVICE_INLINE T miyamoto_nagai_phi(T mass, T scaleRadius, T scaleHeight, T R, T z) {
    T phi;
    miyamoto_nagai_eval(mass, scaleRadius, scaleHeight, R, z,
        &phi, (T*)NULL, (T*)NULL, (T*)NULL, (T*)NULL, (T*)NULL);
    return phi;
}

/** Logarithmic:  Phi(x,y,z) = 0.5 v0^2 ln( (r_c^2 + x^2 + (y/p)^2 + (z/q)^2) / L^2 ),
    plus the Cartesian gradient (grad[3]: dx,dy,dz) and Hessian
    (hess[6]: dx2,dy2,dz2,dxdy,dydz,dxdz). Parameters are passed already-squared
    to match the class member layout. */
template<typename T>
AGAMA_DEVICE_INLINE void logarithmic_eval(T v0squared, T coreRadius2, T p2, T q2, T lengthUnit2,
    T x, T y, T z, T* potential, T* grad, T* hess)
{
    T m2 = coreRadius2 + pow_2(x) + pow_2(y)/p2 + pow_2(z)/q2;
    if(potential)
        *potential = T(0.5) * v0squared * std::log(m2 / lengthUnit2);
    if(grad) {
        grad[0] = x * v0squared/m2;
        grad[1] = y * v0squared/m2/p2;
        grad[2] = z * v0squared/m2/q2;
    }
    if(hess) {
        hess[0] = v0squared * (T(1)/m2    - T(2) * pow_2(x / m2));
        hess[1] = v0squared * (T(1)/m2/p2 - T(2) * pow_2(y / (m2 * p2)));
        hess[2] = v0squared * (T(1)/m2/q2 - T(2) * pow_2(z / (m2 * q2)));
        hess[3] =-v0squared * x * y * T(2) / (pow_2(m2) * p2);
        hess[4] =-v0squared * y * z * T(2) / (pow_2(m2) * p2 * q2);
        hess[5] =-v0squared * z * x * T(2) / (pow_2(m2) * q2);
    }
}

/** Logarithmic density via the Poisson/Laplacian route -- the same
    (dx2+dy2+dz2)/(4 pi) expression as the generic BasePotential::densityCar,
    which is what the CPU path uses (the class has no explicit density override). */
template<typename T>
AGAMA_DEVICE_INLINE T logarithmic_rho(T v0squared, T coreRadius2, T p2, T q2, T lengthUnit2,
    T x, T y, T z)
{
    T hess[6];
    logarithmic_eval(v0squared, coreRadius2, p2, q2, lengthUnit2, x, y, z,
        (T*)NULL, (T*)NULL, hess);
    return (hess[0] + hess[1] + hess[2]) * T(1. / (4*M_PI));
}

/** Logarithmic:  Phi only (thin wrapper over logarithmic_eval). */
template<typename T>
AGAMA_DEVICE_INLINE T logarithmic_phi(T v0squared, T coreRadius2, T p2, T q2, T lengthUnit2,
                                      T x, T y, T z)
{
    T phi;
    logarithmic_eval(v0squared, coreRadius2, p2, q2, lengthUnit2, x, y, z,
        &phi, (T*)NULL, (T*)NULL);
    return phi;
}

/** Harmonic:  Phi(x,y,z) = 0.5 Omega^2 ( x^2 + (y/p)^2 + (z/q)^2 ),
    plus the Cartesian gradient (grad[3]) and Hessian (hess[6], same layout as
    logarithmic_eval). Parameters are passed already-squared to match the class
    member layout. */
template<typename T>
AGAMA_DEVICE_INLINE void harmonic_eval(T Omega2, T p2, T q2, T x, T y, T z,
    T* potential, T* grad, T* hess)
{
    if(potential)
        *potential = T(0.5)*Omega2 * (pow_2(x) + pow_2(y)/p2 + pow_2(z)/q2);
    if(grad) {
        grad[0] = x*Omega2;
        grad[1] = y*Omega2/p2;
        grad[2] = z*Omega2/q2;
    }
    if(hess) {
        hess[0] = Omega2;
        hess[1] = Omega2/p2;
        hess[2] = Omega2/q2;
        hess[3] = hess[4] = hess[5] = T(0);
    }
}

/** Harmonic density via the Poisson/Laplacian route (same as the generic
    BasePotential::densityCar used by the CPU path; spatially constant). */
template<typename T>
AGAMA_DEVICE_INLINE T harmonic_rho(T Omega2, T p2, T q2, T x, T y, T z)
{
    T hess[6];
    harmonic_eval(Omega2, p2, q2, x, y, z, (T*)NULL, (T*)NULL, hess);
    return (hess[0] + hess[1] + hess[2]) * T(1. / (4*M_PI));
}

/** Harmonic:  Phi only (thin wrapper over harmonic_eval). */
template<typename T>
AGAMA_DEVICE_INLINE T harmonic_phi(T Omega2, T p2, T q2, T x, T y, T z) {
    T phi;
    harmonic_eval(Omega2, p2, q2, x, y, z, &phi, (T*)NULL, (T*)NULL);
    return phi;
}

// =====================================================================
// Cartesian acceleration helpers: convert leaf derivatives to a = -grad Phi.
// =====================================================================

/** Spherical:  a = -dPhi/dr * (x,y,z)/r,  with a = 0 at the origin (r=0 guard). */
template<typename T>
AGAMA_DEVICE_INLINE void sph_acc_car(T dPhi_dr, T x, T y, T z, T r, T* acc /*[3]*/)
{
    T s = r > T(0) ? -dPhi_dr / r : T(0);
    acc[0] = s * x;
    acc[1] = s * y;
    acc[2] = s * z;
}

/** Cylindrical (axisymmetric):  a_xy = -dPhi/dR * (x,y)/R  (R=0 guard),  a_z = -dPhi/dz. */
template<typename T>
AGAMA_DEVICE_INLINE void cyl_acc_car(T dPhi_dR, T dPhi_dz, T x, T y, T R, T* acc /*[3]*/)
{
    T s = R > T(0) ? -dPhi_dR / R : T(0);
    acc[0] = s * x;
    acc[1] = s * y;
    acc[2] = -dPhi_dz;
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
        the math is unchanged, only the final store differs (uniform branch).
        time parameter accepted for signature uniformity with time-dependent potentials but ignored. */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = plummer_phi(m, b, r);
            phi[i] = add ? phi[i] + v : v;
        });
    }

    /** Fused batch evaluator: Phi (optional) and Cartesian acceleration a = -grad Phi
        at N positions in ONE kernel (fused per the CLAUDE.md convention).
        phi may be NULL (acceleration-only); acc is packed length 3*N.
        add=true accumulates into the outputs (composite support). */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc,
        double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            T pot, dPhidr;
            plummer_eval(m, b, r, &pot, &dPhidr, (T*)NULL);
            T a[3];
            sph_acc_car(dPhidr, x, y, z, r, a);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            for(int k=0; k<3; k++)
                acc[i*3+k] = add ? acc[i*3+k] + a[k] : a[k];
        });
    }

    /** Batch density evaluator at N Cartesian positions via the plummer_rho leaf.
        add=true accumulates into rho[] (composite support: composite density = sum). */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = plummer_rho(m, b, r);
            rho[i] = add ? rho[i] + v : v;
        });
    }

    /** Export constructor parameters for the GPU force descriptor (Tier 3);
        layout consumed by gpu_term_phi_acc in potential_descriptor.h. */
    void gpuTermParams(double p[]) const { p[0] = mass;  p[1] = scaleRadius; }
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
        const T* xyz, /*out*/ T* phi, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = isochrone_phi(m, b, r);
            phi[i] = add ? phi[i] + v : v;
        });
    }

    /** Fused batch evaluator: Phi (optional, may be NULL) + Cartesian acceleration,
        one kernel. acc packed length 3*N; add=true accumulates (composite support). */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc,
        double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            T pot, dPhidr;
            isochrone_eval(m, b, r, &pot, &dPhidr, (T*)NULL);
            T a[3];
            sph_acc_car(dPhidr, x, y, z, r, a);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            for(int k=0; k<3; k++)
                acc[i*3+k] = add ? acc[i*3+k] + a[k] : a[k];
        });
    }

    /** Batch density evaluator via the isochrone_rho leaf; add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = isochrone_rho(m, b, r);
            rho[i] = add ? rho[i] + v : v;
        });
    }

    /** Export constructor parameters for the GPU force descriptor (Tier 3);
        layout consumed by gpu_term_phi_acc in potential_descriptor.h. */
    void gpuTermParams(double p[]) const { p[0] = mass;  p[1] = scaleRadius; }
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
        const T* xyz, /*out*/ T* phi, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), rs = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = nfw_phi(m, rs, r);
            phi[i] = add ? phi[i] + v : v;
        });
    }

    /** Fused batch evaluator: Phi (optional, may be NULL) + Cartesian acceleration,
        one kernel (Tier 3 on-ramp: the orbit integrator calls this same nfw_eval
        leaf per step). acc packed length 3*N; add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc,
        double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), rs = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            T pot, dPhidr;
            nfw_eval(m, rs, r, &pot, &dPhidr, (T*)NULL);
            T a[3];
            sph_acc_car(dPhidr, x, y, z, r, a);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            for(int k=0; k<3; k++)
                acc[i*3+k] = add ? acc[i*3+k] + a[k] : a[k];
        });
    }

    /** Batch density evaluator via the nfw_rho leaf; add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), rs = static_cast<T>(scaleRadius);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            const T v = nfw_rho(m, rs, r);
            rho[i] = add ? rho[i] + v : v;
        });
    }

    /** Export constructor parameters for the GPU force descriptor (Tier 3);
        layout consumed by gpu_term_phi_acc in potential_descriptor.h. */
    void gpuTermParams(double p[]) const { p[0] = mass;  p[1] = scaleRadius; }
private:
    const double mass;         ///< normalization factor  (M);  equals to mass enclosed within ~5.3r_s
    const double scaleRadius;  ///< scale radius of the NFW model  (r_s)

    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const
    { return nfw_rho(mass, scaleRadius, pos.r); }   // single source: the same leaf as the batch path
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
        const T* xyz, /*out*/ T* phi, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), a = static_cast<T>(scaleRadius), b = static_cast<T>(scaleHeight);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T R = std::sqrt(x*x + y*y);
            const T v = miyamoto_nagai_phi(m, a, b, R, z);
            phi[i] = add ? phi[i] + v : v;
        });
    }

    /** Fused batch evaluator: Phi (optional, may be NULL) + Cartesian acceleration,
        one kernel. Cylindrical leaf derivatives (dR, dz) are converted to Cartesian
        via cyl_acc_car. acc packed length 3*N; add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc,
        double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), ar = static_cast<T>(scaleRadius), b = static_cast<T>(scaleHeight);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T R = std::sqrt(x*x + y*y);
            T pot, dPhidR, dPhidz;
            miyamoto_nagai_eval(m, ar, b, R, z,
                &pot, &dPhidR, &dPhidz, (T*)NULL, (T*)NULL, (T*)NULL);
            T a[3];
            cyl_acc_car(dPhidR, dPhidz, x, y, R, a);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            for(int k=0; k<3; k++)
                acc[i*3+k] = add ? acc[i*3+k] + a[k] : a[k];
        });
    }

    /** Batch density evaluator via the miyamoto_nagai_rho leaf; add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, double /*time*/ = 0, bool add = false) const
    {
        const T m = static_cast<T>(mass), ar = static_cast<T>(scaleRadius), b = static_cast<T>(scaleHeight);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T R = std::sqrt(x*x + y*y);
            const T v = miyamoto_nagai_rho(m, ar, b, R, z);
            rho[i] = add ? rho[i] + v : v;
        });
    }

    /** Export constructor parameters for the GPU force descriptor (Tier 3);
        layout consumed by gpu_term_phi_acc in potential_descriptor.h. */
    void gpuTermParams(double p[]) const
    { p[0] = mass;  p[1] = scaleRadius;  p[2] = scaleHeight; }
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
        const T* xyz, /*out*/ T* phi, double /*time*/ = 0, bool add = false) const
    {
        const T v2 = static_cast<T>(v0squared), c2 = static_cast<T>(coreRadius2);
        const T pp = static_cast<T>(p2), qq = static_cast<T>(q2), L2 = static_cast<T>(lengthUnit2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T v = logarithmic_phi(v2, c2, pp, qq, L2, x, y, z);
            phi[i] = add ? phi[i] + v : v;
        });
    }

    /** Fused batch evaluator: Phi (optional, may be NULL) + Cartesian acceleration
        (triaxial: gradient computed directly in Cartesian coordinates), one kernel.
        acc packed length 3*N; add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc,
        double /*time*/ = 0, bool add = false) const
    {
        const T v2 = static_cast<T>(v0squared), c2 = static_cast<T>(coreRadius2);
        const T pp = static_cast<T>(p2), qq = static_cast<T>(q2), L2 = static_cast<T>(lengthUnit2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            T pot, g[3];
            logarithmic_eval(v2, c2, pp, qq, L2, x, y, z, &pot, g, (T*)NULL);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            for(int k=0; k<3; k++)
                acc[i*3+k] = add ? acc[i*3+k] - g[k] : -g[k];
        });
    }

    /** Batch density evaluator via the logarithmic_rho leaf (Poisson/Laplacian route,
        matching the CPU path); add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, double /*time*/ = 0, bool add = false) const
    {
        const T v2 = static_cast<T>(v0squared), c2 = static_cast<T>(coreRadius2);
        const T pp = static_cast<T>(p2), qq = static_cast<T>(q2), L2 = static_cast<T>(lengthUnit2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T v = logarithmic_rho(v2, c2, pp, qq, L2, x, y, z);
            rho[i] = add ? rho[i] + v : v;
        });
    }

    /** Export constructor parameters for the GPU force descriptor (Tier 3);
        layout consumed by gpu_term_phi_acc in potential_descriptor.h. */
    void gpuTermParams(double p[]) const
    { p[0] = v0squared;  p[1] = coreRadius2;  p[2] = p2;  p[3] = q2;  p[4] = lengthUnit2; }
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
        const T* xyz, /*out*/ T* phi, double /*time*/ = 0, bool add = false) const
    {
        const T w2 = static_cast<T>(Omega2), pp = static_cast<T>(p2), qq = static_cast<T>(q2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T v = harmonic_phi(w2, pp, qq, x, y, z);
            phi[i] = add ? phi[i] + v : v;
        });
    }

    /** Fused batch evaluator: Phi (optional, may be NULL) + Cartesian acceleration
        (triaxial: gradient computed directly in Cartesian coordinates), one kernel.
        acc packed length 3*N; add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc,
        double /*time*/ = 0, bool add = false) const
    {
        const T w2 = static_cast<T>(Omega2), pp = static_cast<T>(p2), qq = static_cast<T>(q2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            T pot, g[3];
            harmonic_eval(w2, pp, qq, x, y, z, &pot, g, (T*)NULL);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            for(int k=0; k<3; k++)
                acc[i*3+k] = add ? acc[i*3+k] - g[k] : -g[k];
        });
    }

    /** Batch density evaluator via the harmonic_rho leaf (Poisson/Laplacian route,
        matching the CPU path); add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, double /*time*/ = 0, bool add = false) const
    {
        const T w2 = static_cast<T>(Omega2), pp = static_cast<T>(p2), qq = static_cast<T>(q2);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T v = harmonic_rho(w2, pp, qq, x, y, z);
            rho[i] = add ? rho[i] + v : v;
        });
    }

    /** Export constructor parameters for the GPU force descriptor (Tier 3);
        layout consumed by gpu_term_phi_acc in potential_descriptor.h. */
    void gpuTermParams(double p[]) const { p[0] = Omega2;  p[1] = p2;  p[2] = q2; }
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