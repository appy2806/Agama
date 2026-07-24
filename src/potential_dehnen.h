/** \file    potential_dehnen.h
    \brief   Triaxial Dehnen potential
    \author  Eugene Vasiliev
    \date    2009-2015
**/
#pragma once
#include "potential_base.h"
#include "potential_analytic.h"  // sph_acc_car (Cartesian acceleration helper) + gpu_policy pull-through
#include "gpu_policy.h"          // agama::forall for templated batch evaluators (Tier 1)
#include <cmath>
#include <stdexcept>

namespace potential {

// =====================================================================
// Tier 1 leaf math: AGAMA_DEVICE_INLINE, single-source with Dehnen::evalCar /
// Dehnen::densityCar (potential_dehnen.cpp). Dehnen can be TRIAXIAL (axisRatioY,
// axisRatioZ != 1); the CPU code only has a closed-form expression for the
// potential and its derivatives in the SPHERICAL case (axisRatioY==axisRatioZ==1)
// -- the triaxial case is computed via math::integrate (numerical quadrature),
// which has no device path and is NOT represented here. dehnen_eval is therefore
// only valid for the spherical case; dehnen_rho (the density) has a genuine
// closed form for arbitrary axis ratios and is fully general.
// =====================================================================

/** Dehnen (1993) SPHERICAL double power-law potential (axisRatioY=axisRatioZ=1
    only -- see file header comment). Phi(r), dPhi/dr, and the Merritt&Fridman
    (1996) "val2" combination used by the CPU Cartesian-Hessian assembly
    (NOT the raw d^2Phi/dr^2 -- deriv2 is never requested by the GPU batch path,
    only by Dehnen::evalCar's own Hessian conversion, so it is passed through
    exactly as the original code computed it, for bit-for-bit equivalence). */
template<typename T>
AGAMA_DEVICE_INLINE void dehnen_eval(T mass, T scalerad, T gamma, T r,
    T* potential, T* deriv, T* deriv2)
{
    T s = scalerad / r;
    if(potential) {
        *potential = s > T(2e-3) / (T(4) - gamma) ?
            mass/scalerad * (gamma == T(2) ? -std::log(T(1)+s) :
                (T(1) - math::powT(T(1)+s, gamma-T(2))) / (gamma-T(2))) :
            // asymptotic expansion for r->infinity, or equivalently s->0
            mass/scalerad * -s * (T(1) + s*(gamma-T(3))/T(2) * (T(1) + s*(gamma-T(4))/T(3) *
                (T(1) + s*(gamma-T(5))/T(4))));
    }
    if(!deriv && !deriv2)
        return;
    T val = mass * math::powT(r, T(1)-gamma) * math::powT(r+scalerad, gamma-T(3));
    if(deriv)
        *deriv = val;
    if(deriv2)
        *deriv2 = -val * (gamma*s + T(3)) / (r+scalerad);
}

/** Dehnen density (explicit closed form, same as Dehnen::densityCar); unlike
    dehnen_eval above, this is valid for the FULL triaxial case (arbitrary
    axisRatioY, axisRatioZ), since the density profile itself never required
    numerical quadrature -- only the potential's triaxial branch does. */
template<typename T>
AGAMA_DEVICE_INLINE T dehnen_rho(T mass, T scalerad, T gamma, T axisRatioY, T axisRatioZ,
    T x, T y, T z)
{
    T m = std::sqrt(pow_2(x) + pow_2(y/axisRatioY) + pow_2(z/axisRatioZ));
    return mass * scalerad * (T(3)-gamma) / (T(4*M_PI)*axisRatioY*axisRatioZ) *
        math::powT(m, -gamma) * math::powT(scalerad+m, gamma-T(4));
}

/** Dehnen(1993) double power-law model **/
class Dehnen: public BasePotentialCar {
public:
    Dehnen(double _mass, double _scalerad, double _gamma, double _axisRatioY=1., double _axisRatioZ=1.);
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Dehnen"; }
    virtual coord::SymmetryType symmetry() const {
        return (axisRatioY==1 ?
            (axisRatioZ==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL); }
    virtual double totalMass() const { return mass; }

    /** Tier 1 batch evaluator: Phi at N Cartesian positions via the dehnen_eval
        leaf. SPHERICAL CASE ONLY: throws std::runtime_error for a triaxial
        instance instead of silently computing the wrong potential (the
        triaxial branch needs math::integrate, which has no device path).
        The GPU dispatch layer (potential_gpu.cpp / potential_descriptor.h)
        checks isSpherical(symmetry()) before ever reaching this method, so a
        triaxial Dehnen is reported as unsupported (POT_GPU_EUNSUPP) rather
        than throwing in practice; the guard here is a second, defensive line
        in case this method is ever called directly. add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, bool add = false) const
    {
        if(axisRatioY != 1. || axisRatioZ != 1.)
            throw std::runtime_error("Dehnen: GPU batch evaluation only supports the "
                "spherical case (axisRatioY=axisRatioZ=1); the triaxial potential requires "
                "CPU quadrature with no device path");
        const T m = static_cast<T>(mass), b = static_cast<T>(scalerad), g = static_cast<T>(gamma);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            T pot;
            dehnen_eval(m, b, g, r, &pot, (T*)NULL, (T*)NULL);
            phi[i] = add ? phi[i] + pot : pot;
        });
    }

    /** Fused batch evaluator: Phi (optional) + Cartesian acceleration, one
        kernel. SPHERICAL CASE ONLY -- see evalmanyCarT. add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc, bool add = false) const
    {
        if(axisRatioY != 1. || axisRatioZ != 1.)
            throw std::runtime_error("Dehnen: GPU batch evaluation only supports the "
                "spherical case (axisRatioY=axisRatioZ=1); the triaxial potential requires "
                "CPU quadrature with no device path");
        const T m = static_cast<T>(mass), b = static_cast<T>(scalerad), g = static_cast<T>(gamma);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T r = std::sqrt(x*x + y*y + z*z);
            T pot, dPhidr;
            dehnen_eval(m, b, g, r, &pot, &dPhidr, (T*)NULL);
            T a[3];
            sph_acc_car(dPhidr, x, y, z, r, a);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            for(int k=0; k<3; k++)
                acc[i*3+k] = add ? acc[i*3+k] + a[k] : a[k];
        });
    }

    /** Batch density evaluator via the dehnen_rho leaf; unlike the two methods
        above, this is valid for the FULL triaxial case (density has a closed
        form for any axis ratio). Not currently exposed through the shared GPU
        dispatch tables for a triaxial instance regardless (potential_gpu.cpp
        gates the whole Dehnen type on sphericity, for one uniform per-type
        capability check across Phi/acc/density), but the method itself
        computes the correct triaxial value if called directly. add=true
        accumulates. */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, bool add = false) const
    {
        const T m = static_cast<T>(mass), b = static_cast<T>(scalerad), g = static_cast<T>(gamma);
        const T ay = static_cast<T>(axisRatioY), az = static_cast<T>(axisRatioZ);
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            const T v = dehnen_rho(m, b, g, ay, az, x, y, z);
            rho[i] = add ? rho[i] + v : v;
        });
    }

    /** Export constructor parameters for the GPU force descriptor (Tier 3);
        only meaningful/called for the spherical case -- buildGpuPotDesc
        (potential_descriptor.h) checks isSpherical(symmetry()) first.
        layout: { mass, scalerad, gamma }. */
    void gpuTermParams(double p[]) const { p[0] = mass;  p[1] = scalerad;  p[2] = gamma; }
private:
    const double mass;       ///< total mass of the model
    const double scalerad;   ///< scale radius
    const double gamma;      ///< cusp exponent for Dehnen potential
    const double axisRatioY; ///< axis ratio y/x of equidensity surfaces
    const double axisRatioZ; ///< axis ratio z/x of equidensity surfaces

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
};

}  // namespace
