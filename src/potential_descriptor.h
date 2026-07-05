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
*/
#pragma once
#include "potential_base.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "gpu_device.h"
#include <cmath>

namespace potential {

/// concrete potential types representable in a GpuPotTerm;
/// must match the GPU-capable set in potential_gpu.cpp (AGAMA_GPU_POT_LIST)
enum GpuPotTag {
    GPU_POT_PLUMMER,
    GPU_POT_ISOCHRONE,
    GPU_POT_NFW,
    GPU_POT_MIYAMOTONAGAI,
    GPU_POT_LOGARITHMIC,
    GPU_POT_HARMONIC
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
      HARMONIC                  : p = { Omega2, p2, q2 }                     */
template<typename T>
struct GpuPotTerm {
    int tag;   ///< a GpuPotTag value
    T p[5];    ///< constructor parameters, layout per tag (unused slots zero)
};

/** Flattened by-value snapshot of a (possibly Composite) potential. */
template<typename T>
struct GpuPotDesc {
    int nterms;
    GpuPotTerm<T> terms[GPU_POT_DESC_MAX_TERMS];
};

/** Accumulate Phi (optional) and the Cartesian acceleration a = -grad Phi of a
    single term into *phi / acc[3]. The per-tag bodies are the same
    leaf-plus-glue sequences as the corresponding evalmanyPhiAccCarT lambdas in
    potential_analytic.h. */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_term_phi_acc(const GpuPotTerm<T>& t,
    T x, T y, T z, /*accumulated, nullable*/ T* phi, /*accumulated*/ T acc[3])
{
    T pot = 0, a[3];
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
    default:  // unreachable if the descriptor was built by buildGpuPotDesc
        a[0] = a[1] = a[2] = 0;
        break;
    }
    if(phi) *phi += pot;
    acc[0] += a[0];
    acc[1] += a[1];
    acc[2] += a[2];
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

/** Host-side builder: flatten `pot` (recursing into Composite) into `desc`.
    \return true on success; false if any member is not a representable type
    or the flattened member count exceeds GPU_POT_DESC_MAX_TERMS -- in which
    case the caller should fall back to the CPU path (desc is left partially
    filled and must not be used). Use potential::unsupportedGPUPotentialName()
    for the user-facing error message. */
inline bool buildGpuPotDesc(const BasePotential& pot, GpuPotDesc<double>& desc,
    bool topLevel = true)
{
    if(topLevel)
        desc.nterms = 0;
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!buildGpuPotDesc(*comp->component(c), desc, false))
                return false;
        return true;
    }
    if(desc.nterms >= GPU_POT_DESC_MAX_TERMS)
        return false;
    GpuPotTerm<double>& term = desc.terms[desc.nterms];
    for(int k = 0; k < 5; k++)
        term.p[k] = 0;
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
    else
        return false;
    desc.nterms++;
    return true;
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
        for(int k = 0; k < 5; k++)
            out.terms[c].p[k] = static_cast<T>(d.terms[c].p[k]);
    }
    return out;
}

}  // namespace potential
