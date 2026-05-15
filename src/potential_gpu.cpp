/** \file    potential_gpu.cpp
    \brief   GPU-routed dispatch for batch potential evaluation. Compiled by nvcc
             (CUDA_TUS in Makefile.list) when HAVE_CUDA=1; compiles cleanly under
             g++ when HAVE_CUDA=0 with the "cuda" branch returning POT_GPU_ENOTBUILT.
    \author  GPU-unification fork
*/
#include "potential_gpu.h"
#include "potential_analytic.h"
#include "potential_base.h"
#include "gpu_policy.h"
#include <cstring>

namespace potential {

namespace {

/** Try-dispatch on concrete potential type. Returns true if `pot` matched one
    of the known GPU-capable types and the batch eval was invoked. The lambda
    captures `pol`, `xyz_p`, `phi_p` by value; pol determines whether the
    forall() inside each evalmanyCarT becomes a Serial/OpenMP loop or a CUDA
    kernel launch.

    The list below grows as more potentials get migrated. Order doesn't matter
    for correctness (dynamic_cast is exact); listing the most-common first is
    a tiny perf hint.

    Why C++14 generic lambdas: lets the same dispatch body service three
    different Policy types without three copies. The compiler instantiates a
    distinct dispatch function per (T, Policy) pair. */
template<typename T, class Policy>
bool try_dispatch(const BasePotential& pot, Policy pol,
                  std::size_t N, const T* xyz_p, T* phi_p)
{
    #define AGAMA_GPU_TRY(PotClass) \
        if(const PotClass* p = dynamic_cast<const PotClass*>(&pot)) { \
            p->template evalmanyCarT<T>(pol, N, xyz_p, phi_p);        \
            return true;                                              \
        }
    AGAMA_GPU_TRY(NFW)
    AGAMA_GPU_TRY(Plummer)
    AGAMA_GPU_TRY(Isochrone)
    AGAMA_GPU_TRY(MiyamotoNagai)
    AGAMA_GPU_TRY(Logarithmic)
    AGAMA_GPU_TRY(Harmonic)
    #undef AGAMA_GPU_TRY
    return false;
}

}  // anonymous namespace

template<typename T>
int evalPotentialGPU(const BasePotential& pot,
                     std::size_t N, const T* xyz_h, T* phi_h,
                     const char* device)
{
    if(std::strcmp(device, "cpu") == 0) {
        return try_dispatch<T>(pot, agama::Serial{}, N, xyz_h, phi_h)
            ? POT_GPU_OK : POT_GPU_EUNSUPP;
    }
    if(std::strcmp(device, "openmp") == 0) {
        return try_dispatch<T>(pot, agama::OpenMP{}, N, xyz_h, phi_h)
            ? POT_GPU_OK : POT_GPU_EUNSUPP;
    }
    if(std::strcmp(device, "cuda") == 0) {
#ifdef HAVE_CUDA
        agama::device_array<T> d_xyz(N * 3), d_phi(N);
        d_xyz.from_host(xyz_h, N * 3);
        bool ok = try_dispatch<T>(pot, agama::Cuda{}, N, d_xyz.data(), d_phi.data());
        if(!ok) return POT_GPU_EUNSUPP;
        d_phi.to_host(phi_h, N);
        return POT_GPU_OK;
#else
        return POT_GPU_ENOTBUILT;
#endif
    }
    return POT_GPU_EBADDEV;
}

// Explicit instantiations for the two precisions the Python boundary exposes.
template int evalPotentialGPU<float >(const BasePotential&, std::size_t,
                                      const float*,  float*,  const char*);
template int evalPotentialGPU<double>(const BasePotential&, std::size_t,
                                      const double*, double*, const char*);

}  // namespace potential
