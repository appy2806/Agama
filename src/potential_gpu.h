/** \file    potential_gpu.h
    \brief   Boundary between the CPython extension (interface_python.cpp, g++) and
             the GPU-routed potential dispatch (potential_gpu.cpp, nvcc).
    \date    2026
    \author  GPU-unification fork
*/
#pragma once
#include <cstddef>

namespace potential {

class BasePotential;  // forward; full def in potential_base.h

/** Result codes returned by evalPotentialGPU<T>. Stable; interface_python.cpp
    compares against these to raise the appropriate Python exception. */
enum PotentialGPUResult {
    POT_GPU_OK         = 0,  ///< success
    POT_GPU_EBADDEV    = 1,  ///< unknown device string (not cpu/openmp/cuda)
    POT_GPU_ENOTBUILT  = 2,  ///< device='cuda' but library built with HAVE_CUDA=0
    POT_GPU_EUNSUPP    = 3   ///< this potential type is not GPU-capable yet
};

/** Compute Phi at N Cartesian positions through the templated batch path.
    Coordinates are in *internal* units (length=1, mass=G=1); the caller in
    interface_python.cpp handles unit conversion at the Python boundary.

    \tparam T        precision (float or double); explicitly instantiated for both
    \param  pot      the potential object (any concrete subclass of BasePotential)
    \param  N        number of input positions
    \param  xyz      packed input host buffer, length 3*N: [x0,y0,z0, x1,y1,z1, ...]
    \param  phi      output host buffer, length N
    \param  device   one of "cpu" (Serial), "openmp", or "cuda"
    \returns         a PotentialGPUResult code; POT_GPU_OK on success.

    On POT_GPU_EUNSUPP, the caller should query `pot.name()` for the type name
    when constructing the user-facing NotImplementedError message. */
template<typename T>
int evalPotentialGPU(const BasePotential& pot,
                     std::size_t N, const T* xyz, T* phi,
                     const char* device);

}  // namespace potential
