/** \file    potential_gpu.h
    \brief   Boundary between the CPython extension (interface_python.cpp, g++) and
             the GPU-routed potential dispatch (potential_gpu.cpp, nvcc).
    \date    2026
    \author  GPU-unification fork
*/
#pragma once
#include <cstddef>
#include <string>

namespace potential {

class BasePotential;  // forward; full def in potential_base.h

/** Result codes returned by evalPotentialGPU<T>. Stable; interface_python.cpp
    compares against these to raise the appropriate Python exception. */
enum PotentialGPUResult {
    POT_GPU_OK         = 0,  ///< success
    POT_GPU_EBADDEV    = 1,  ///< unknown device string (not cpu/openmp/serial/cuda)
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
    \param  device   one of "cpu" (OpenMP -- the legacy-equivalent parallel CPU
                     path), "openmp" (alias of "cpu"), "serial" (single-thread
                     debugging/baseline), or "cuda"
    \returns         a PotentialGPUResult code; POT_GPU_OK on success.

    On POT_GPU_EUNSUPP, the caller should query `pot.name()` (or, better,
    unsupportedGPUPotentialName(pot) below, which recurses into composites)
    for the type name when constructing the user-facing NotImplementedError
    message. */
template<typename T>
int evalPotentialGPU(const BasePotential& pot,
                     std::size_t N, const T* xyz, T* phi,
                     const char* device);

/** For a POT_GPU_EUNSUPP result: the name of the potential type that blocked
    the dispatch. For a plain potential this is just pot.name(); for a
    Composite it recurses into the members and names the FIRST one that is not
    GPU-capable (annotated with its component index), which is far more useful
    in the error message than the composite's joined name. */
std::string unsupportedGPUPotentialName(const BasePotential& pot);

/** Compute Phi at N Cartesian positions that are ALREADY resident in GPU
    memory (Phase B: `__cuda_array_interface__` passthrough). No host<->device
    copies, no scratch buffers, no mutex — the kernel reads/writes the caller's
    device pointers directly. Coordinates are in *internal* units; the caller
    in interface_python.cpp rejects non-trivial unit systems for this path.

    Synchronization contract (simple synchronous v1 semantics):
    - if `input_stream` names a real producer stream (nonzero and not the CAI-v3
      sentinels 1 = legacy default stream, 2 = per-thread default stream), we
      cudaStreamSynchronize it BEFORE launching, so writes to d_xyz enqueued by
      the producer are visible to our kernel (which runs on the default stream);
    - we always cudaStreamSynchronize the default stream AFTER the kernel, so
      d_phi is fully written when this function returns and the Python caller
      may consume it on any stream without further ordering.

    v1 multi-GPU limitation: the __cuda_array_interface__ carries no device id,
    so the input buffers must reside on the same device as AGAMA's CUDA context
    (the current device of the calling process); passing pointers from another
    GPU is undefined behaviour.

    \tparam T            precision (float or double); explicitly instantiated for both
    \param  pot          the potential object (any concrete subclass of BasePotential)
    \param  N            number of input positions
    \param  d_xyz        packed input DEVICE buffer, length 3*N (C-contiguous)
    \param  d_phi        output DEVICE buffer, length N
    \param  input_stream the 'stream' entry of the producer's __cuda_array_interface__
                         (0 if the key was absent)
    \returns             a PotentialGPUResult code; POT_GPU_OK on success.
                         POT_GPU_ENOTBUILT when the library was built with HAVE_CUDA=0. */
template<typename T>
int evalPotentialGPUDevice(const BasePotential& pot,
                           std::size_t N, const T* d_xyz, T* d_phi,
                           unsigned long long input_stream);

}  // namespace potential
