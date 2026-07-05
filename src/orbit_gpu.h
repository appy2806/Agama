/** \file    orbit_gpu.h
    \brief   Batch orbit integration through the execution-policy backends
             (Tier 3): the same DOP853 core as the CPU stepper classes, run
             one-thread-per-orbit under Serial / OpenMP / Cuda policies.
    \date    2026
    \author  GPU-unification fork

    Boundary header between the CPython extension (interface_python.cpp, g++)
    and the nvcc-routed implementation (orbit_gpu.cpp, CUDA_TUS): plain C++
    declarations only, no CUDA types.

    Scope (v1):
    - potentials representable by a GpuPotDesc (the analytic GPU-capable set +
      Composite of those; see potential_descriptor.h);
    - static potentials, inertial frame (no pattern speed Omega, no
      time-dependence), Cartesian integration variables;
    - DOP853 with the same adaptive-step logic as orbit::OrbitIntegrator
      (accAbs=0, per-eval accuracy factor from the virial balance, identical
      to OrbitIntegrator<coord::Car>::eval with Omega=0);
    - dense trajectory output at trajsize regular sampling intervals per orbit
      via the 7th-order DOP853 interpolant, reproducing the sample placement
      of orbit::RuntimeTrajectory (sample j of orbit i is at time
      j * times[i] / (trajsize-1), sample 0 = the initial conditions);
      trajsize=1 stores only the final state.
    Unsupported requests must be rejected by the caller beforehand (the Python
    binding raises NotImplementedError, naming the offending feature).
*/
#pragma once
#include <cstddef>

namespace potential { class BasePotential; }

namespace orbit {

/** Result codes returned by integrateOrbitsGPU<T> (mirrors PotentialGPUResult). */
enum OrbitGPUResult {
    ORBIT_GPU_OK        = 0,  ///< success
    ORBIT_GPU_EBADDEV   = 1,  ///< unknown device string (not cpu/openmp/serial/cuda)
    ORBIT_GPU_ENOTBUILT = 2,  ///< device='cuda' but library built with HAVE_CUDA=0
    ORBIT_GPU_EUNSUPP   = 3   ///< potential not representable as a GPU force descriptor
};

/** Integrate Norb orbits in the given potential, one thread per orbit.
    All buffers are host pointers in *internal* units; the caller handles unit
    conversion and (for the Python boundary) array validation.

    \tparam T          working precision of the integration state (float or
                       double; explicitly instantiated for both). Time
                       bookkeeping is always double; coefficient storage in
                       the force descriptor is double, cast once on entry.
    \param  pot        the potential (must be representable: analytic
                       GPU-capable types or Composite thereof)
    \param  Norb       number of orbits
    \param  ic         initial conditions, packed 6*Norb:
                       [x,y,z,vx,vy,vz] per orbit (Cartesian, internal units)
    \param  times      integration time per orbit, length Norb (signed;
                       nonzero — a zero entry yields NAN samples for that orbit)
    \param  trajsize   number of trajectory samples per orbit (>= 1)
    \param  accuracy   relative accuracy parameter of the ODE integrator
                       (OrbitIntParams::accuracy; accAbs is 0)
    \param  maxNumSteps upper limit on the number of ODE steps per orbit
    \param  traj       output, packed Norb*trajsize*6:
                       traj[(i*trajsize + j)*6 + k] = component k of sample j
                       of orbit i. Samples not reached before an integrator
                       error or the step limit are filled with NAN.
    \param  device     "cpu"/"openmp" (OpenMP), "serial", or "cuda"
    \return an OrbitGPUResult code; ORBIT_GPU_OK on success.
*/
template<typename T>
int integrateOrbitsGPU(const potential::BasePotential& pot,
                       std::size_t Norb,
                       const double* ic,
                       const double* times,
                       std::size_t trajsize,
                       double accuracy,
                       std::size_t maxNumSteps,
                       T* traj,
                       const char* device);

}  // namespace orbit
