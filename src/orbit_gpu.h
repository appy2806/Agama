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
    ORBIT_GPU_EUNSUPP   = 3,  ///< potential not representable as a GPU force descriptor
    /** the potential IS representable, but only at a single instant: it contains a
        time-varying modifier (a Shifted/Rotating/Scaled whose center/angle/
        amplitude/scale actually changes with time). The orbit kernel builds the
        descriptor once and then integrates across many times, so baking in one
        instant would silently integrate the wrong potential -- this is reported
        instead. A CONSTANT modifier chain, and Tilted in any case, are fine.
        Distinguished from EUNSUPP so the Python layer can say which of the two
        it is: "this potential type has no GPU path" and "this potential's time
        dependence has no GPU path yet" call for different user actions. */
    ORBIT_GPU_ETIMEDEP  = 4
};

/** ODE integrator selectable on the batch path. Kept as a plain int on this
    boundary header (no dependency on orbit.h's OrbitIntParams::Method); the
    Python binding maps the method string to one of these. Only the two
    force-only schemes are exposed: Hermite needs the potential Hessian (jerk =
    -d(grad)/dx . v) which the GPU force descriptor does not provide. */
enum OrbitGPUMethod {
    ORBIT_GPU_DOP853 = 0,  ///< 8th-order Runge-Kutta (1st-order ODE, 6D state)
    ORBIT_GPU_DPRKN8 = 1   ///< 8th-order Runge-Kutta-Nystrom (2nd-order ODE)
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
                       (OrbitIntParams::accuracy; accAbs is 0). For the DPRKN8
                       method the same empirical rescaling as the CPU stepper
                       (10 * accuracy^0.9) is applied internally, so the value
                       passed here has the identical meaning across methods.
    \param  maxNumSteps upper limit on the number of ODE steps per orbit
    \param  traj       output, packed Norb*trajsize*6:
                       traj[(i*trajsize + j)*6 + k] = component k of sample j
                       of orbit i. Samples not reached before an integrator
                       error or the step limit are filled with NAN.
    \param  device     "cpu"/"openmp" (OpenMP), "serial", or "cuda"
    \param  method     an OrbitGPUMethod value (DOP853 or DPRKN8)
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
                       const char* device,
                       int method = ORBIT_GPU_DOP853,
                       /** per-orbit absolute start time, length Norb, in internal units;
                           NULL means all zeros. Only affects a TIME-DEPENDENT potential:
                           it is the absolute time at which each orbit's integration
                           begins, which the kernel adds to the ODE cores' per-step time
                           offsets before asking the potential for a force. Mirrors
                           OrbitIntegrator's timeBegin. */
                       const double* timeStart = NULL);

/** Device-resident-output variant of integrateOrbitsGPU: the trajectory is left in
    a caller-supplied GPU buffer and never copied to the host.

    This exists because the output is where the volume is. The initial conditions are
    6*Norb doubles (2.4 MB at Norb=50k) but the trajectory is Norb*trajsize*6 values
    (307 MB at Norb=50k, trajsize=256, fp32) — two orders of magnitude more. A
    workflow that integrates orbits and then evaluates a DF or bins a density on the
    GPU should never round-trip that through host memory: the host output path was
    measured at roughly 1 GB/s effective and was 58% of fp32 orbit wall time before
    commit `ec4f269`, against on-device bandwidth two orders of magnitude higher.

    Everything except the trajectory destination stays a host pointer: `ic`, `times`
    and `timeStart` are all O(Norb) and are uploaded internally exactly as the
    host-output entry point does, so there is no benefit to making the caller manage
    them and no change in their meaning.

    Synchronization contract, matching `potential::evalPotentialGPUDevice`:
    - if `output_stream` names a real producer stream (nonzero and not the
      __cuda_array_interface__ v3 sentinels 1 = legacy default stream, 2 =
      per-thread default stream), it is cudaStreamSynchronize'd BEFORE the kernel
      launches, so any prior work the producer enqueued on `d_traj` is complete;
    - this call's own stream is always synchronized before returning, so `d_traj` is
      fully written on return and the caller may consume it on any stream.

    Same v1 multi-GPU limitation as the potential device entry points: the
    __cuda_array_interface__ carries no device id, so `d_traj` must live on the same
    device as AGAMA's CUDA context (the calling process's current device).

    `d_traj` holds values in **internal units**, exactly like the `traj` argument of
    the host-output overload — this function performs no unit conversion. Use
    scaleTrajectoryGPUDevice() below when the caller's unit system is non-trivial.

    CUDA-only by construction: there is no such thing as a device-resident buffer
    under device="cpu"/"openmp"/"serial", so this entry point takes no `device`
    argument at all rather than accepting one and rejecting most of its values.
    Returns ORBIT_GPU_ENOTBUILT when built with HAVE_CUDA=0, and ORBIT_GPU_EUNSUPP
    for a NULL `d_traj`.

    \param  d_traj  output DEVICE buffer, length Norb*trajsize*6, same packing and
                    same NAN-fill-on-unreached-sample guarantee as `traj` above. */
template<typename T>
int integrateOrbitsGPUDevice(const potential::BasePotential& pot,
                             std::size_t Norb,
                             const double* ic,
                             const double* times,
                             std::size_t trajsize,
                             double accuracy,
                             std::size_t maxNumSteps,
                             T* d_traj,
                             unsigned long long output_stream,
                             int method = ORBIT_GPU_DOP853,
                             const double* timeStart = NULL);

/** Convert a device-resident trajectory buffer from internal units to the caller's
    unit system, in place: positions are divided by `lengthUnit` and velocities by
    `velocityUnit`.

    Kept separate from integrateOrbitsGPUDevice rather than folded into the orbit
    kernel for two reasons: the integrator's documented contract is that it works
    purely in internal units, and folding a scale into the per-sample store would put
    it on the hot path of every call including the overwhelmingly common one where
    both factors are exactly 1 (no unit system configured), where the caller should
    skip this entirely.

    The division is performed in **double** even when T=float, and by division rather
    than by multiplication by a precomputed reciprocal. That is deliberate: it
    reproduces the host path's `float(w[k] / conv->lengthUnit)` — promote to double,
    divide, round once to float — so a device-resident result is bit-identical to the
    host-resident one rather than merely close. On a consumer card with a 1:64
    fp64:fp32 ratio those are slow double divisions, which is accepted because this is
    one pass over the buffer on a path only taken under a non-trivial unit system.

    \param  n  number of PHASE-SPACE SAMPLES (Norb*trajsize), not the element count;
               each sample is 6 consecutive values [x,y,z,vx,vy,vz]. */
template<typename T>
int scaleTrajectoryGPUDevice(std::size_t n, T* d_traj,
                             double lengthUnit, double velocityUnit,
                             unsigned long long output_stream = 0);

}  // namespace orbit
