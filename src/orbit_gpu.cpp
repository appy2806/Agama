/** \file    orbit_gpu.cpp
    \brief   Batch orbit integration under the execution-policy backends
             (Tier 3). Compiled by nvcc (CUDA_TUS in Makefile.list) when
             HAVE_CUDA=1; compiles cleanly under g++ when HAVE_CUDA=0 with the
             "cuda" branch returning ORBIT_GPU_ENOTBUILT.
    \author  GPU-unification fork

    The per-orbit body is the header-only DOP853 core from math_ode.h -- the
    SAME dop853_init / dop853_step / dop853_dense that the CPU class
    OdeStepperDOP853 wraps -- driven by a force callable that evaluates the
    potential through the tagged-union GpuPotDesc (potential_descriptor.h),
    which in turn calls the same AGAMA_DEVICE_INLINE leaves as every other
    backend. One thread per orbit; adaptive timesteps diverge across threads
    and we deliberately accept the intra-warp divergence cost (design decision
    recorded in project_report.md: N >> 1 orbits saturate the card anyway).

    v1 execution model: a single kernel launch integrates every orbit to
    completion. On WSL2 / WDDM display GPUs a multi-second launch can trip the
    Windows TDR watchdog; if that is ever observed, the escape hatch is to
    chunk the integration (persist per-orbit state in global memory, relaunch
    until done) -- not implemented until measured necessary.
*/
#include "orbit_gpu.h"
#include "math_ode.h"
#include "potential_descriptor.h"
#include "gpu_policy.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <limits>
#include <vector>

namespace orbit {

namespace {

/** Force callable for the DOP853 core: the equations of motion
    dw/dt = {v, -grad Phi} for w = {x,y,z,vx,vy,vz}, with the same optional
    accuracy factor as OrbitIntegrator<coord::Car>::eval (Omega = 0):
    accFac = min(1, |Epot+Ekin| / max(|Epot|, Ekin)).

    The ODE cores hand the force a time OFFSET relative to the start of the
    current step (see the `time offset` comments in math_ode.h), so the absolute
    time a time-dependent potential needs is timeBegin + offset -- exactly what
    OrbitIntegrator<coord::Car>::eval does on the CPU with its own timeBegin.
    The caller updates timeBegin before each step.

    `desc` is a POINTER, not a copy: the descriptor is a few kilobytes and both
    this functor and GpuDescForce2 would otherwise duplicate it into the kernel
    argument space. Each thread instead builds its own tiny functor pointing at
    the single captured descriptor, which is also what makes timeBegin safe to
    mutate per thread.

    ORDER is the descriptor's compile-time Multipole scratch cap (see GpuDescScratch
    in potential_descriptor.h). It is 0 for every descriptor without a Multipole
    term, which keeps the analytic/modifier orbit kernels byte-for-byte the kernels
    they were -- 580 T of per-thread local memory next to DOP853's state[60]+xt[60]
    is not something to charge to every orbit run. run_batch() picks the
    instantiation on the host from gpuDescNeedsMultipole(). */
template<typename T, int ORDER>
struct GpuDescForce {
    const potential::GpuPotDesc<T>* desc;
    double timeBegin;   ///< absolute time at the start of the current step

    AGAMA_DEVICE_INLINE void operator()(T t, const T w[], T dwdt[], T* accFac) const
    {
        T Epot, acc[3];
        potential::gpu_desc_phi_acc_ord<T, ORDER>(*desc, w[0], w[1], w[2],
            accFac ? &Epot : (T*)NULL, acc, T(timeBegin + t));
        // time derivative of position
        dwdt[0] = w[3];
        dwdt[1] = w[4];
        dwdt[2] = w[5];
        // time derivative of velocity
        dwdt[3] = acc[0];
        dwdt[4] = acc[1];
        dwdt[5] = acc[2];
        if(accFac) {
            T Ekin = T(0.5) * (pow_2(w[3]) + pow_2(w[4]) + pow_2(w[5]));
            *accFac = std::fmin(T(1), std::fabs(Epot + Ekin) /
                std::fmax(std::fabs(Epot), Ekin));
        }
    }
};

/** Second-order force callable for the DPRKN8 core: the acceleration
    d2x/dt2 = -grad Phi(x) as a function of position, mirroring
    OrbitIntegrator<coord::Car>::eval2 with Omega = 0. The signature is
    void force2(t, x[], d2xdt2[], d3xdt3, accFac):
    - the three RK-stage evaluations pass x = positions only (3 elements) and
      request neither the jerk nor the accuracy factor;
    - the end-of-step evaluation requests accFac, and (exactly as in the CPU
      OdeRhs2 path) is handed a buffer whose positions are immediately followed
      in memory by the step's velocities, so x[3..5] are the velocities used
      for Ekin -- see dprkn8_step's contiguous xn|vn scratch layout.
    d3xdt3 (jerk) is never requested here: it needs the potential Hessian, which
    the force descriptor does not carry, and only the Hermite scheme uses it. */
template<typename T, int ORDER>
struct GpuDescForce2 {
    const potential::GpuPotDesc<T>* desc;
    double timeBegin;   ///< absolute time at the start of the current step

    AGAMA_DEVICE_INLINE void operator()(T t, const T x[], T d2xdt2[],
        T* /*d3xdt3 (unused: Hermite-only)*/, T* accFac) const
    {
        T Epot, acc[3];
        potential::gpu_desc_phi_acc_ord<T, ORDER>(*desc, x[0], x[1], x[2],
            accFac ? &Epot : (T*)NULL, acc, T(timeBegin + t));
        d2xdt2[0] = acc[0];
        d2xdt2[1] = acc[1];
        d2xdt2[2] = acc[2];
        if(accFac) {
            // x[3..5] are the velocities (contiguous with the positions, see above)
            T Ekin = T(0.5) * (pow_2(x[3]) + pow_2(x[4]) + pow_2(x[5]));
            *accFac = std::fmin(T(1), std::fabs(Epot + Ekin) /
                std::fmax(std::fabs(Epot), Ekin));
        }
    }
};

/** Integrate a single orbit and store its sampled trajectory.
    Runs identically as the body of a CUDA thread or a (Serial/OpenMP) CPU
    loop iteration. Integration state is in precision T; time bookkeeping is
    always double (an fp32 running time would lose sampling accuracy over
    thousands of steps). The trajectory sample placement mirrors
    orbit::RuntimeTrajectory::processTimestep with t0 = 0: sample j lies at
    time sign * interval * j, interval = |totalTime| / (trajsize-1); samples
    that the integration never reaches (stepper error / step limit) stay NAN.

    TOut (defaults to T) is the STORAGE precision of `traj`: every value that
    would otherwise be written as T is narrowed to TOut at the point of store
    (dense-output sample, NAN prefill, and the trajsize==1 final-state copy
    alike), so an fp64 integration can write directly into an fp32 destination
    with no separate host-side narrowing pass. See orbit_gpu.h for why this is
    only safe when no unit scaling is pending. */
template<typename T, typename TOut, int ORDER>
AGAMA_DEVICE_INLINE void integrate_one_orbit(GpuDescForce<T, ORDER> force,
    const T ic6[6], double timeStart, double totalTime, T accuracy,
    unsigned long long maxNumSteps, std::size_t trajsize, TOut* traj)
{
    // `force` is taken BY VALUE (it is a pointer plus a scalar) so this thread can
    // advance its timeBegin without disturbing any other thread.
    force.timeBegin = timeStart;
    const int NDIM = 6;
    // pre-fill with NAN: anything not overwritten below signals "not reached"
    for(std::size_t j = 0; j < trajsize * 6; j++)
        traj[j] = TOut(NAN);

    T state[60],  // persistent DOP853 storage: x, dx/dt, 8 dense-output blocks
      xt[60];     // per-step scratch (also covers the 4*NDIM init scratch)
    T nextTimeStep = 0;
    math::dop853_init(force, NDIM, ic6, accuracy, /*accAbs*/ T(0),
        state, nextTimeStep, xt);

    const double sign = totalTime >= 0 ? +1 : -1;
    // positive sampling interval (as in RuntimeTrajectory); unused if trajsize==1
    const double interval = trajsize > 1 ? sign * totalTime / double(trajsize - 1) : 0;
    // roundoff guard on the sample-index upper bound, as in RuntimeTrajectory
    // (ROUNDOFF = 10*DBL_EPSILON there; time bookkeeping here is double too)
    const double ROUNDOFF = 10 * DBL_EPSILON;

    double tcur = 0;
    bool ok = totalTime != 0;
    unsigned long long numSteps = 0;
    while(tcur != totalTime) {
        double timeRemaining = totalTime - tcur;
        // the stepper's stage times are offsets from here
        force.timeBegin = timeStart + tcur;
        T h = math::dop853_step(force, NDIM, accuracy, /*accAbs*/ T(0),
            state, xt, nextTimeStep, T(timeRemaining));
        if(!(double(h) * sign > 0)) {   // stepper signalled an error
            ok = false;
            break;
        }
        double hd   = double(h);
        double tend = h == T(timeRemaining) ? totalTime : tcur + hd;
        if(trajsize > 1) {
            // samples falling within the just-completed step [tcur, tend]
            double dtroundoff = ROUNDOFF * std::fmax(std::fabs(tend), std::fabs(tcur));
            long long iout = (long long)std::ceil(sign * tcur / interval);
            long long iend = (long long)((sign * tend + dtroundoff) / interval);
            if(iend > (long long)trajsize - 1)
                iend = (long long)trajsize - 1;
            for(; iout <= iend; iout++) {
                double timeout   = sign * interval * iout;
                double offsetout = std::fmin(std::fmax(sign * (timeout - tcur), 0.0),
                    sign * hd) * sign;
                for(int k = 0; k < NDIM; k++)
                    traj[iout*6 + k] =
                        TOut(math::dop853_dense(state, NDIM, h, T(offsetout), k));
            }
        }
        tcur = tend;
        if(++numSteps >= maxNumSteps) {
            ok = tcur == totalTime;
            break;
        }
    }
    if(trajsize == 1 && ok) {
        // single-sample mode: only the final state (samplingInterval=INFINITY on CPU)
        for(int k = 0; k < NDIM; k++)
            traj[k] = TOut(state[k]);
    }
}

/** DPRKN8 counterpart of integrate_one_orbit: same trajectory-sampling logic,
    but the 8th-order Runge-Kutta-Nystrom core (2nd-order ODE) from math_ode.h --
    the same dprkn8_init / dprkn8_step / dprkn8_dense that the CPU class
    OdeStepperDPRKN8 wraps. `accuracy` is the ALREADY-RESCALED tolerance
    (10 * userAccuracy^0.9, matching the CPU stepper's constructor), so this body
    is method-agnostic about that rescaling. force1 is used only for the initial
    timestep estimate (as in dprkn8_init); force2 supplies the acceleration.
    TOut (defaults to T) is the storage precision of `traj` -- see integrate_one_orbit above. */
template<typename T, typename TOut, int ORDER>
AGAMA_DEVICE_INLINE void integrate_one_orbit_dprkn8(GpuDescForce<T, ORDER> force1,
    GpuDescForce2<T, ORDER> force2, const T ic6[6], double timeStart, double totalTime,
    T accuracy, unsigned long long maxNumSteps, std::size_t trajsize, TOut* traj)
{
    force1.timeBegin = timeStart;
    force2.timeBegin = timeStart;
    const int NDIM = 6;         // full 2nd-order system size (numVar = 3)
    for(std::size_t j = 0; j < trajsize * 6; j++)
        traj[j] = TOut(NAN);

    T state[18],   // persistent DPRKN8 storage: 3*NDIM (x, v, and the a/jerk/... blocks)
      scratch[39]; // 13*numVar step scratch (also covers the 4*NDIM=24 init scratch)
    T nextTimeStep = 0, qold = 0;
    math::dprkn8_init(force1, force2, NDIM, ic6, accuracy,
        state, nextTimeStep, qold, scratch);

    const double sign = totalTime >= 0 ? +1 : -1;
    const double interval = trajsize > 1 ? sign * totalTime / double(trajsize - 1) : 0;
    const double ROUNDOFF = 10 * DBL_EPSILON;

    double tcur = 0;
    bool ok = totalTime != 0;
    unsigned long long numSteps = 0;
    while(tcur != totalTime) {
        double timeRemaining = totalTime - tcur;
        force2.timeBegin = timeStart + tcur;
        T h = math::dprkn8_step(force2, NDIM, accuracy,
            state, scratch, nextTimeStep, qold, T(timeRemaining));
        if(!(double(h) * sign > 0)) {   // stepper signalled an error
            ok = false;
            break;
        }
        double hd   = double(h);
        double tend = h == T(timeRemaining) ? totalTime : tcur + hd;
        if(trajsize > 1) {
            double dtroundoff = ROUNDOFF * std::fmax(std::fabs(tend), std::fabs(tcur));
            long long iout = (long long)std::ceil(sign * tcur / interval);
            long long iend = (long long)((sign * tend + dtroundoff) / interval);
            if(iend > (long long)trajsize - 1)
                iend = (long long)trajsize - 1;
            for(; iout <= iend; iout++) {
                double timeout   = sign * interval * iout;
                double offsetout = std::fmin(std::fmax(sign * (timeout - tcur), 0.0),
                    sign * hd) * sign;
                for(int k = 0; k < NDIM; k++)
                    traj[iout*6 + k] =
                        TOut(math::dprkn8_dense(state, NDIM, h, T(offsetout), k));
            }
        }
        tcur = tend;
        if(++numSteps >= maxNumSteps) {
            ok = tcur == totalTime;
            break;
        }
    }
    if(trajsize == 1 && ok) {
        // single-sample mode: final state is state[0..5] = {x(3), v(3)}
        for(int k = 0; k < NDIM; k++)
            traj[k] = TOut(state[k]);
    }
}

/** Launch the batch under the given execution policy, ALWAYS running the
    DOP853 core. All pointers must be accessible by that policy's execution
    space (host pointers for Serial/OpenMP, device pointers for Cuda).

    `method` used to be a runtime argument to a single run_batch(), with the
    lambda branching at runtime between integrate_one_orbit (DOP853) and
    integrate_one_orbit_dprkn8 (DPRKN8). Because both callees are
    AGAMA_DEVICE_INLINE, BOTH integrators got inlined into the SAME kernel body
    even though any one launch only ever executes one branch -- register
    allocation then had to satisfy the union of both integrators' live state
    (DOP853's state[60]+xt[60] AND DPRKN8's state[18]+scratch[39] at once),
    which is most of the ~255-register/thread saturation and 2.6-5.5 KB/thread
    local-memory spill measured on this kernel. Splitting into two functions
    (this one and run_batch_dprkn8 below), selected ONCE on the host in
    run_batch()'s dispatch wrapper (further down), makes each a SEPARATE kernel
    that only ever inlines the integrator it actually calls -- ptxas then only
    has to allocate for one integrator's state, not both.

    TOut (defaults to T) is the storage precision of `traj`, threaded straight through
    to integrate_one_orbit -- see its doc comment for the narrow-on-store rationale. */
template<typename T, typename TOut, int ORDER, class Policy>
void run_batch_dop853(Policy pol, const potential::GpuPotDesc<T>& desc,
    std::size_t Norb, const T* ic, const double* times, const double* timeStart,
    std::size_t trajsize, T accuracy, unsigned long long maxNumSteps, TOut* traj)
{
    // The functor is built INSIDE the kernel so that each thread gets its own
    // (tiny) copy pointing at the single descriptor captured by value here. Built
    // outside, a host-side &desc would be meaningless on the device, and holding
    // the descriptor by value in the functor would put a multi-KiB copy into
    // the kernel argument space.
    agama::forall(pol, Norb, [=] AGAMA_DEVICE (std::size_t i) {
        T ic6[6];
        for(int k = 0; k < 6; k++)
            ic6[k] = ic[i*6 + k];
        const double t0 = timeStart ? timeStart[i] : 0.0;
        GpuDescForce<T, ORDER> force = { &desc, t0 };
        integrate_one_orbit<T, TOut, ORDER>(force, ic6, t0, times[i], accuracy,
            maxNumSteps, trajsize, traj + i * trajsize * 6);
    });
}

/** DPRKN8 counterpart of run_batch_dop853: same launch shape, but the kernel
    body only ever inlines integrate_one_orbit_dprkn8, so it never carries
    DOP853's state[60]+xt[60] register/spill cost. See run_batch_dop853 above
    for why this is a separate function rather than a runtime branch, and for
    the TOut storage-precision parameter. */
template<typename T, typename TOut, int ORDER, class Policy>
void run_batch_dprkn8(Policy pol, const potential::GpuPotDesc<T>& desc,
    std::size_t Norb, const T* ic, const double* times, const double* timeStart,
    std::size_t trajsize, T accuracy, unsigned long long maxNumSteps, TOut* traj)
{
    agama::forall(pol, Norb, [=] AGAMA_DEVICE (std::size_t i) {
        T ic6[6];
        for(int k = 0; k < 6; k++)
            ic6[k] = ic[i*6 + k];
        const double t0 = timeStart ? timeStart[i] : 0.0;
        GpuDescForce<T, ORDER>  force  = { &desc, t0 };  // 1st-order r.h.s. (DPRKN8 init)
        GpuDescForce2<T, ORDER> force2 = { &desc, t0 };  // 2nd-order r.h.s. (DPRKN8 stages)
        integrate_one_orbit_dprkn8<T, TOut, ORDER>(force, force2, ic6, t0, times[i],
            accuracy, maxNumSteps, trajsize, traj + i * trajsize * 6);
    });
}

/** Dispatch wrapper: picks DOP853 vs DPRKN8 ONCE on the host, BEFORE entering
    agama::forall / the kernel launch, so each launch instantiates only the
    kernel it needs (see run_batch_dop853 above for the register-pressure
    rationale). Signature and behaviour are unchanged from the old single
    run_batch() for TOut=T callers; TOut is threaded through unchanged. */
template<typename T, typename TOut, class Policy>
void run_batch(Policy pol, const potential::GpuPotDesc<T>& desc, int method,
    std::size_t Norb, const T* ic, const double* times, const double* timeStart,
    std::size_t trajsize, T accuracy, unsigned long long maxNumSteps, TOut* traj)
{
    // Second host-side switch, on the same principle as the method switch: the
    // Multipole scratch cap is a compile-time template parameter, so choosing it
    // here means an analytic-only orbit kernel never contains the 580-T local array
    // and its register/spill profile is untouched. The price is that this TU now
    // compiles four saturated kernels instead of two.
    // ORDER is fixed at 0 here: prepareOrbitBatch rejects any descriptor carrying a
    // Multipole, so these kernels are byte-for-byte the kernels they were before
    // Tier 2. See MULTIPOLE ON THE ORBIT PATH in prepareOrbitBatch for the measured
    // compile-time reason, and for what the ORDER template parameter is already
    // wired up to do once that is solved.
    if(method == orbit::ORBIT_GPU_DPRKN8)
        run_batch_dprkn8<T, TOut, 0>(pol, desc, Norb, ic, times, timeStart,
            trajsize, accuracy, maxNumSteps, traj);
    else
        run_batch_dop853<T, TOut, 0>(pol, desc, Norb, ic, times, timeStart,
            trajsize, accuracy, maxNumSteps, traj);
}

/** Shared prologue of the host-output and device-output entry points: validate the
    request, build the GPU force descriptor, and derive the working-precision accuracy.
    Extracted so the two public functions cannot drift apart -- in particular so the
    device-output path cannot miss the spline-buffer argument that makes time-VARYING
    modifier chains re-evaluable per step rather than silently frozen at t=0.
    Does NOT set desc.splineData: the caller points it at host or device memory
    depending on the policy it is about to run. */
template<typename T>
int prepareOrbitBatch(const potential::BasePotential& pot, std::size_t trajsize,
    double accuracy, int method,
    potential::GpuPotDesc<T>& desc, std::vector<T>& splineT, T& accT)
{
    if(trajsize < 1)
        return ORBIT_GPU_EUNSUPP;
    if(method != ORBIT_GPU_DOP853 && method != ORBIT_GPU_DPRKN8)
        return ORBIT_GPU_EUNSUPP;
    potential::GpuPotDesc<double> desc0;
    // Pass a spline buffer, which is what tells the builder to emit re-evaluable
    // GpuModStage entries for time-VARYING modifier chains instead of folding them
    // at one nominal time. That distinction is load-bearing here: the descriptor is
    // built ONCE and then evaluated at every RK stage of every orbit, at times this
    // function never sees, so a folded moving potential would silently integrate a
    // frozen snapshot. Constant chains are still folded on the host and cost the
    // kernel nothing.
    std::vector<double> splineData0;
    if(!potential::buildGpuPotDesc(pot, desc0, /*time*/ 0, &splineData0))
        return ORBIT_GPU_EUNSUPP;
    // ---- MULTIPOLE ON THE ORBIT PATH: FAIL CLOSED, and why -------------------
    // Batch potential/force/density evaluation DOES run a Multipole on the GPU (see
    // dispatch_desc in potential_gpu.cpp); orbit integration does not yet, and a
    // Multipole-bearing potential is rejected here so it integrates on the CPU rather
    // than against a kernel with no scratch for it.
    //
    // The reason is compile time, not correctness or throughput, and it was MEASURED
    // rather than assumed. The mechanism the batch kernels use -- an ORDER template
    // parameter selecting a per-thread scratch array, so kernels without a Multipole
    // pay nothing -- is already threaded through run_batch/GpuDescForce/
    // integrate_one_orbit here and needs only the two-way host switch restored. What
    // stops it is that instantiating it doubles the number of REG:255 kernels in this
    // TU from 4 to 8 AND inlines the whole ~1000-line four-branch Multipole evaluator
    // into DOP853's body, whose state[60]+xt[60] already saturates the register file:
    //
    //   baseline (this TU as it ships)          ~460 s
    //   + ORDER=32 orbit kernels                 one ptxas invocation still running
    //                                            after 27 min; abandoned
    //   + ORDER=8 orbit kernels                  cicc ~1,100 s then ptxas >890 s on a
    //                                            single kernel; abandoned at 2,192 s
    //
    // Lowering the order cap does not help because the cost is the size of the
    // inlined evaluator, not the array. The two things that plausibly would, in order
    // of preference:
    //   1. the "separate translation unit per integrator" item in pending_tasks.md --
    //      four saturated kernels currently compile serially inside one nvcc
    //      invocation, and this is exactly the case that would parallelize;
    //   2. marking the Multipole device evaluator __noinline__ for device
    //      compilation, so the orbit kernel calls it instead of absorbing it. The
    //      per-call cost should be negligible against ~1000 flops of physics (this
    //      code is ALU-bound, see findings.md), but it changes the batch kernels too
    //      and needs its own measurement.
    // Neither is a Tier 2 change, so this commit stops here rather than shipping a
    // build whose compile time is unusable.
    if(desc0.nmp > 0)
        return ORBIT_GPU_EUNSUPP;
    desc = potential::castGpuPotDesc<T>(desc0);
    // Spline coefficients in the kernel's working precision; `desc.splineData` is
    // pointed at whichever memory space the chosen policy will read from, by the caller.
    splineT = potential::castGpuSplineData<T>(splineData0);
    // Per-method base tolerance (in double): DPRKN8 applies the SAME empirical
    // rescaling as the CPU OdeStepperDPRKN8 constructor (10 * accuracy^0.9) so
    // the user-facing `accuracy` has an identical meaning for both methods and
    // the batch result matches the CPU stepper bit-for-bit in double.
    const double accBase = method == ORBIT_GPU_DPRKN8
        ? 10.0 * std::pow(accuracy, 0.9) : accuracy;
    // Working-precision floor on the accuracy parameter: an accRel near or below
    // one ULP of T (e.g. the fp64-oriented default 1e-8 under T=float) is
    // unattainable -- the error estimate sits at the rounding-noise floor, so the
    // adaptive controller grinds through micro-steps (measured 30-100x SLOWER than
    // fp64, with NaNs when the stepper gives up). 10 ULP is the measured knee on
    // this workload: for T=float the default 1e-8 clamps to ~1.2e-6 (123k orbits/s,
    // |dE/E| ~ 8e-6 on the crosscheck workload; explicitly passing accuracy~1e-5
    // reaches ~300k orbits/s). A no-op for T=double at any sane accuracy. The
    // clamp is applied AFTER the per-method rescaling so DPRKN8 in fp32 is guarded
    // too.
    accT = T(std::max(accBase, 10 * double(std::numeric_limits<T>::epsilon())));
    return ORBIT_GPU_OK;
}

#ifdef HAVE_CUDA
/* ---- Path A: per-call CUDA streams --------------------------------------
   Each integrateOrbitsGPU call runs on its OWN non-blocking stream with
   stream-ordered allocation and copies, so concurrent calls issued from
   different host threads (the Python binding releases the GIL around this
   function) are co-scheduled by the GPU's hardware scheduler instead of
   serializing on the legacy default stream. N small batches in flight then
   fill the card like one big batch. Stream choice affects scheduling only,
   never arithmetic: results are bit-identical to a solo run. */

/// true iff the device supports stream-ordered memory pools
/// (cudaMallocAsync); evaluated once, thread-safe per C++11 magic statics.
/// Without pool support we fall back to plain cudaMalloc, which is correct
/// but synchronizes the device and thus degrades (not breaks) overlap.
bool gpu_mem_pools_supported()
{
    static const bool supported = [] {
        int dev = 0, attr = 0;
        if(cudaGetDevice(&dev) != cudaSuccess)
            return false;
        if(cudaDeviceGetAttribute(&attr, cudaDevAttrMemoryPoolsSupported, dev) != cudaSuccess)
            return false;
        return attr != 0;
    }();
    return supported;
}

/// RAII per-call stream, non-blocking w.r.t. the legacy default stream so
/// that concurrent orbit calls never implicitly synchronize with stream 0
struct OrbitCallStream {
    cudaStream_t s = 0;
    OrbitCallStream() { AGAMA_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~OrbitCallStream() { if(s) cudaStreamDestroy(s); }   // dtor: no-throw
    OrbitCallStream(const OrbitCallStream&) = delete;
    OrbitCallStream& operator=(const OrbitCallStream&) = delete;
};

/// RAII device buffer ordered on the call's stream: allocation, copies and
/// deallocation are all enqueued on `s`, so nothing here synchronizes the
/// whole device (unlike device_array's cudaMalloc/cudaMemcpy)
template<typename T>
struct StreamBuffer {
    T* d = NULL;
    cudaStream_t s;
    StreamBuffer(std::size_t n, cudaStream_t stream) : s(stream) {
        if(gpu_mem_pools_supported())
            AGAMA_CUDA_CHECK(cudaMallocAsync(&d, n * sizeof(T), s));
        else
            AGAMA_CUDA_CHECK(cudaMalloc(&d, n * sizeof(T)));
    }
    ~StreamBuffer() {   // dtor: no-throw (errors here mean the context is dying anyway)
        if(!d) return;
        if(gpu_mem_pools_supported())
            cudaFreeAsync(d, s);
        else
            cudaFree(d);
    }
    StreamBuffer(const StreamBuffer&) = delete;
    StreamBuffer& operator=(const StreamBuffer&) = delete;
    T*       data()       { return d; }
    const T* data() const { return d; }
    void from_host(const T* h, std::size_t n) {
        // pageable-source async copy: the runtime returns after staging the
        // pageable buffer, so the host source may be freed after this call
        AGAMA_CUDA_CHECK(cudaMemcpyAsync(d, h, n * sizeof(T), cudaMemcpyHostToDevice, s));
    }
    void to_host(T* h, std::size_t n) const {
        AGAMA_CUDA_CHECK(cudaMemcpyAsync(h, d, n * sizeof(T), cudaMemcpyDeviceToHost, s));
    }
};

/// Order this call against a producer that wrote a caller-supplied device buffer on
/// a different stream. Values 1 and 2 are the __cuda_array_interface__ v3 sentinels
/// for the legacy default stream and the per-thread default stream respectively, and
/// 0 means the key was absent -- none of them names a stream we can or should
/// synchronize, so they are skipped. Mirrors the contract of the potential device
/// entry points so a caller reasons about one rule, not two.
void syncProducerStream(unsigned long long producer_stream)
{
    if(producer_stream > 2)
        AGAMA_CUDA_CHECK(cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(producer_stream)));
}

/** Shared CUDA body of the host-output and device-output entry points.

    Exactly one of `d_trajCaller` (a device buffer owned by the caller, left in place)
    and `h_traj` (a host buffer, filled by a D2H copy) must be non-NULL; that single
    difference is the entire distinction between the two public functions, so keeping
    one body means the stream discipline, the spline-data wiring and the NAN-fill
    guarantee cannot diverge between them.

    `ic`, `times` and `timeStart` are host pointers in both cases -- they are O(Norb)
    and uploaded here. `desc` is taken by value-modifying reference because
    desc.splineData must be repointed at the device spline buffer allocated below.

    TOut (defaults to T) is the storage precision of `d_trajCaller`/`h_traj`, threaded
    through to run_batch/integrate_one_orbit -- see integrateOrbitsGPU's doc comment
    in orbit_gpu.h for the narrow-on-store rationale. The device-side trajectory
    buffer (`d_traj`, whether caller-supplied or owned here) is always TOut-wide;
    only the working buffers (`d_ic`, the descriptor's spline data) stay T-wide. */
template<typename T, typename TOut>
int cudaOrbitBatch(potential::GpuPotDesc<T>& desc, const std::vector<T>& splineT,
    std::size_t Norb, const double* ic, const double* times, std::size_t trajsize,
    T accT, std::size_t maxNumSteps, int method, const double* timeStart,
    TOut* d_trajCaller, TOut* h_traj, unsigned long long producer_stream)
{
    if(Norb == 0)
        return ORBIT_GPU_OK;
    // ICs in working precision (identity copy for T=double)
    std::vector<T> icT(Norb * 6);
    for(std::size_t j = 0; j < Norb * 6; j++)
        icT[j] = static_cast<T>(ic[j]);
    // Path A: the whole call (allocation, copies, kernel, teardown) is
    // ordered on its own non-blocking stream, so concurrent calls from
    // different host threads overlap on the GPU. Per-call RAII buffers,
    // no shared scratch, no mutex: orbit batches amortize the allocation
    // cost over thousands of ODE steps.
    OrbitCallStream stream;
    StreamBuffer<T>      d_ic   (Norb * 6, stream.s);
    StreamBuffer<double> d_times(Norb,     stream.s);
    // Allocate a trajectory buffer only when the caller did not supply one. Length 0
    // is a legal cudaMallocAsync/cudaMalloc request and yields a NULL/unused pointer.
    StreamBuffer<TOut>   d_trajOwned(d_trajCaller ? 0 : Norb * trajsize * 6, stream.s);
    TOut* d_traj = d_trajCaller ? d_trajCaller : d_trajOwned.data();
    // modifier spline coefficients must be device-resident for the kernel to
    // re-evaluate the chain per step; empty for a time-independent potential
    StreamBuffer<T>      d_spline(splineT.size(),        stream.s);
    StreamBuffer<double> d_tstart(timeStart ? Norb : 0,  stream.s);
    d_ic.from_host(icT.data(), Norb * 6);
    d_times.from_host(times, Norb);
    if(!splineT.empty()) {
        d_spline.from_host(splineT.data(), splineT.size());
        desc.splineData = d_spline.data();
    }
    if(timeStart)
        d_tstart.from_host(timeStart, Norb);
    // If the caller's buffer came from another stream's producer, make that work
    // visible before we launch. No-op for the host-output path (producer_stream 0).
    syncProducerStream(producer_stream);
    agama::Cuda pol;
    pol.stream = stream.s;
    run_batch<T, TOut>(pol, desc, method, Norb, d_ic.data(), d_times.data(),
        timeStart ? d_tstart.data() : NULL,
        trajsize, accT, maxNumSteps, d_traj);
    if(h_traj)
        d_trajOwned.to_host(h_traj, Norb * trajsize * 6);
    // wait only for THIS call's work; other threads' streams keep running. For the
    // device-output path this is what lets the caller consume d_traj on any stream.
    AGAMA_CUDA_CHECK(cudaStreamSynchronize(stream.s));
    return ORBIT_GPU_OK;
}
#endif  // HAVE_CUDA

}  // anonymous namespace

template<typename T, typename TOut>
int integrateOrbitsGPU(const potential::BasePotential& pot,
                       std::size_t Norb,
                       const double* ic,
                       const double* times,
                       std::size_t trajsize,
                       double accuracy,
                       std::size_t maxNumSteps,
                       TOut* traj,
                       const char* device,
                       int method,
                       const double* timeStart)
{
    potential::GpuPotDesc<T> desc;
    std::vector<T> splineT;
    T accT;
    {
        const int rc = prepareOrbitBatch<T>(pot, trajsize, accuracy, method,
            desc, splineT, accT);
        if(rc != ORBIT_GPU_OK)
            return rc;
    }

    if(std::strcmp(device, "cpu") == 0 || std::strcmp(device, "openmp") == 0 ||
       std::strcmp(device, "serial") == 0) {
        // ICs in working precision (identity copy for T=double)
        std::vector<T> icT(Norb * 6);
        for(std::size_t j = 0; j < Norb * 6; j++)
            icT[j] = static_cast<T>(ic[j]);
        desc.splineData = splineT.empty() ? NULL : splineT.data();
        if(std::strcmp(device, "serial") == 0)
            run_batch<T, TOut>(agama::Serial{}, desc, method, Norb, icT.data(), times,
                timeStart, trajsize, accT, maxNumSteps, traj);
        else
            run_batch<T, TOut>(agama::OpenMP{}, desc, method, Norb, icT.data(), times,
                timeStart, trajsize, accT, maxNumSteps, traj);
        return ORBIT_GPU_OK;
    }
    if(std::strcmp(device, "cuda") == 0) {
#ifdef HAVE_CUDA
        return cudaOrbitBatch<T, TOut>(desc, splineT, Norb, ic, times, trajsize,
            accT, maxNumSteps, method, timeStart,
            /*d_trajCaller*/ NULL, /*h_traj*/ traj, /*producer_stream*/ 0);
#else
        return ORBIT_GPU_ENOTBUILT;
#endif
    }
    return ORBIT_GPU_EBADDEV;
}

template<typename T, typename TOut>
int integrateOrbitsGPUDevice(const potential::BasePotential& pot,
                             std::size_t Norb,
                             const double* ic,
                             const double* times,
                             std::size_t trajsize,
                             double accuracy,
                             std::size_t maxNumSteps,
                             TOut* d_traj,
                             unsigned long long output_stream,
                             int method,
                             const double* timeStart)
{
#ifdef HAVE_CUDA
    // Validation and descriptor construction are shared with the host-output entry
    // point via the helper below; only the destination differs.
    potential::GpuPotDesc<T> desc;
    std::vector<T> splineT;
    T accT;
    const int rc = prepareOrbitBatch<T>(pot, trajsize, accuracy, method,
        desc, splineT, accT);
    if(rc != ORBIT_GPU_OK)
        return rc;
    if(Norb == 0)
        return ORBIT_GPU_OK;
    if(d_traj == NULL)
        return ORBIT_GPU_EUNSUPP;
    return cudaOrbitBatch<T, TOut>(desc, splineT, Norb, ic, times, trajsize,
        accT, maxNumSteps, method, timeStart,
        /*d_trajCaller*/ d_traj, /*h_traj*/ NULL, output_stream);
#else
    (void)pot; (void)Norb; (void)ic; (void)times; (void)trajsize; (void)accuracy;
    (void)maxNumSteps; (void)d_traj; (void)output_stream; (void)method; (void)timeStart;
    return ORBIT_GPU_ENOTBUILT;
#endif
}

template<typename T>
int scaleTrajectoryGPUDevice(std::size_t n, T* d_traj,
                             double lengthUnit, double velocityUnit,
                             unsigned long long output_stream)
{
#ifdef HAVE_CUDA
    if(n == 0)
        return ORBIT_GPU_OK;
    if(d_traj == NULL)
        return ORBIT_GPU_EUNSUPP;
    OrbitCallStream stream;
    // order against a producer that wrote d_traj on a different stream
    syncProducerStream(output_stream);
    agama::Cuda pol;
    pol.stream = stream.s;
    // Divide in double, and by division rather than by a precomputed reciprocal, so
    // the result is bit-identical to the host path's promote-divide-round -- see the
    // rationale in the header. `lu`/`vu` are captured by value into the lambda.
    const double lu = lengthUnit, vu = velocityUnit;
    T* p = d_traj;
    agama::forall(pol, n, [=] AGAMA_DEVICE (std::size_t i) {
        T* w = p + i * 6;
        w[0] = T(double(w[0]) / lu);
        w[1] = T(double(w[1]) / lu);
        w[2] = T(double(w[2]) / lu);
        w[3] = T(double(w[3]) / vu);
        w[4] = T(double(w[4]) / vu);
        w[5] = T(double(w[5]) / vu);
    });
    AGAMA_CUDA_CHECK(cudaStreamSynchronize(stream.s));
    return ORBIT_GPU_OK;
#else
    (void)n; (void)d_traj; (void)lengthUnit; (void)velocityUnit; (void)output_stream;
    return ORBIT_GPU_ENOTBUILT;
#endif
}

// Explicit instantiations for the two precisions the Python boundary exposes (TOut
// defaulted to T, i.e. the storage-narrowing path is opt-in only), PLUS the
// double-integration/float-storage narrow-on-store combination (interface_python.cpp's
// `narrowStoreDefault`): fp64 integration writing directly into a float32 destination,
// used only under a trivial unit system -- see orbit_gpu.h's doc comment.
template int integrateOrbitsGPU<float >(const potential::BasePotential&, std::size_t,
    const double*, const double*, std::size_t, double, std::size_t, float*,  const char*, int, const double*);
template int integrateOrbitsGPU<double>(const potential::BasePotential&, std::size_t,
    const double*, const double*, std::size_t, double, std::size_t, double*, const char*, int, const double*);
template int integrateOrbitsGPU<double, float>(const potential::BasePotential&, std::size_t,
    const double*, const double*, std::size_t, double, std::size_t, float*,  const char*, int, const double*);
template int integrateOrbitsGPUDevice<float >(const potential::BasePotential&, std::size_t,
    const double*, const double*, std::size_t, double, std::size_t, float*,  unsigned long long, int, const double*);
template int integrateOrbitsGPUDevice<double>(const potential::BasePotential&, std::size_t,
    const double*, const double*, std::size_t, double, std::size_t, double*, unsigned long long, int, const double*);
template int scaleTrajectoryGPUDevice<float >(std::size_t, float*,  double, double, unsigned long long);
template int scaleTrajectoryGPUDevice<double>(std::size_t, double*, double, double, unsigned long long);

}  // namespace orbit
