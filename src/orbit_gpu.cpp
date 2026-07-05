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
    accFac = min(1, |Epot+Ekin| / max(|Epot|, Ekin)). The potential is static,
    so the time argument is unused. */
template<typename T>
struct GpuDescForce {
    potential::GpuPotDesc<T> desc;

    AGAMA_DEVICE_INLINE void operator()(T /*t*/, const T w[], T dwdt[], T* accFac) const
    {
        T Epot, acc[3];
        potential::gpu_desc_phi_acc(desc, w[0], w[1], w[2],
            accFac ? &Epot : (T*)NULL, acc);
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

/** Integrate a single orbit and store its sampled trajectory.
    Runs identically as the body of a CUDA thread or a (Serial/OpenMP) CPU
    loop iteration. Integration state is in precision T; time bookkeeping is
    always double (an fp32 running time would lose sampling accuracy over
    thousands of steps). The trajectory sample placement mirrors
    orbit::RuntimeTrajectory::processTimestep with t0 = 0: sample j lies at
    time sign * interval * j, interval = |totalTime| / (trajsize-1); samples
    that the integration never reaches (stepper error / step limit) stay NAN. */
template<typename T>
AGAMA_DEVICE_INLINE void integrate_one_orbit(const GpuDescForce<T>& force,
    const T ic6[6], double totalTime, T accuracy,
    unsigned long long maxNumSteps, std::size_t trajsize, T* traj)
{
    const int NDIM = 6;
    // pre-fill with NAN: anything not overwritten below signals "not reached"
    for(std::size_t j = 0; j < trajsize * 6; j++)
        traj[j] = T(NAN);

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
                        math::dop853_dense(state, NDIM, h, T(offsetout), k);
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
            traj[k] = state[k];
    }
}

/** Launch the batch under the given execution policy. All pointers must be
    accessible by that policy's execution space (host pointers for
    Serial/OpenMP, device pointers for Cuda). */
template<typename T, class Policy>
void run_batch(Policy pol, const potential::GpuPotDesc<T>& desc,
    std::size_t Norb, const T* ic, const double* times, std::size_t trajsize,
    T accuracy, unsigned long long maxNumSteps, T* traj)
{
    GpuDescForce<T> force = { desc };
    agama::forall(pol, Norb, [=] AGAMA_DEVICE (std::size_t i) {
        T ic6[6];
        for(int k = 0; k < 6; k++)
            ic6[k] = ic[i*6 + k];
        integrate_one_orbit(force, ic6, times[i], accuracy, maxNumSteps,
            trajsize, traj + i * trajsize * 6);
    });
}

}  // anonymous namespace

template<typename T>
int integrateOrbitsGPU(const potential::BasePotential& pot,
                       std::size_t Norb,
                       const double* ic,
                       const double* times,
                       std::size_t trajsize,
                       double accuracy,
                       std::size_t maxNumSteps,
                       T* traj,
                       const char* device)
{
    if(trajsize < 1)
        return ORBIT_GPU_EUNSUPP;
    potential::GpuPotDesc<double> desc0;
    if(!potential::buildGpuPotDesc(pot, desc0))
        return ORBIT_GPU_EUNSUPP;
    const potential::GpuPotDesc<T> desc = potential::castGpuPotDesc<T>(desc0);
    // Working-precision floor on the accuracy parameter: an accRel near or below
    // one ULP of T (e.g. the fp64-oriented default 1e-8 under T=float) is
    // unattainable -- the error estimate sits at the rounding-noise floor, so the
    // adaptive controller grinds through micro-steps (measured 30-100x SLOWER than
    // fp64, with NaNs when the stepper gives up). 10 ULP is the measured knee on
    // this workload: for T=float the default 1e-8 clamps to ~1.2e-6 (123k orbits/s,
    // |dE/E| ~ 8e-6 on the crosscheck workload; explicitly passing accuracy~1e-5
    // reaches ~300k orbits/s). A no-op for T=double at any sane accuracy.
    const T accT = T(std::max(accuracy, 10 * double(std::numeric_limits<T>::epsilon())));

    if(std::strcmp(device, "cpu") == 0 || std::strcmp(device, "openmp") == 0 ||
       std::strcmp(device, "serial") == 0) {
        // ICs in working precision (identity copy for T=double)
        std::vector<T> icT(Norb * 6);
        for(std::size_t j = 0; j < Norb * 6; j++)
            icT[j] = static_cast<T>(ic[j]);
        if(std::strcmp(device, "serial") == 0)
            run_batch<T>(agama::Serial{}, desc, Norb, icT.data(), times,
                trajsize, accT, maxNumSteps, traj);
        else
            run_batch<T>(agama::OpenMP{}, desc, Norb, icT.data(), times,
                trajsize, accT, maxNumSteps, traj);
        return ORBIT_GPU_OK;
    }
    if(std::strcmp(device, "cuda") == 0) {
#ifdef HAVE_CUDA
        if(Norb == 0)
            return ORBIT_GPU_OK;
        std::vector<T> icT(Norb * 6);
        for(std::size_t j = 0; j < Norb * 6; j++)
            icT[j] = static_cast<T>(ic[j]);
        // Per-call RAII buffers (no shared scratch, no mutex): orbit batches
        // amortize the ~ms allocation cost over thousands of ODE steps, unlike
        // the cheap one-shot potential evaluations in potential_gpu.cpp.
        agama::device_array<T>      d_ic   (Norb * 6);
        agama::device_array<double> d_times(Norb);
        agama::device_array<T>      d_traj (Norb * trajsize * 6);
        d_ic.from_host(icT.data(), Norb * 6);
        d_times.from_host(times, Norb);
        run_batch<T>(agama::Cuda{}, desc, Norb, d_ic.data(), d_times.data(),
            trajsize, accT, maxNumSteps, d_traj.data());
        // blocking D2H on the default stream doubles as the kernel sync point
        d_traj.to_host(traj, Norb * trajsize * 6);
        return ORBIT_GPU_OK;
#else
        return ORBIT_GPU_ENOTBUILT;
#endif
    }
    return ORBIT_GPU_EBADDEV;
}

// Explicit instantiations for the two precisions the Python boundary exposes.
template int integrateOrbitsGPU<float >(const potential::BasePotential&, std::size_t,
    const double*, const double*, std::size_t, double, std::size_t, float*,  const char*);
template int integrateOrbitsGPU<double>(const potential::BasePotential&, std::size_t,
    const double*, const double*, std::size_t, double, std::size_t, double*, const char*);

}  // namespace orbit
