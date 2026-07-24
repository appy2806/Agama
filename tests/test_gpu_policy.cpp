// test_gpu_policy.cpp — smoke test for src/gpu_policy.h
//
// Exercises every execution policy (Serial, OpenMP, Cuda) through both
// primitives (forall, parallel_reduce_sum) and compares results bit-for-bit.
//
// CPU-only build (HAVE_CUDA=0): runs Serial + OpenMP paths and reports PASS
// without needing a CUDA toolchain.
//
// HAVE_CUDA=1 build: this TU is routed through nvcc (see Makefile rule for
// $(EXEDIR)/test_gpu_policy.exe) so that the kernel-launch syntax inside
// forall<Cuda>/parallel_reduce_sum<Cuda> is parseable.

#include "gpu_policy.h"
#include "math_sphharm.h"        // for math::trigMultiAngle (Tier 0 device-inline leaf)
#include "coord.h"               // toPos<Car,Cyl>, toPos<Car,Sph> (Tier 0 Phase 2 device-inline)
#include "potential_analytic.h"  // potential::NFW + nfw_phi leaf (Tier 1 worked example)
#include "potential_dehnen.h"    // potential::Dehnen (spherical GPU path) + dehnen_eval leaf
#include "potential_disk.h"      // potential::DiskAnsatz + disk_ansatz_eval/_rho leaves
#include "potential_composite.h"    // Composite for the Tier 3 orbit workload; UniformAcceleration
#include "potential_descriptor.h"   // GpuPotDesc + gpu_desc_phi_acc (Tier 3 force descriptor)
#include "orbit.h"                  // orbit::integrateTraj (CPU reference integrator)
#include "orbit_gpu.h"              // orbit::integrateOrbitsGPU (Tier 3 batch path)
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace {
#ifdef HAVE_CUDA
// Templated per-potential parity check. Runs evalmanyCarT<double, Serial> and
// evalmanyCarT<double, Cuda> on N Cartesian points (packed xyz_h length 3*N),
// compares element-by-element to a relative tol vs max|phi|, and cross-checks
// the first 8 points against the existing `pot.value(pos)` virtual to confirm
// leaf-vs-virtual single-source. Returns true on success.
template<class Pot>
bool check_pot_parity(Pot& pot, const char* name,
    const std::vector<double>& xyz_h, double rel_tol = 1e-13)
{
    using namespace agama;
    const std::size_t N = xyz_h.size() / 3;
    std::vector<double> phi_s(N);
    pot.template evalmanyCarT<double>(Serial{}, N, xyz_h.data(), phi_s.data());
    double max_phi = 1e-300;
    for(std::size_t i = 0; i < N; ++i)
        max_phi = std::max(max_phi, std::fabs(phi_s[i]));
    double max_virt_err = 0.0;
    for(std::size_t i = 0; i < 8 && i < N; ++i) {
        const coord::PosCar p(xyz_h[i*3+0], xyz_h[i*3+1], xyz_h[i*3+2]);
        max_virt_err = std::max(max_virt_err, std::fabs(pot.value(p) - phi_s[i]));
    }
    device_array<double> d_xyz(N * 3);
    d_xyz.from_host(xyz_h.data(), N * 3);
    device_array<double> d_phi(N);
    pot.template evalmanyCarT<double>(Cuda{}, N, d_xyz.data(), d_phi.data());
    std::vector<double> phi_c(N);
    d_phi.to_host(phi_c.data(), N);
    double max_err = 0.0;
    for(std::size_t i = 0; i < N; ++i)
        max_err = std::max(max_err, std::fabs(phi_s[i] - phi_c[i]));
    const double abs_tol = rel_tol * max_phi;
    bool ok = (max_err <= abs_tol) && (max_virt_err <= abs_tol);
    std::printf("[CUDA]  %-14s  max|phi|=%.3e   Serial-vs-Cuda |err|=%.3e   leaf-vs-virtual |err|=%.3e   tol=%.1e (rel %.0e) -> %s\n",
        name, max_phi, max_err, max_virt_err, abs_tol, rel_tol, ok ? "OK" : "FAIL");
    return ok;
}
#endif

// Min over N trials of `fn()` in milliseconds. Excludes the first warm-up call,
// which carries one-time costs (CUDA context init, first-touch allocation, JIT).
template<class F>
double time_ms_min(int trials, F fn) {
    using clk = std::chrono::high_resolution_clock;
    fn();   // warm-up, discarded
    double best = 1e300;
    for(int t = 0; t < trials; ++t) {
        auto t0 = clk::now();
        fn();
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if(ms < best) best = ms;
    }
    return best;
}

// Templated forall + reduce timing harness — runs the same workload for any T.
// Important on RTX 30xx/L40 (Ada/Ampere consumer) because fp64:fp32 throughput is 1:64;
// the fp32 path is the realistic one for GPU acceleration on those cards.
// HAVE_CUDA-gated: references the Cuda policy, which does not exist in CPU-only builds
// (and g++ rejects the unknown identifier even in this uninstantiated template).
#ifdef HAVE_CUDA
template<typename T>
void run_perf_T(int trials, std::size_t N, const char* tname) {
    using namespace agama;
    std::printf("\n=== T = %-6s   N = %zu (%.1f MiB device) ===\n",
        tname, N, double(N * sizeof(T)) / (1024.0 * 1024.0));

    // --- forall: out[i] = 1 + 2*sin(1e-6 * i) ---
    std::vector<T> buf_s(N), buf_o(N);
    double t_fs = time_ms_min(trials, [&]{
        forall(Serial{}, N, [&](std::size_t i){
            buf_s[i] = T(1) + T(2) * std::sin(T(1e-6) * static_cast<T>(i));
        });
    });
    double t_fo = time_ms_min(trials, [&]{
        forall(OpenMP{}, N, [&](std::size_t i){
            buf_o[i] = T(1) + T(2) * std::sin(T(1e-6) * static_cast<T>(i));
        });
    });
    device_array<T> d_buf(N);
    T* dp = d_buf.data();
    double t_fc = time_ms_min(trials, [&]{
        forall(Cuda{}, N, [=] AGAMA_DEVICE (std::size_t i){
            dp[i] = T(1) + T(2) * std::sin(T(1e-6) * static_cast<T>(i));
        });
        // pull one element back so timing reflects kernel completion + transfer
        T tmp; cudaMemcpy(&tmp, dp, sizeof(T), cudaMemcpyDeviceToHost);
    });

    // --- reduce: sum sin(1e-6 i) + cos(2e-6 i) ---
    auto reduce_op = [] AGAMA_DEVICE (std::size_t i) -> T {
        return std::sin(static_cast<T>(i) * T(1e-6)) + std::cos(static_cast<T>(i) * T(2e-6));
    };
    T rs = 0, ro = 0, rc = 0;
    double t_rs = time_ms_min(trials, [&]{ rs = parallel_reduce_sum(Serial{}, N, T(0), reduce_op); });
    double t_ro = time_ms_min(trials, [&]{ ro = parallel_reduce_sum(OpenMP{}, N, T(0), reduce_op); });
    double t_rc = time_ms_min(trials, [&]{ rc = parallel_reduce_sum(Cuda{},   N, T(0), reduce_op); });

    auto row = [](const char* name, double s, double o, double c) {
        std::printf("  %-20s  Serial %9.3f ms   OpenMP %9.3f ms (%5.1fx)   Cuda %9.3f ms (%6.1fx)\n",
            name, s, o, s / o, c, s / c);
    };
    row("forall (sin write)",   t_fs, t_fo, t_fc);
    row("reduce (sin+cos sum)", t_rs, t_ro, t_rc);
    std::printf("  reduce results: Serial=%.6f  OpenMP=%.6f  Cuda=%.6f\n",
        double(rs), double(ro), double(rc));
}
#endif  // HAVE_CUDA (run_perf_T)
}  // namespace

int main() {
    using namespace agama;
    const std::size_t N = 1024;

    // ----- forall<Serial> -----
    std::vector<double> out_s(N);
    forall(Serial{}, N, [&](std::size_t i) { out_s[i] = 2.0 * static_cast<double>(i); });

    // ----- forall<OpenMP> -----
    std::vector<double> out_o(N);
    forall(OpenMP{}, N, [&](std::size_t i) { out_o[i] = 2.0 * static_cast<double>(i); });

    // ----- parallel_reduce_sum<Serial / OpenMP> -----
    double rsum_s = parallel_reduce_sum(Serial{}, N, 0.0,
        [](std::size_t i) { return static_cast<double>(i); });
    double rsum_o = parallel_reduce_sum(OpenMP{}, N, 0.0,
        [](std::size_t i) { return static_cast<double>(i); });

    bool ok_cpu = (out_s == out_o) && (rsum_s == rsum_o);
    std::printf("[CPU]   forall Serial==OpenMP: %s   reduce Serial==OpenMP: %s\n",
        (out_s == out_o) ? "yes" : "NO",
        (rsum_s == rsum_o) ? "yes" : "NO");
    std::printf("[CPU]   reduce result: %.1f (expected %.1f)\n",
        rsum_s, 0.5 * static_cast<double>(N) * static_cast<double>(N - 1));

    // ----- coord::toPos<Car,Cyl> / toPos<Car,Sph> on Serial (reference) -----
    // Tier 0 Phase 2 leaf: cartesian -> cylindrical / spherical conversion.
    // Header-inlined so the same body runs CPU + GPU.
    const std::size_t NPT = 1024;
    std::vector<double> car_xyz(NPT * 3);
    for(std::size_t i = 0; i < NPT; ++i) {
        // a spread of points well off the singular axes / origin
        car_xyz[i * 3 + 0] = 0.5 + 1.7 * static_cast<double>(i % 13);
        car_xyz[i * 3 + 1] = -0.3 + 0.9 * static_cast<double>(i % 17);
        car_xyz[i * 3 + 2] =  0.2 + 0.4 * static_cast<double>(i %  7);
    }
    std::vector<double> cyl_s(NPT * 3), sph_s(NPT * 3);
    for(std::size_t i = 0; i < NPT; ++i) {
        const coord::PosCar p(car_xyz[i*3+0], car_xyz[i*3+1], car_xyz[i*3+2]);
        const coord::PosCyl c = coord::toPos<coord::Car, coord::Cyl>(p, coord::Cyl());
        const coord::PosSph s = coord::toPos<coord::Car, coord::Sph>(p, coord::Sph());
        cyl_s[i*3+0] = c.R;   cyl_s[i*3+1] = c.z;     cyl_s[i*3+2] = c.phi;
        sph_s[i*3+0] = s.r;   sph_s[i*3+1] = s.theta; sph_s[i*3+2] = s.phi;
    }

    // ----- math::trigMultiAngle on Serial (reference) -----
    // Tier 0 leaf math: cos(k*phi), sin(k*phi) for k=1..mmax via the Num.Rec. recurrence.
    // Inlined in math_sphharm.h with AGAMA_DEVICE_INLINE so the same body runs CPU+GPU.
    const std::size_t NP = 64;            // number of phi samples
    const unsigned    MM = 8;             // mmax (8 multiples of phi)
    std::vector<double> trig_s(NP * 2 * MM);  // [cos1..cosM, sin1..sinM] per sample
    for(std::size_t p = 0; p < NP; ++p) {
        const double phi = (p + 0.5) * (2.0 * 3.14159265358979323846 / NP);
        math::trigMultiAngle(phi, MM, /*needSine=*/true, &trig_s[p * 2 * MM]);
    }

    // =====================================================================
    // Tier 3: GPU force descriptor + batch orbit integration.
    // (a) gpu_desc_phi_acc vs the virtual Composite::eval at scattered points
    //     (locks the tagged-union glue against the class path);
    // (b) integrateOrbitsGPU 'serial' vs the CPU class integrator
    //     (orbit::integrateTraj) -- same DOP853 core, different force-eval
    //     glue, so trajectories agree to a slowly-growing tolerance, and
    //     energy is conserved to the integrator accuracy;
    // (c) [HAVE_CUDA] 'cuda' vs 'serial' parity + fp32 sanity.
    // The composite (Plummer bulge + MiyamotoNagai disk + NFW halo) mirrors
    // the stream-modelling workload of local_notes/crosscheck_orbits.py.
    // =====================================================================
    {
        std::vector<potential::PtrPotential> comps;
        comps.push_back(potential::PtrPotential(new potential::Plummer(0.1, 0.3)));
        comps.push_back(potential::PtrPotential(new potential::MiyamotoNagai(0.5, 1.0, 0.3)));
        comps.push_back(potential::PtrPotential(new potential::NFW(10.0, 5.0)));
        potential::Composite pot(comps);

        // ----- (a) descriptor-vs-virtual force parity -----
        potential::GpuPotDesc<double> desc;
        bool ok_desc = potential::buildGpuPotDesc(pot, desc);
        double max_rel_force = 0;
        if(ok_desc) {
            for(int i = 0; i < 256; i++) {
                const double x = 0.05 + 0.03  * i,
                             y = -0.4 + 0.021 * (i % 37),
                             z = -0.2 + 0.013 * (i % 29);
                double phi_d, acc_d[3];
                potential::gpu_desc_phi_acc(desc, x, y, z, &phi_d, acc_d);
                double phi_v;
                coord::GradCar grad;
                pot.eval(coord::PosCar(x, y, z), &phi_v, &grad, NULL);
                const double acc_v[3] = { -grad.dx, -grad.dy, -grad.dz };
                double scale = std::fabs(phi_v);
                max_rel_force = std::max(max_rel_force, std::fabs(phi_d - phi_v) / scale);
                for(int k = 0; k < 3; k++) {
                    scale = std::max(1e-300, std::fabs(acc_v[k]));
                    max_rel_force = std::max(max_rel_force,
                        std::fabs(acc_d[k] - acc_v[k]) / scale);
                }
            }
        }
        const double DESC_TOL = 1e-13;
        bool ok_desc_parity = ok_desc && max_rel_force <= DESC_TOL;
        std::printf("[T3]    force descriptor vs virtual eval (composite, 256 pts): "
            "max rel err = %.3e, tol = %.1e -> %s\n",
            max_rel_force, DESC_TOL, ok_desc_parity ? "OK" : "FAIL");

        // ----- (b) batch orbits, 'serial' backend vs the CPU class integrator -----
        const std::size_t NORB = 64, TRAJ = 16;
        const double TTOT = 40.0, ACC = 1e-8;
        std::vector<double> ic(NORB * 6), times(NORB, TTOT);
        for(std::size_t i = 0; i < NORB; i++) {
            const double r   = 0.5 + 2.5 * double(i) / NORB;
            const double ang = 0.7 * i;
            // roughly circular-speed ICs with a vertical kick, scattered in phase
            double phi_v;
            coord::GradCar grad;
            pot.eval(coord::PosCar(r * std::cos(ang), r * std::sin(ang), 0.05), &phi_v, &grad, NULL);
            const double vc = std::sqrt(r * std::sqrt(grad.dx*grad.dx + grad.dy*grad.dy));
            ic[i*6+0] = r * std::cos(ang);
            ic[i*6+1] = r * std::sin(ang);
            ic[i*6+2] = 0.05;
            ic[i*6+3] = -vc * std::sin(ang) * 0.9;
            ic[i*6+4] =  vc * std::cos(ang) * 0.9;
            ic[i*6+5] =  0.1 * vc;
        }
        std::vector<double> traj_s(NORB * TRAJ * 6);
        int rc_s = orbit::integrateOrbitsGPU<double>(pot, NORB, ic.data(), times.data(),
            TRAJ, ACC, /*maxNumSteps*/ 100000000, traj_s.data(), "serial");
        // reference: the ordinary CPU orbit integrator, same sampling
        double max_rel_orbit = 0;
        for(std::size_t i = 0; i < NORB; i++) {
            coord::PosVelCar ic_i(&ic[i*6]);
            orbit::Trajectory ref = orbit::integrateTraj(ic_i, TTOT,
                /*samplingInterval*/ TTOT / (TRAJ - 1), pot);
            if(ref.size() != TRAJ) { max_rel_orbit = INFINITY; break; }
            for(std::size_t j = 0; j < TRAJ; j++) {
                double refv[6], scale = 0;
                ref[j].first.unpack_to(refv);
                for(int k = 0; k < 6; k++)
                    scale = std::max(scale, std::fabs(refv[k]));
                for(int k = 0; k < 6; k++)
                    max_rel_orbit = std::max(max_rel_orbit,
                        std::fabs(traj_s[(i*TRAJ+j)*6+k] - refv[k]) / scale);
            }
        }
        // Two independent 1e-8-accuracy solutions of the same ODE with
        // ULP-different force glue: the difference grows secularly with time;
        // 1e-5 over ~10 orbital periods is the expected scale, NOT a bug.
        const double ORB_TOL = 1e-5;
        bool ok_orb_serial = rc_s == 0 && max_rel_orbit <= ORB_TOL;
        std::printf("[T3]    batch orbits 'serial' vs OrbitIntegrator (N=%zu, T=%g, %zu samples): "
            "max rel err = %.3e, tol = %.1e -> %s\n",
            NORB, TTOT, TRAJ, max_rel_orbit, ORB_TOL, ok_orb_serial ? "OK" : "FAIL");

        // energy conservation of the batch output (an absolute quality gate
        // that does not depend on comparing two step sequences)
        auto energy = [&](const double w[6]) {
            double phi_v;
            pot.eval(coord::PosCar(w[0], w[1], w[2]), &phi_v, NULL, NULL);
            return phi_v + 0.5 * (w[3]*w[3] + w[4]*w[4] + w[5]*w[5]);
        };
        double max_dE = 0;
        for(std::size_t i = 0; i < NORB; i++) {
            const double E0 = energy(&traj_s[(i*TRAJ+0)*6]);
            const double E1 = energy(&traj_s[(i*TRAJ+TRAJ-1)*6]);
            max_dE = std::max(max_dE, std::fabs((E1 - E0) / E0));
        }
        const double DE_TOL = 1e-7;
        bool ok_energy = max_dE <= DE_TOL;
        std::printf("[T3]    batch orbits 'serial' energy conservation: max |dE/E| = %.3e, "
            "tol = %.1e -> %s\n", max_dE, DE_TOL, ok_energy ? "OK" : "FAIL");

        bool ok_t3 = ok_desc_parity && ok_orb_serial && ok_energy;

#ifdef HAVE_CUDA
        // ----- (c) 'cuda' vs 'serial' parity (fp64) + fp32 sanity -----
        std::vector<double> traj_c(NORB * TRAJ * 6);
        int rc_c = orbit::integrateOrbitsGPU<double>(pot, NORB, ic.data(), times.data(),
            TRAJ, ACC, 100000000, traj_c.data(), "cuda");
        double max_rel_cuda = 0;
        for(std::size_t i = 0; i < NORB * TRAJ; i++) {
            double scale = 0;
            for(int k = 0; k < 6; k++)
                scale = std::max(scale, std::fabs(traj_s[i*6+k]));
            for(int k = 0; k < 6; k++)
                max_rel_cuda = std::max(max_rel_cuda,
                    std::fabs(traj_c[i*6+k] - traj_s[i*6+k]) / scale);
        }
        bool ok_orb_cuda = rc_c == 0 && max_rel_cuda <= ORB_TOL;
        std::printf("[CUDA]  batch orbits 'cuda' vs 'serial' (fp64): max rel err = %.3e, "
            "tol = %.1e -> %s\n", max_rel_cuda, ORB_TOL, ok_orb_cuda ? "OK" : "FAIL");

        // fp32: integration quality is limited by single precision; gate on
        // energy conservation at a loose tolerance and absence of NaNs
        std::vector<float> traj_f(NORB * TRAJ * 6);
        int rc_f = orbit::integrateOrbitsGPU<float>(pot, NORB, ic.data(), times.data(),
            TRAJ, /*accuracy*/ 1e-5, 100000000, traj_f.data(), "cuda");
        double max_dE_f = 0;
        bool nan_f = false;
        for(std::size_t i = 0; i < NORB; i++) {
            double w0[6], w1[6];
            for(int k = 0; k < 6; k++) {
                w0[k] = traj_f[(i*TRAJ+0)*6+k];
                w1[k] = traj_f[(i*TRAJ+TRAJ-1)*6+k];
                nan_f |= w0[k] != w0[k] || w1[k] != w1[k];
            }
            if(nan_f) break;
            max_dE_f = std::max(max_dE_f, std::fabs((energy(w1) - energy(w0)) / energy(w0)));
        }
        const double DE_TOL_F = 1e-2;
        bool ok_orb_f = rc_f == 0 && !nan_f && max_dE_f <= DE_TOL_F;
        std::printf("[CUDA]  batch orbits 'cuda' fp32 sanity: max |dE/E| = %.3e, tol = %.1e, "
            "NaN: %s -> %s\n", max_dE_f, DE_TOL_F, nan_f ? "yes" : "no", ok_orb_f ? "OK" : "FAIL");

        ok_t3 = ok_t3 && ok_orb_cuda && ok_orb_f;
#endif
        if(!ok_t3) {
            std::fprintf(stderr, "FAIL (Tier 3 orbit integration)\n");
            return 1;
        }
    }

#ifdef HAVE_CUDA
    // ----- forall<Cuda> -----
    device_array<double> d_out(N);
    double* dptr = d_out.data();
    forall(Cuda{}, N, [=] AGAMA_DEVICE (std::size_t i) {
        dptr[i] = 2.0 * static_cast<double>(i);
    });
    std::vector<double> out_c(N);
    d_out.to_host(out_c.data(), N);

    // ----- parallel_reduce_sum<Cuda> -----
    double rsum_c = parallel_reduce_sum(Cuda{}, N, 0.0,
        [] AGAMA_DEVICE (std::size_t i) { return static_cast<double>(i); });

    // ----- coord::toPos<Car,Cyl> / toPos<Car,Sph> on Cuda (Tier 0 Phase 2) -----
    // One thread per point: read xyz, call the inline header-only transforms,
    // write (R,z,phi) and (r,theta,phi). Compare bit-exact with the Serial
    // reference; the formulas use math::sincos / math::atan2 (also header-inline)
    // so host and device evaluate exactly the same arithmetic.
    device_array<double> d_car(NPT * 3);
    d_car.from_host(car_xyz.data(), NPT * 3);
    const double* d_car_ptr = d_car.data();
    device_array<double> d_cyl(NPT * 3), d_sph(NPT * 3);
    double* d_cyl_ptr = d_cyl.data();
    double* d_sph_ptr = d_sph.data();
    forall(Cuda{}, NPT, [=] AGAMA_DEVICE (std::size_t i) {
        const coord::PosCar p(d_car_ptr[i*3+0], d_car_ptr[i*3+1], d_car_ptr[i*3+2]);
        const coord::PosCyl c = coord::toPos<coord::Car, coord::Cyl>(p, coord::Cyl());
        const coord::PosSph s = coord::toPos<coord::Car, coord::Sph>(p, coord::Sph());
        d_cyl_ptr[i*3+0] = c.R;   d_cyl_ptr[i*3+1] = c.z;     d_cyl_ptr[i*3+2] = c.phi;
        d_sph_ptr[i*3+0] = s.r;   d_sph_ptr[i*3+1] = s.theta; d_sph_ptr[i*3+2] = s.phi;
    });
    std::vector<double> cyl_c(NPT * 3), sph_c(NPT * 3);
    d_cyl.to_host(cyl_c.data(), NPT * 3);
    d_sph.to_host(sph_c.data(), NPT * 3);
    double max_coord_err = 0.0;
    for(std::size_t i = 0; i < NPT * 3; ++i) {
        max_coord_err = std::max(max_coord_err, std::fabs(cyl_s[i] - cyl_c[i]));
        max_coord_err = std::max(max_coord_err, std::fabs(sph_s[i] - sph_c[i]));
    }
    // host glibc vs CUDA-intrinsic sin/cos in math::sincos diverge by ~1 ULP per call;
    // atan2 here uses our own polynomial so it is bit-exact except for FMA-contraction
    // differences. 1e-13 is the same loose-but-meaningful threshold used for trigMultiAngle.
    const double COORD_TOL = 1e-13;
    bool ok_coord = (max_coord_err <= COORD_TOL);

    // ----- Tier 1 analytic potentials: Serial vs Cuda parity + leaf vs virtual -----
    // Each potential's evalmanyCarT<T,Policy> template is instantiated here for Cuda
    // (kernel generated by nvcc in this TU) and Serial (CPU plain loop). The existing
    // CPU virtual eval (potential_analytic.o, linked from agama.so) is the cross-check
    // for the leaf math — same Phi expression must produce the same value at FP level.
    {
        const std::size_t NN = 1024;
        std::vector<double> xyz_h(NN * 3);  // packed input array
        for(std::size_t i = 0; i < NN; ++i) {
            const double xi = 0.05 * static_cast<double>(i + 1);
            xyz_h[i*3+0] = 0.1 + xi;
            xyz_h[i*3+1] = 0.07 - 0.3 * xi;
            xyz_h[i*3+2] = 0.02 + 0.5 * xi;
        }
        potential::Plummer       plummer (1.0, 1.0);
        potential::Isochrone     iso     (1.0, 1.0);
        potential::NFW           nfw     (1.0, 1.0);
        potential::MiyamotoNagai mn      (1.0, 1.0, 0.3);
        potential::Logarithmic   logp    (/*v0=*/1.0, /*core=*/0.1,
                                          /*p=*/0.9, /*q=*/0.7, /*L=*/1.0);
        potential::Harmonic      harm    (/*Omega=*/1.0, /*p=*/0.8, /*q=*/0.5);
        // spherical case only (axisRatioY=axisRatioZ=1) -- the GPU batch path
        // rejects triaxial Dehnen (see potential_dehnen.h / potential_gpu.cpp)
        potential::Dehnen        dehnen  (/*mass=*/1.0, /*scalerad=*/1.0, /*gamma=*/1.0);
        bool ok_pots =
            check_pot_parity(plummer, "Plummer",       xyz_h) &&
            check_pot_parity(iso,     "Isochrone",     xyz_h) &&
            check_pot_parity(nfw,     "NFW",           xyz_h) &&
            check_pot_parity(mn,      "MiyamotoNagai", xyz_h) &&
            check_pot_parity(logp,    "Logarithmic",   xyz_h) &&
            check_pot_parity(harm,    "Harmonic",      xyz_h) &&
            check_pot_parity(dehnen,  "Dehnen(sph)",   xyz_h);
        if(!ok_pots) {
            std::fprintf(stderr, "FAIL (Tier 1 analytic potential parity)\n");
            return 1;
        }

        // ----- Tier 3 on-ramp: NFW fused Phi+acc batch, Serial vs Cuda -----
        // evalmanyPhiAccCarT runs the same nfw_eval leaf on both policies; the
        // fused kernel writes Phi and the packed Cartesian acceleration in one
        // launch. Absolute tolerance 1e-13 (values are O(1) for M=rs=1).
        {
            std::vector<double> phi_s(NN), acc_s(NN * 3);
            nfw.evalmanyPhiAccCarT<double>(Serial{}, NN, xyz_h.data(),
                phi_s.data(), acc_s.data());
            device_array<double> d_xyz2(NN * 3);
            d_xyz2.from_host(xyz_h.data(), NN * 3);
            device_array<double> d_phi2(NN), d_acc2(NN * 3);
            nfw.evalmanyPhiAccCarT<double>(Cuda{}, NN, d_xyz2.data(),
                d_phi2.data(), d_acc2.data());
            std::vector<double> phi_c(NN), acc_c(NN * 3);
            d_phi2.to_host(phi_c.data(), NN);
            d_acc2.to_host(acc_c.data(), NN * 3);
            double max_err = 0.0;
            for(std::size_t i = 0; i < NN; ++i)
                max_err = std::max(max_err, std::fabs(phi_s[i] - phi_c[i]));
            for(std::size_t i = 0; i < NN * 3; ++i)
                max_err = std::max(max_err, std::fabs(acc_s[i] - acc_c[i]));
            const double ACC_TOL = 1e-13;
            bool ok_acc = (max_err <= ACC_TOL);
            std::printf("[CUDA]  NFW evalmanyPhiAccCarT (fused Phi+acc, N=%zu): "
                "Serial-vs-Cuda max |err| = %.3e, tol = %.1e -> %s\n",
                NN, max_err, ACC_TOL, ok_acc ? "OK" : "FAIL");
            if(!ok_acc) {
                std::fprintf(stderr, "FAIL (NFW fused Phi+acc parity)\n");
                return 1;
            }
        }

        // ----- Dehnen (spherical) fused Phi+acc batch, Serial vs Cuda -----
        // Same fused-kernel pattern as NFW above, via the dehnen_eval leaf.
        {
            std::vector<double> phi_s(NN), acc_s(NN * 3);
            dehnen.evalmanyPhiAccCarT<double>(Serial{}, NN, xyz_h.data(),
                phi_s.data(), acc_s.data());
            device_array<double> d_xyz2(NN * 3);
            d_xyz2.from_host(xyz_h.data(), NN * 3);
            device_array<double> d_phi2(NN), d_acc2(NN * 3);
            dehnen.evalmanyPhiAccCarT<double>(Cuda{}, NN, d_xyz2.data(),
                d_phi2.data(), d_acc2.data());
            std::vector<double> phi_c(NN), acc_c(NN * 3);
            d_phi2.to_host(phi_c.data(), NN);
            d_acc2.to_host(acc_c.data(), NN * 3);
            double max_err = 0.0;
            for(std::size_t i = 0; i < NN; ++i)
                max_err = std::max(max_err, std::fabs(phi_s[i] - phi_c[i]));
            for(std::size_t i = 0; i < NN * 3; ++i)
                max_err = std::max(max_err, std::fabs(acc_s[i] - acc_c[i]));
            const double ACC_TOL = 1e-13;
            bool ok_acc = (max_err <= ACC_TOL);
            std::printf("[CUDA]  Dehnen(sph) evalmanyPhiAccCarT (fused Phi+acc, N=%zu): "
                "Serial-vs-Cuda max |err| = %.3e, tol = %.1e -> %s\n",
                NN, max_err, ACC_TOL, ok_acc ? "OK" : "FAIL");
            if(!ok_acc) {
                std::fprintf(stderr, "FAIL (Dehnen fused Phi+acc parity)\n");
                return 1;
            }
        }

        // ----- Dehnen density (triaxial-general leaf), Serial vs Cuda -----
        // evalmanyDensCarT supports arbitrary axis ratios (density has a closed
        // form even where the potential does not); exercise it on a genuinely
        // triaxial instance to confirm the leaf itself (not just the spherical
        // gate) is correct -- this instance is NOT dispatched through
        // potential_gpu.cpp/try_dispatch (which gates the whole type on
        // sphericity), only called directly here.
        {
            potential::Dehnen dehnenTri(/*mass=*/1.0, /*scalerad=*/1.0, /*gamma=*/1.2,
                /*axisRatioY=*/0.8, /*axisRatioZ=*/0.6);
            std::vector<double> rho_s(NN);
            dehnenTri.evalmanyDensCarT<double>(Serial{}, NN, xyz_h.data(), rho_s.data());
            device_array<double> d_xyz3(NN * 3);
            d_xyz3.from_host(xyz_h.data(), NN * 3);
            device_array<double> d_rho3(NN);
            dehnenTri.evalmanyDensCarT<double>(Cuda{}, NN, d_xyz3.data(), d_rho3.data());
            std::vector<double> rho_c(NN);
            d_rho3.to_host(rho_c.data(), NN);
            double max_rho = 1e-300, max_err = 0.0, max_virt_err = 0.0;
            for(std::size_t i = 0; i < NN; ++i)
                max_rho = std::max(max_rho, std::fabs(rho_s[i]));
            for(std::size_t i = 0; i < NN; ++i)
                max_err = std::max(max_err, std::fabs(rho_s[i] - rho_c[i]));
            for(std::size_t i = 0; i < 8 && i < NN; ++i) {
                const coord::PosCar p(xyz_h[i*3+0], xyz_h[i*3+1], xyz_h[i*3+2]);
                max_virt_err = std::max(max_virt_err,
                    std::fabs(dehnenTri.density(p) - rho_s[i]));
            }
            const double RHO_TOL = 1e-13 * max_rho;
            bool ok_rho = (max_err <= RHO_TOL) && (max_virt_err <= RHO_TOL);
            std::printf("[CUDA]  Dehnen(triaxial) evalmanyDensCarT: max|rho|=%.3e   "
                "Serial-vs-Cuda |err|=%.3e   leaf-vs-virtual |err|=%.3e   tol=%.1e -> %s\n",
                max_rho, max_err, max_virt_err, RHO_TOL, ok_rho ? "OK" : "FAIL");
            if(!ok_rho) {
                std::fprintf(stderr, "FAIL (Dehnen triaxial density parity)\n");
                return 1;
            }
        }

        // ----- UniformAcceleration: Serial vs Cuda + leaf-vs-virtual -----
        // Not registered in potential_gpu.cpp/potential_descriptor.h (see the
        // class-level comment in potential_composite.h for why -- it is
        // genuinely time-dependent and the dispatch tables carry no time
        // parameter), so this is a direct call, not a check_pot_parity() /
        // try_dispatch() exercise. accx/accy/accz are non-constant splines so
        // that evaluating at a nonzero, non-grid-point `time` is a meaningful
        // check of the host-side spline evaluation feeding the device leaf.
        {
            std::vector<double> tk = {0.0, 1.0, 2.0, 3.0};
            math::CubicSpline accx(tk, {0.1, 0.3, -0.2, 0.5});
            math::CubicSpline accy(tk, {-0.4, 0.2, 0.1, -0.1});
            math::CubicSpline accz(tk, {0.05, -0.05, 0.2, 0.0});
            potential::UniformAcceleration ua(accx, accy, accz);
            const double T_EVAL = 1.35;   // interior, non-grid-point time
            std::vector<double> phi_s(NN), acc_s(NN * 3);
            ua.evalmanyPhiAccCarT<double>(Serial{}, NN, xyz_h.data(),
                phi_s.data(), acc_s.data(), T_EVAL);
            device_array<double> d_xyz4(NN * 3);
            d_xyz4.from_host(xyz_h.data(), NN * 3);
            device_array<double> d_phi4(NN), d_acc4(NN * 3);
            ua.evalmanyPhiAccCarT<double>(Cuda{}, NN, d_xyz4.data(),
                d_phi4.data(), d_acc4.data(), T_EVAL);
            std::vector<double> phi_c(NN), acc_c(NN * 3);
            d_phi4.to_host(phi_c.data(), NN);
            d_acc4.to_host(acc_c.data(), NN * 3);
            double max_phi = 1e-300, max_err = 0.0, max_virt_err = 0.0;
            for(std::size_t i = 0; i < NN; ++i)
                max_phi = std::max(max_phi, std::fabs(phi_s[i]));
            for(std::size_t i = 0; i < NN; ++i)
                max_err = std::max(max_err, std::fabs(phi_s[i] - phi_c[i]));
            for(std::size_t i = 0; i < NN * 3; ++i)
                max_err = std::max(max_err, std::fabs(acc_s[i] - acc_c[i]));
            for(std::size_t i = 0; i < 8 && i < NN; ++i) {
                const coord::PosCar p(xyz_h[i*3+0], xyz_h[i*3+1], xyz_h[i*3+2]);
                max_virt_err = std::max(max_virt_err,
                    std::fabs(ua.value(p, T_EVAL) - phi_s[i]));
            }
            const double UA_TOL = 1e-13 * max_phi;
            bool ok_ua = (max_err <= UA_TOL) && (max_virt_err <= UA_TOL);
            std::printf("[CUDA]  UniformAcceleration evalmanyPhiAccCarT: max|phi|=%.3e   "
                "Serial-vs-Cuda |err|=%.3e   leaf-vs-virtual |err|=%.3e   tol=%.1e -> %s\n",
                max_phi, max_err, max_virt_err, UA_TOL, ok_ua ? "OK" : "FAIL");
            if(!ok_ua) {
                std::fprintf(stderr, "FAIL (UniformAcceleration parity)\n");
                return 1;
            }
        }

        // ----- DiskAnsatz (Tier 1): 5 recognized radial/vertical functor
        // combinations, Serial-vs-Cuda parity + leaf-vs-virtual, via the
        // disk_ansatz_eval/disk_ansatz_rho leaves (potential_disk.h). Exercises
        // both radial types (Exp, RichExp with innerCutoffRadius>0 AND
        // sersicIndex!=1) crossed with all 3 vertical types (Exp, Isothermal,
        // Thin) -- check_pot_parity() below cross-checks against the existing
        // CPU virtual eval, which still calls the SAME functor evalDeriv
        // methods (now thin leaf wrappers, see potential_disk.h), so this also
        // confirms the relocation didn't change anything CPU-side.
        {
            potential::DiskAnsatz diskExpExp(potential::DiskParam(
                /*surfaceDensity=*/1.0, /*scaleRadius=*/1.0, /*scaleHeight=*/0.3));
            potential::DiskAnsatz diskExpIso(potential::DiskParam(
                /*surfaceDensity=*/2.0, /*scaleRadius=*/1.5, /*scaleHeight=*/-0.4));
            potential::DiskAnsatz diskExpThin(potential::DiskParam(
                /*surfaceDensity=*/1.5, /*scaleRadius=*/1.0, /*scaleHeight=*/0.0));
            potential::DiskAnsatz diskRichExp(potential::DiskParam(
                /*surfaceDensity=*/1.0, /*scaleRadius=*/1.0, /*scaleHeight=*/0.3,
                /*innerCutoffRadius=*/0.2, /*modulationAmplitude=*/0.1, /*sersicIndex=*/1.5));
            potential::DiskAnsatz diskRichIso(potential::DiskParam(
                /*surfaceDensity=*/1.0, /*scaleRadius=*/1.0, /*scaleHeight=*/-0.3,
                /*innerCutoffRadius=*/0.15, /*modulationAmplitude=*/0.0, /*sersicIndex=*/2.0));
            bool ok_disk =
                check_pot_parity(diskExpExp,  "Disk(Exp,Exp)",     xyz_h) &&
                check_pot_parity(diskExpIso,  "Disk(Exp,Iso)",     xyz_h) &&
                check_pot_parity(diskExpThin, "Disk(Exp,Thin)",    xyz_h) &&
                check_pot_parity(diskRichExp, "Disk(RichExp,Exp)", xyz_h) &&
                check_pot_parity(diskRichIso, "Disk(RichExp,Iso)", xyz_h);
            if(!ok_disk) {
                std::fprintf(stderr, "FAIL (DiskAnsatz potential parity)\n");
                return 1;
            }

            // ----- DiskAnsatz fused Phi+acc + density, Serial vs Cuda -----
            // Same fused-kernel pattern as NFW/Dehnen above, via the
            // disk_ansatz_eval/disk_ansatz_rho leaves, on the RichExp+Exp
            // combination (exercises the Sersic/inner-cutoff/modulation branch).
            {
                std::vector<double> phi_s(NN), acc_s(NN * 3), rho_s(NN);
                diskRichExp.evalmanyPhiAccCarT<double>(Serial{}, NN, xyz_h.data(),
                    phi_s.data(), acc_s.data());
                diskRichExp.evalmanyDensCarT<double>(Serial{}, NN, xyz_h.data(), rho_s.data());
                device_array<double> d_xyz5(NN * 3);
                d_xyz5.from_host(xyz_h.data(), NN * 3);
                device_array<double> d_phi5(NN), d_acc5(NN * 3), d_rho5(NN);
                diskRichExp.evalmanyPhiAccCarT<double>(Cuda{}, NN, d_xyz5.data(),
                    d_phi5.data(), d_acc5.data());
                diskRichExp.evalmanyDensCarT<double>(Cuda{}, NN, d_xyz5.data(), d_rho5.data());
                std::vector<double> phi_c(NN), acc_c(NN * 3), rho_c(NN);
                d_phi5.to_host(phi_c.data(), NN);
                d_acc5.to_host(acc_c.data(), NN * 3);
                d_rho5.to_host(rho_c.data(), NN);
                double max_err = 0.0, max_rho = 1e-300, max_rho_err = 0.0, max_virt_rho_err = 0.0;
                for(std::size_t i = 0; i < NN; ++i)
                    max_err = std::max(max_err, std::fabs(phi_s[i] - phi_c[i]));
                for(std::size_t i = 0; i < NN * 3; ++i)
                    max_err = std::max(max_err, std::fabs(acc_s[i] - acc_c[i]));
                for(std::size_t i = 0; i < NN; ++i) {
                    max_rho = std::max(max_rho, std::fabs(rho_s[i]));
                    max_rho_err = std::max(max_rho_err, std::fabs(rho_s[i] - rho_c[i]));
                }
                for(std::size_t i = 0; i < 8 && i < NN; ++i) {
                    const coord::PosCar p(xyz_h[i*3+0], xyz_h[i*3+1], xyz_h[i*3+2]);
                    max_virt_rho_err = std::max(max_virt_rho_err,
                        std::fabs(diskRichExp.density(p) - rho_s[i]));
                }
                const double ACC_TOL = 1e-13;
                const double RHO_TOL = 1e-13 * max_rho;
                bool ok_disk2 = (max_err <= ACC_TOL) && (max_rho_err <= RHO_TOL) &&
                    (max_virt_rho_err <= RHO_TOL);
                std::printf("[CUDA]  Disk(RichExp,Exp) evalmanyPhiAccCarT+evalmanyDensCarT: "
                    "Phi/acc Serial-vs-Cuda |err|=%.3e (tol %.1e)   "
                    "dens Serial-vs-Cuda |err|=%.3e   leaf-vs-virtual |err|=%.3e (tol %.1e) -> %s\n",
                    max_err, ACC_TOL, max_rho_err, max_virt_rho_err, RHO_TOL, ok_disk2 ? "OK" : "FAIL");
                if(!ok_disk2) {
                    std::fprintf(stderr, "FAIL (DiskAnsatz fused Phi+acc / density parity)\n");
                    return 1;
                }
            }

            // ----- DiskAnsatz edge cases: R=0, z=0, full origin -----
            // The combine leaf has explicit guards for r=0 (rinv fallback,
            // matching DiskAnsatz::evalCyl) that the generic xyz_h grid above
            // never exercises (z is never exactly 0 there); check leaf-vs-
            // virtual directly at these points via the fused batch method (N=1).
            {
                const double edge_xyz[5][3] = {
                    {0.0,  0.0,  0.0},   // full origin: r=0, R=0, z=0
                    {0.0,  0.0,  1.3},   // R=0, z!=0 (on the symmetry axis)
                    {1.7,  0.0,  0.0},   // z=0, R!=0 (in the disk plane)
                    {0.9, -0.4,  0.0},   // z=0, R!=0 (general in-plane point)
                    {0.3,  0.2, -2.1},   // ordinary point
                };
                double max_edge_err = 0.0;
                for(int i = 0; i < 5; i++) {
                    const coord::PosCar p(edge_xyz[i][0], edge_xyz[i][1], edge_xyz[i][2]);
                    double phi_v, phi_l;
                    coord::GradCar grad_v;
                    diskRichIso.eval(p, &phi_v, &grad_v, NULL);
                    double acc_l[3];
                    diskRichIso.evalmanyPhiAccCarT<double>(Serial{}, 1, edge_xyz[i], &phi_l, acc_l);
                    max_edge_err = std::max(max_edge_err, std::fabs(phi_v - phi_l));
                    max_edge_err = std::max(max_edge_err, std::fabs(-grad_v.dx - acc_l[0]));
                    max_edge_err = std::max(max_edge_err, std::fabs(-grad_v.dy - acc_l[1]));
                    max_edge_err = std::max(max_edge_err, std::fabs(-grad_v.dz - acc_l[2]));
                }
                const double EDGE_TOL = 1e-12;
                bool ok_edge = max_edge_err <= EDGE_TOL;
                std::printf("[CUDA]  Disk(RichExp,Iso) edge cases (R=0 / z=0 / origin), "
                    "leaf-vs-virtual: max |err| = %.3e, tol = %.1e -> %s\n",
                    max_edge_err, EDGE_TOL, ok_edge ? "OK" : "FAIL");
                if(!ok_edge) {
                    std::fprintf(stderr, "FAIL (DiskAnsatz edge-case parity)\n");
                    return 1;
                }
            }

            // ----- DiskAnsatz Tier 3 force descriptor vs virtual eval -----
            // Exercises the GPU_POT_DISK_ANSATZ case in gpu_term_phi_acc
            // (potential_descriptor.h), including the radialType/verticalType
            // pair packed into GpuPotTerm::aux -- buildGpuPotDesc/
            // gpu_desc_phi_acc are called directly on a bare DiskAnsatz (no
            // Composite wrapper needed; buildGpuPotDesc handles a single
            // non-composite potential too).
            {
                potential::GpuPotDesc<double> desc;
                bool ok_desc = potential::buildGpuPotDesc(diskRichExp, desc);
                double max_rel = 0;
                if(ok_desc && desc.nterms == 1) {
                    for(int i = 0; i < 64; i++) {
                        const double x = 0.05 + 0.037 * i,
                                     y = -0.3 + 0.021 * (i % 17),
                                     z = 0.4 - 0.013 * (i % 23);
                        double phi_d, acc_d[3];
                        potential::gpu_desc_phi_acc(desc, x, y, z, &phi_d, acc_d);
                        double phi_v;
                        coord::GradCar grad;
                        diskRichExp.eval(coord::PosCar(x, y, z), &phi_v, &grad, NULL);
                        const double acc_v[3] = { -grad.dx, -grad.dy, -grad.dz };
                        double scale = std::max(1e-300, std::fabs(phi_v));
                        max_rel = std::max(max_rel, std::fabs(phi_d - phi_v) / scale);
                        for(int k = 0; k < 3; k++) {
                            scale = std::max(1e-300, std::fabs(acc_v[k]));
                            max_rel = std::max(max_rel, std::fabs(acc_d[k] - acc_v[k]) / scale);
                        }
                    }
                }
                const double DESC_TOL = 1e-13;
                bool ok_desc_parity = ok_desc && desc.nterms == 1 && max_rel <= DESC_TOL;
                std::printf("[T3]    DiskAnsatz force descriptor vs virtual eval (64 pts): "
                    "max rel err = %.3e, tol = %.1e -> %s\n",
                    max_rel, DESC_TOL, ok_desc_parity ? "OK" : "FAIL");
                if(!ok_desc_parity) {
                    std::fprintf(stderr, "FAIL (DiskAnsatz force descriptor parity)\n");
                    return 1;
                }
            }
        }
    }

    // ----- math::trigMultiAngle on Cuda (Tier 0 worked example) -----
    // Each thread handles one phi sample, writes its 2*MM trig values into a
    // device buffer. We compare bit-for-bit with the Serial reference above.
    device_array<double> d_trig(NP * 2 * MM);
    double* dtrig = d_trig.data();
    forall(Cuda{}, NP, [=] AGAMA_DEVICE (std::size_t p) {
        const double phi = (p + 0.5) * (2.0 * 3.14159265358979323846 / NP);
        math::trigMultiAngle(phi, MM, /*needSine=*/true, &dtrig[p * 2 * MM]);
    });
    std::vector<double> trig_c(NP * 2 * MM);
    d_trig.to_host(trig_c.data(), NP * 2 * MM);

    bool ok_gpu  = (out_s == out_c) && (rsum_s == rsum_c);
    // trigMultiAngle: tolerance check, not bit-exact. Host glibc sin/cos and device
    // CUDA sin/cos differ by ~1-3 ULPs, plus nvcc's default FMA contraction shifts
    // intermediates. fp64 in [-1, 1] gives 1 ULP ≈ 1e-16, so 1e-13 (~1000 ULPs) is
    // a loose-but-meaningful threshold for an 8-term recurrence.
    double max_trig_err = 0.0;
    for(std::size_t i = 0; i < trig_s.size(); ++i) {
        const double e = std::fabs(trig_s[i] - trig_c[i]);
        if (e > max_trig_err) max_trig_err = e;
    }
    const double TRIG_TOL = 1e-13;
    bool ok_trig = (max_trig_err <= TRIG_TOL);
    std::printf("[CUDA]  forall Serial==Cuda:   %s   reduce Serial==Cuda:   %s\n",
        (out_s == out_c) ? "yes" : "NO",
        (rsum_s == rsum_c) ? "yes" : "NO");
    std::printf("[CUDA]  reduce result: %.1f\n", rsum_c);
    std::printf("[CUDA]  trigMultiAngle Serial vs Cuda (NP=%zu, mmax=%u): max |err| = %.3e, tol = %.1e -> %s\n",
        NP, MM, max_trig_err, TRIG_TOL, ok_trig ? "OK" : "FAIL");
    std::printf("[CUDA]  toPos<Car,Cyl>/<Car,Sph> Serial vs Cuda (N=%zu): max |err| = %.3e, tol = %.1e -> %s\n",
        NPT, max_coord_err, COORD_TOL, ok_coord ? "OK" : "FAIL");

    if (!(ok_cpu && ok_gpu && ok_trig && ok_coord)) {
        std::fprintf(stderr, "FAIL\n");
        return 1;
    }
    std::printf("PASS (Serial/OpenMP/Cuda agree on forall, reduce, trigMultiAngle, and toPos to %.0e)\n", TRIG_TOL);

    // ============================================================
    // Timing: Serial vs OpenMP vs Cuda. Two sizes (1M, 16M) × two precisions (fp64, fp32).
    // On RTX 30xx/L40 the fp64:fp32 ratio is 1:64, so the fp32 vs fp64 gap is
    // the most important perf data on those cards.
    // ============================================================
    const int TRIALS = 5;
    std::printf("\nTimings (best of %d, after warm-up). Speedup factors are vs Serial.\n", TRIALS);

    run_perf_T<double>(TRIALS, std::size_t(1) << 20, "double");  // 1 M  / 8 MiB fp64 / 4 MiB fp32
    run_perf_T<float >(TRIALS, std::size_t(1) << 20, "float ");
    run_perf_T<double>(TRIALS, std::size_t(1) << 24, "double");  // 16 M / 128 MiB fp64 / 64 MiB fp32
    run_perf_T<float >(TRIALS, std::size_t(1) << 24, "float ");

    // --- trigMultiAngle (fp64 only at this point; templating it requires editing math_sphharm.h)
    // Two sizes to show how kernel-launch overhead amortizes as NP grows.
    auto run_trig = [&](std::size_t NP_T, unsigned MM_T, const char* sizename) {
        std::vector<double> tbuf_s(NP_T * 2 * MM_T), tbuf_o(NP_T * 2 * MM_T);
        auto host_one = [&](std::vector<double>& dst, std::size_t p) {
            const double phi = (p + 0.5) * (2.0 * 3.14159265358979323846 / NP_T);
            math::trigMultiAngle(phi, MM_T, true, &dst[p * 2 * MM_T]);
        };
        double ts = time_ms_min(TRIALS, [&]{ forall(Serial{}, NP_T, [&](std::size_t p){ host_one(tbuf_s, p); }); });
        double to = time_ms_min(TRIALS, [&]{ forall(OpenMP{}, NP_T, [&](std::size_t p){ host_one(tbuf_o, p); }); });
        device_array<double> d_trig_T(NP_T * 2 * MM_T);
        double* dt = d_trig_T.data();
        double tc = time_ms_min(TRIALS, [&]{
            forall(Cuda{}, NP_T, [=] AGAMA_DEVICE (std::size_t p) {
                const double phi = (p + 0.5) * (2.0 * 3.14159265358979323846 / NP_T);
                math::trigMultiAngle(phi, MM_T, true, &dt[p * 2 * MM_T]);
            });
            double tmp; cudaMemcpy(&tmp, dt, sizeof(double), cudaMemcpyDeviceToHost);
        });
        std::printf("  trigMultiAngle fp64  NP=%-8zu mmax=%u  %s : Serial %8.3f ms   OpenMP %8.3f ms (%5.1fx)   Cuda %8.3f ms (%6.1fx)\n",
            NP_T, MM_T, sizename, ts, to, ts / to, tc, ts / tc);
    };
    std::printf("\n=== trigMultiAngle launch-amortization sweep (fp64) ===\n");
    run_trig(1u << 14, 16, "small ");   //  16K threads × 16 ops
    run_trig(1u << 18, 16, "medium");   // 256K
    run_trig(1u << 20, 16, "large ");   //   1M

    // ============================================================
    // Tier 1 NFW timing: fp64 + fp32 via NFW::evalmanyCarT<T,Policy>. Both precisions
    // run through the same class template method; the kernel body is templated on T.
    // Measures end-to-end Phi(x,y,z) at N points (packed xyz input, length 3N).
    // ============================================================
    std::printf("\n=== Tier 1 NFW: Phi(xyz) timings (Serial / OpenMP / Cuda) ===\n");
    auto run_nfw = [&](std::size_t N, const char* sizename) {
        potential::NFW pot(1.0, 1.0);

        auto run_precision = [&](auto sample_T, const char* tname) {
            using T = decltype(sample_T);
            std::vector<T> xyz_h(N * 3), phi_s(N), phi_o(N);
            for(std::size_t i = 0; i < N; ++i) {
                const T xi = T(1e-3) + T(1e-5) * static_cast<T>(i);
                xyz_h[i*3+0] = T(0.1) + xi;
                xyz_h[i*3+1] = T(0.07) - T(0.3) * xi;
                xyz_h[i*3+2] = T(0.02) + T(0.5) * xi;
            }
            double ts = time_ms_min(TRIALS, [&]{ pot.template evalmanyCarT<T>(Serial{}, N, xyz_h.data(), phi_s.data()); });
            double to = time_ms_min(TRIALS, [&]{ pot.template evalmanyCarT<T>(OpenMP{}, N, xyz_h.data(), phi_o.data()); });
            device_array<T> d_xyz(N * 3);
            d_xyz.from_host(xyz_h.data(), N * 3);
            device_array<T> d_phi(N);
            double tc = time_ms_min(TRIALS, [&]{
                pot.template evalmanyCarT<T>(Cuda{}, N, d_xyz.data(), d_phi.data());
                T tmp; cudaMemcpy(&tmp, d_phi.data(), sizeof(T), cudaMemcpyDeviceToHost);
            });
            std::printf("  NFW %s  N=%-8zu  %s : Serial %8.3f ms   OpenMP %8.3f ms (%5.1fx)   Cuda %8.3f ms (%6.1fx vs Serial, %5.1fx vs OpenMP)\n",
                tname, N, sizename, ts, to, ts / to, tc, ts / tc, to / tc);
        };
        run_precision(double{}, "fp64");
        run_precision(float{},  "fp32");
    };
    run_nfw(1u << 18, "small ");   // 262K — kernel-launch + H2D-bound regime
    run_nfw(1u << 20, "medium");   //   1M
    run_nfw(1u << 22, "large ");   //   4M — bandwidth-bound regime

    return 0;
#else
    if (!ok_cpu) {
        std::fprintf(stderr, "FAIL (CPU only)\n");
        return 1;
    }
    std::printf("[CPU]   trigMultiAngle ran for %zu samples, mmax=%u (no GPU to compare against)\n", NP, MM);
    std::printf("PASS (CPU only — HAVE_CUDA not defined)\n");
    return 0;
#endif
}
