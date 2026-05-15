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
#include "math_sphharm.h"   // for math::trigMultiAngle (Tier 0 device-inline leaf)
#include "coord.h"          // toPos<Car,Cyl>, toPos<Car,Sph> (Tier 0 Phase 2 device-inline)
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace {
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
