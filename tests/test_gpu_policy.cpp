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
#include "math_spline.h"         // evalQuinticSplineRaw/evalQuinticSpline2dRaw (Tier 2 prereq)
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
#include <cstdint>   // uint64_t (bitwise double comparison via memcpy)
#include <cstring>   // std::memcmp (bitwise trajectory comparison, NAN-safe)
#include <cstdint>   // std::uint64_t (bitwise quintic-spline comparison)
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>    // std::mt19937_64 (Tier 2 toGrad/toHess Cuda-parity block, CD_N random inputs)

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

/** Max scale-normalized error of a descriptor against the virtual eval of the same
    potential, at one time. Scale-normalized rather than pointwise-relative because a
    triaxial Logarithmic's Phi crosses zero on the sampled shell, where dividing by
    ~0 reports a huge error for a perfectly good result. */
static double desc_vs_virtual(const potential::GpuPotDesc<double>& d,
    const potential::BasePotential& pot, double time)
{
    double worst = 0, ref = 0;
    for(int i = 0; i < 64; i++) {
        const double x = 0.3 + 0.05 * i, y = -0.7 + 0.031 * (i % 17),
                     z = 0.2 + 0.017 * (i % 13);
        double phi_d, acc_d[3];
        potential::gpu_desc_phi_acc(d, x, y, z, &phi_d, acc_d, time);
        double phi_v;
        coord::GradCar grad;
        pot.eval(coord::PosCar(x, y, z), &phi_v, &grad, NULL, time);
        worst = std::max(worst, std::fabs(phi_d - phi_v));
        ref   = std::max(ref,   std::fabs(phi_v));
        const double av[3] = { -grad.dx, -grad.dy, -grad.dz };
        for(int k = 0; k < 3; k++) {
            worst = std::max(worst, std::fabs(acc_d[k] - av[k]));
            ref   = std::max(ref,   std::fabs(av[k]));
        }
    }
    return worst / std::max(1e-300, ref);
}

// =============================================================================
// Tier 2 commit 1 FROZEN REFERENCE: verbatim copies of the PRE-PROMOTION
// coord.cpp bodies for the 12 Car/Cyl/Sph toGrad/toHess specializations, kept
// byte-identical to what coord.cpp had before those bodies moved to coord.h as
// AGAMA_DEVICE_INLINE. These are the ground truth the header versions are
// compared against below, in a same-TU pseudo-random sweep. DO NOT "clean up",
// reformat, or refactor these -- their entire value is being an untouched,
// independently-compiled restatement of the pre-move arithmetic living in the
// SAME translation unit as the code under test, so that any compiler-context
// difference (inlining, scheduling, FMA contraction -- see the comment on the
// sweep below) moves BOTH sides together and the comparison isolates changes
// to coord.h's arithmetic itself, not changes to surrounding code.
// =============================================================================
namespace {

coord::GradCar ref_toGrad_Car_from_Cyl(const coord::GradCyl& src, const coord::PosDerivT<coord::Car, coord::Cyl>& deriv) {
    coord::GradCar dest;
    dest.dx = src.dR*deriv.dRdx + src.dphi*deriv.dphidx;
    dest.dy = src.dR*deriv.dRdy + src.dphi*deriv.dphidy;
    dest.dz = src.dz;
    return dest;
}

coord::GradCar ref_toGrad_Car_from_Sph(const coord::GradSph& src, const coord::PosDerivT<coord::Car, coord::Sph>& deriv) {
    coord::GradCar dest;
    dest.dx = src.dr*deriv.drdx + src.dtheta*deriv.dthetadx + src.dphi*deriv.dphidx;
    dest.dy = src.dr*deriv.drdy + src.dtheta*deriv.dthetady + src.dphi*deriv.dphidy;
    dest.dz = src.dr*deriv.drdz + src.dtheta*deriv.dthetadz;
    return dest;
}

coord::GradCyl ref_toGrad_Cyl_from_Car(const coord::GradCar& src, const coord::PosDerivT<coord::Cyl, coord::Car>& deriv) {
    coord::GradCyl dest;
    dest.dR = src.dx*deriv.dxdR + src.dy*deriv.dydR;
    dest.dz = src.dz;
    dest.dphi = src.dx*deriv.dxdphi + src.dy*deriv.dydphi;
    return dest;
}

coord::GradCyl ref_toGrad_Cyl_from_Sph(const coord::GradSph& src, const coord::PosDerivT<coord::Cyl, coord::Sph>& deriv) {
    coord::GradCyl dest;
    dest.dR = src.dr*deriv.drdR + src.dtheta*deriv.dthetadR;
    dest.dz = src.dr*deriv.drdz + src.dtheta*deriv.dthetadz;
    dest.dphi = src.dphi;
    return dest;
}

coord::GradSph ref_toGrad_Sph_from_Car(const coord::GradCar& src, const coord::PosDerivT<coord::Sph, coord::Car>& deriv) {
    coord::GradSph dest;
    dest.dr     = src.dx*deriv.dxdr     + src.dy*deriv.dydr     + src.dz*deriv.dzdr;
    dest.dtheta = src.dx*deriv.dxdtheta + src.dy*deriv.dydtheta + src.dz*deriv.dzdtheta;
    dest.dphi   = src.dx*deriv.dxdphi   + src.dy*deriv.dydphi;
    return dest;
}

coord::GradSph ref_toGrad_Sph_from_Cyl(const coord::GradCyl& src, const coord::PosDerivT<coord::Sph, coord::Cyl>& deriv) {
    coord::GradSph dest;
    dest.dr     = src.dR*deriv.dRdr     + src.dz*deriv.dzdr;
    dest.dtheta = src.dR*deriv.dRdtheta + src.dz*deriv.dzdtheta;
    dest.dphi   = src.dphi;
    return dest;
}

coord::HessCar ref_toHess_Car_from_Cyl(const coord::GradCyl& srcGrad, const coord::HessCyl& srcHess,
    const coord::PosDerivT<coord::Car, coord::Cyl>& deriv, const coord::PosDeriv2T<coord::Car, coord::Cyl>& deriv2) {
    coord::HessCar dest;
    dest.dx2 =
        (srcHess.dR2   *deriv.dRdx + srcHess.dRdphi*deriv.dphidx) * deriv.dRdx +
        (srcHess.dRdphi*deriv.dRdx + srcHess.dphi2 *deriv.dphidx) * deriv.dphidx +
        srcGrad.dR*deriv2.d2Rdx2   + srcGrad.dphi*deriv2.d2phidx2;
    dest.dxdy =
        (srcHess.dR2   *deriv.dRdy + srcHess.dRdphi*deriv.dphidy) * deriv.dRdx +
        (srcHess.dRdphi*deriv.dRdy + srcHess.dphi2 *deriv.dphidy) * deriv.dphidx +
        srcGrad.dR*deriv2.d2Rdxdy  + srcGrad.dphi*deriv2.d2phidxdy;
    dest.dy2 =
        (srcHess.dR2   *deriv.dRdy + srcHess.dRdphi*deriv.dphidy) * deriv.dRdy +
        (srcHess.dRdphi*deriv.dRdy + srcHess.dphi2 *deriv.dphidy) * deriv.dphidy +
        srcGrad.dR*deriv2.d2Rdy2   + srcGrad.dphi*deriv2.d2phidy2;
    dest.dxdz = srcHess.dRdz*deriv.dRdx + srcHess.dzdphi*deriv.dphidx;
    dest.dydz = srcHess.dRdz*deriv.dRdy + srcHess.dzdphi*deriv.dphidy;
    dest.dz2  = srcHess.dz2;
    return dest;
}

coord::HessCar ref_toHess_Car_from_Sph(const coord::GradSph& srcGrad, const coord::HessSph& srcHess,
    const coord::PosDerivT<coord::Car, coord::Sph>& deriv, const coord::PosDeriv2T<coord::Car, coord::Sph>& deriv2) {
    coord::HessCar dest;
    dest.dx2 =
        (srcHess.dr2     *deriv.drdx + srcHess.drdtheta  *deriv.dthetadx + srcHess.drdphi    *deriv.dphidx) * deriv.drdx +
        (srcHess.drdtheta*deriv.drdx + srcHess.dtheta2   *deriv.dthetadx + srcHess.dthetadphi*deriv.dphidx) * deriv.dthetadx +
        (srcHess.drdphi  *deriv.drdx + srcHess.dthetadphi*deriv.dthetadx + srcHess.dphi2     *deriv.dphidx) * deriv.dphidx +
        srcGrad.dr*deriv2.d2rdx2     + srcGrad.dtheta*deriv2.d2thetadx2  + srcGrad.dphi*deriv2.d2phidx2;
    dest.dxdy =
        (srcHess.dr2     *deriv.drdy + srcHess.drdtheta  *deriv.dthetady + srcHess.drdphi    *deriv.dphidy) * deriv.drdx +
        (srcHess.drdtheta*deriv.drdy + srcHess.dtheta2   *deriv.dthetady + srcHess.dthetadphi*deriv.dphidy) * deriv.dthetadx +
        (srcHess.drdphi  *deriv.drdy + srcHess.dthetadphi*deriv.dthetady + srcHess.dphi2     *deriv.dphidy) * deriv.dphidx +
        srcGrad.dr*deriv2.d2rdxdy    + srcGrad.dtheta*deriv2.d2thetadxdy + srcGrad.dphi*deriv2.d2phidxdy;
    dest.dxdz =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta  *deriv.dthetadz) * deriv.drdx +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2   *deriv.dthetadz) * deriv.dthetadx +
        (srcHess.drdphi  *deriv.drdz + srcHess.dthetadphi*deriv.dthetadz) * deriv.dphidx +
        srcGrad.dr*deriv2.d2rdxdz    + srcGrad.dtheta*deriv2.d2thetadxdz;
    dest.dy2 =
        (srcHess.dr2     *deriv.drdy + srcHess.drdtheta  *deriv.dthetady + srcHess.drdphi    *deriv.dphidy) * deriv.drdy +
        (srcHess.drdtheta*deriv.drdy + srcHess.dtheta2   *deriv.dthetady + srcHess.dthetadphi*deriv.dphidy) * deriv.dthetady +
        (srcHess.drdphi  *deriv.drdy + srcHess.dthetadphi*deriv.dthetady + srcHess.dphi2     *deriv.dphidy) * deriv.dphidy +
        srcGrad.dr*deriv2.d2rdy2     + srcGrad.dtheta*deriv2.d2thetady2  + srcGrad.dphi*deriv2.d2phidy2;
    dest.dydz =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta  *deriv.dthetadz) * deriv.drdy +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2   *deriv.dthetadz) * deriv.dthetady +
        (srcHess.drdphi  *deriv.drdz + srcHess.dthetadphi*deriv.dthetadz) * deriv.dphidy +
        srcGrad.dr*deriv2.d2rdydz    + srcGrad.dtheta*deriv2.d2thetadydz;
    dest.dz2 =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta  *deriv.dthetadz) * deriv.drdz +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2   *deriv.dthetadz) * deriv.dthetadz +
        srcGrad.dr*deriv2.d2rdz2     + srcGrad.dtheta*deriv2.d2thetadz2;
    return dest;
}

coord::HessCyl ref_toHess_Cyl_from_Car(const coord::GradCar& srcGrad, const coord::HessCar& srcHess,
    const coord::PosDerivT<coord::Cyl, coord::Car>& deriv, const coord::PosDeriv2T<coord::Cyl, coord::Car>& deriv2) {
    coord::HessCyl dest;
    dest.dR2 =
        (srcHess.dx2 *deriv.dxdR + srcHess.dxdy*deriv.dydR) * deriv.dxdR +
        (srcHess.dxdy*deriv.dxdR + srcHess.dy2 *deriv.dydR) * deriv.dydR;
    dest.dRdz = srcHess.dxdz*deriv.dxdR + srcHess.dydz*deriv.dydR;
    dest.dRdphi =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdR +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydR +
        srcGrad.dx*deriv2.d2xdRdphi + srcGrad.dy*deriv2.d2ydRdphi;
    dest.dz2 = srcHess.dz2;
    dest.dzdphi = (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi);
    dest.dphi2 =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdphi +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydphi +
        srcGrad.dx*deriv2.d2xdphi2 + srcGrad.dy*deriv2.d2ydphi2;
    return dest;
}

coord::HessCyl ref_toHess_Cyl_from_Sph(const coord::GradSph& srcGrad, const coord::HessSph& srcHess,
    const coord::PosDerivT<coord::Cyl, coord::Sph>& deriv, const coord::PosDeriv2T<coord::Cyl, coord::Sph>& deriv2) {
    coord::HessCyl dest;
    dest.dR2 =
        (srcHess.dr2     *deriv.drdR + srcHess.drdtheta*deriv.dthetadR) * deriv.drdR +
        (srcHess.drdtheta*deriv.drdR + srcHess.dtheta2 *deriv.dthetadR) * deriv.dthetadR +
        srcGrad.dr*deriv2.d2rdR2 + srcGrad.dtheta*deriv2.d2thetadR2;
    dest.dRdz =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta*deriv.dthetadz) * deriv.drdR +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2 *deriv.dthetadz) * deriv.dthetadR +
        srcGrad.dr*deriv2.d2rdRdz + srcGrad.dtheta*deriv2.d2thetadRdz;
    dest.dz2 =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta*deriv.dthetadz) * deriv.drdz +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2 *deriv.dthetadz) * deriv.dthetadz +
        srcGrad.dr*deriv2.d2rdz2 + srcGrad.dtheta*deriv2.d2thetadz2;
    dest.dRdphi = srcHess.drdphi*deriv.drdR + srcHess.dthetadphi*deriv.dthetadR;
    dest.dzdphi = srcHess.drdphi*deriv.drdz + srcHess.dthetadphi*deriv.dthetadz;
    dest.dphi2  = srcHess.dphi2;
    return dest;
}

coord::HessSph ref_toHess_Sph_from_Car(const coord::GradCar& srcGrad, const coord::HessCar& srcHess,
    const coord::PosDerivT<coord::Sph, coord::Car>& deriv, const coord::PosDeriv2T<coord::Sph, coord::Car>& deriv2) {
    coord::HessSph dest;
    dest.dr2 =
        (srcHess.dx2 *deriv.dxdr + srcHess.dxdy*deriv.dydr + srcHess.dxdz*deriv.dzdr) * deriv.dxdr +
        (srcHess.dxdy*deriv.dxdr + srcHess.dy2 *deriv.dydr + srcHess.dydz*deriv.dzdr) * deriv.dydr +
        (srcHess.dxdz*deriv.dxdr + srcHess.dydz*deriv.dydr + srcHess.dz2 *deriv.dzdr) * deriv.dzdr;
    dest.drdtheta =
        (srcHess.dx2 *deriv.dxdtheta + srcHess.dxdy*deriv.dydtheta + srcHess.dxdz*deriv.dzdtheta) * deriv.dxdr +
        (srcHess.dxdy*deriv.dxdtheta + srcHess.dy2 *deriv.dydtheta + srcHess.dydz*deriv.dzdtheta) * deriv.dydr +
        (srcHess.dxdz*deriv.dxdtheta + srcHess.dydz*deriv.dydtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdr +
        srcGrad.dx*deriv2.d2xdrdtheta + srcGrad.dy*deriv2.d2ydrdtheta + srcGrad.dz*deriv2.d2zdrdtheta;
    dest.drdphi =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi)*deriv.dxdr +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi)*deriv.dydr +
        (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi)*deriv.dzdr +
        srcGrad.dx*deriv2.d2xdrdphi + srcGrad.dy*deriv2.d2ydrdphi;
    dest.dtheta2 =
        (srcHess.dx2 *deriv.dxdtheta + srcHess.dxdy*deriv.dydtheta + srcHess.dxdz*deriv.dzdtheta) * deriv.dxdtheta +
        (srcHess.dxdy*deriv.dxdtheta + srcHess.dy2 *deriv.dydtheta + srcHess.dydz*deriv.dzdtheta) * deriv.dydtheta +
        (srcHess.dxdz*deriv.dxdtheta + srcHess.dydz*deriv.dydtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdtheta +
        srcGrad.dx*deriv2.d2xdtheta2 + srcGrad.dy*deriv2.d2ydtheta2 + srcGrad.dz*deriv2.d2zdtheta2;
    dest.dthetadphi =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdtheta +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydtheta +
        (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi) * deriv.dzdtheta +
        srcGrad.dx*deriv2.d2xdthetadphi + srcGrad.dy*deriv2.d2ydthetadphi;
    dest.dphi2 =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdphi +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydphi +
        srcGrad.dx*deriv2.d2xdphi2 + srcGrad.dy*deriv2.d2ydphi2;
    return dest;
}

coord::HessSph ref_toHess_Sph_from_Cyl(const coord::GradCyl& srcGrad, const coord::HessCyl& srcHess,
    const coord::PosDerivT<coord::Sph, coord::Cyl>& deriv, const coord::PosDeriv2T<coord::Sph, coord::Cyl>& deriv2) {
    coord::HessSph dest;
    dest.dr2 =
        (srcHess.dR2 *deriv.dRdr + srcHess.dRdz*deriv.dzdr) * deriv.dRdr +
        (srcHess.dRdz*deriv.dRdr + srcHess.dz2 *deriv.dzdr) * deriv.dzdr;
    dest.drdtheta =
        (srcHess.dR2 *deriv.dRdtheta + srcHess.dRdz*deriv.dzdtheta) * deriv.dRdr +
        (srcHess.dRdz*deriv.dRdtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdr +
        srcGrad.dR*deriv2.d2Rdrdtheta + srcGrad.dz*deriv2.d2zdrdtheta;
    dest.dtheta2 =
        (srcHess.dR2 *deriv.dRdtheta + srcHess.dRdz*deriv.dzdtheta) * deriv.dRdtheta +
        (srcHess.dRdz*deriv.dRdtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdtheta +
        srcGrad.dR*deriv2.d2Rdtheta2 + srcGrad.dz*deriv2.d2zdtheta2;
    dest.drdphi     = srcHess.dRdphi*deriv.dRdr     + srcHess.dzdphi*deriv.dzdr;
    dest.dthetadphi = srcHess.dRdphi*deriv.dRdtheta + srcHess.dzdphi*deriv.dzdtheta;
    dest.dphi2      = srcHess.dphi2;
    return dest;
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

    // =====================================================================
    // Tier 0 Legendre helpers: legendrePmm / sphHarmArray became
    // AGAMA_DEVICE_INLINE, and legendrePmm's tabulated normalization
    // constants were extended from m<=16 to m<=LEGENDRE_MMAX (=32) so that
    // the device never needs the GSL-backed factorial fallback.
    //
    // (a) THE CPU-INVARIANCE GATE. Every m <= LEGENDRE_MMAX must return
    //     BIT-FOR-BIT what it returned before this change, for two different
    //     reasons over two ranges:
    //       m = 0..16  -- upstream's hand-tabulated literals. These are NOT
    //         equal to the closed form (they were printed to 16 significant
    //         digits, so they sit within 1 ULP of it), which is exactly why
    //         they must be checked against the literals themselves, restated
    //         here independently of the header.
    //       m = 17..32 -- our extension, generated FROM the closed form, so
    //         here bit-for-bit equality with the closed form is the gate.
    //         Before this change these m fell through to the same closed
    //         form at runtime, so CPU output is unchanged.
    //     A fat-fingered table digit, or a GSL factorial/dfactorial change,
    //     fails here loudly instead of silently moving Multipole coefficients.
    // =====================================================================
    bool ok_legtab = true;
    {
        // upstream's literals, restated so the header is not its own reference
        const int UPMMAX = 16;
        const double UP_PREFACT[UPMMAX+1] = { 0.2820947917738782,
            0.3454941494713355,    0.1287580673410632,    0.02781492157551894,   0.004214597070904597,
            0.0004911451888263050, 4.647273819914057e-05, 3.700296470718545e-06, 2.542785532478802e-07,
            1.536743406172476e-08, 8.287860012085477e-10, 4.035298721198747e-11, 1.790656309174350e-12,
            7.299068453727266e-14, 2.751209457796109e-15, 9.643748535232993e-17, 3.159120301003413e-18 };
        const double UP_COEF[UPMMAX+1] = { 0.2820947917738782,
            -0.3454941494713355, 0.3862742020231896, -0.4172238236327841, 0.4425326924449826,
            -0.4641322034408582, 0.4830841135800662, -0.5000395635705506, 0.5154289843972843,
            -0.5295529414924496, 0.5426302919442215, -0.5548257538066191, 0.5662666637421912,
            -0.5770536647012670, 0.5872677968601020, -0.5969753602424046, 0.6062313441538353 };

        for(int m = 0; m <= math::LEGENDRE_MMAX; ++m) {
            // an off-axis theta so that neither sin nor cos is degenerate
            const double theta = 0.7, ct = std::cos(theta), st = std::sin(theta);
            double prefact = 0, value = 0, der = 0, der2 = 0;
            math::legendrePmm(m, ct, st, prefact, &value, &der, &der2);

            // reference constants: upstream's literals where they exist, else the
            // closed form that legendrePmm used to fall through to at runtime
            const double pref_ref = m <= UPMMAX ? UP_PREFACT[m] :
                0.5/M_SQRTPI * std::sqrt( (2*m+1) / math::factorial(2*m) );
            const double coef_ref = m <= UPMMAX ? UP_COEF[m] :
                pref_ref * math::dfactorial(2*m-1) * (m%2 == 1 ? -1 : 1);
            // rebuild the output with the same operations in the same order,
            // so equality must be exact
            const double val_ref = m == 0 ? pref_ref : m == 1 ? -st * pref_ref :
                coef_ref * math::powT(st, m-2) * pow_2(st);

            if(prefact != pref_ref) {
                ok_legtab = false;
                std::printf("[T0]    legendrePmm prefact[%d] MISMATCH: got %.17g  want %.17g\n",
                    m, prefact, pref_ref);
            }
            if(value != val_ref) {
                ok_legtab = false;
                std::printf("[T0]    legendrePmm Pmm[%d] MISMATCH: got %.17g  want %.17g\n",
                    m, value, val_ref);
            }
        }
        std::printf("[T0]    legendrePmm m=0..%d bit-for-bit vs reference constants "
            "(0..%d upstream literals, %d..%d closed form) -> %s\n",
            math::LEGENDRE_MMAX, UPMMAX, UPMMAX+1, math::LEGENDRE_MMAX,
            ok_legtab ? "OK" : "FAIL");
    }

    // (b) math::pow(double,int) must be bit-for-bit its device-callable
    //     single source math::powT<double>(double,int) -- the .cpp now just
    //     forwards, so this guards the forwarding (and the overload pick:
    //     an int exponent must NOT land in powT(T,T)).
    bool ok_powint = true;
    {
        const double xs[] = { 1.5, 0.25, -3.0, 1e-4, 7.125, -0.5 };
        for(std::size_t i = 0; i < sizeof(xs)/sizeof(xs[0]); ++i)
            for(int n = -20; n <= 20; ++n)
                if(math::pow(xs[i], n) != math::powT(xs[i], n))
                    ok_powint = false;
        std::printf("[T0]    math::pow(double,int) == math::powT<double>(x,int) "
            "(6 bases x n=-20..20, bit-for-bit) -> %s\n", ok_powint ? "OK" : "FAIL");
    }

    // (c) Serial reference for sphHarmArray, compared against Cuda further down.
    //     Sweep tau over the whole range including the |tau|->1 asymptotic
    //     branches (theta -> 0 / pi) where the derivative formulas switch.
    const int    LEG_LMAX = 12;
    const int    LEG_NM   = 5;           // orders m = 0,1,2,3,LEG_LMAX
    const int    LEG_NTAU = 48;
    const int    LEG_MS[LEG_NM] = { 0, 1, 2, 3, LEG_LMAX };
    const std::size_t LEG_STRIDE = LEG_LMAX + 1;  // room for l = m..lmax
    // layout: [itau][im][3 quantities][LEG_STRIDE]
    const std::size_t LEG_N = (std::size_t)LEG_NTAU * LEG_NM * 3 * LEG_STRIDE;
    std::vector<double> leg_s(LEG_N, 0.0);
    for(int it = 0; it < LEG_NTAU; ++it) {
        // tau in (-1, 1); the endpoints are approached to 1e-12 to exercise
        // the small-sin(theta) asymptotic branches
        const double tau = -1.0 + 1e-12 + (2.0 - 2e-12) * (it + 0.5) / LEG_NTAU;
        for(int im = 0; im < LEG_NM; ++im) {
            double* base = &leg_s[(((std::size_t)it * LEG_NM + im) * 3) * LEG_STRIDE];
            math::sphHarmArray(LEG_LMAX, LEG_MS[im], tau,
                base, base + LEG_STRIDE, base + 2 * LEG_STRIDE);
        }
    }

    // =====================================================================
    // Tier 2 commit 1: coord::toGrad/toHess Car/Cyl/Sph triangle (6 toGrad +
    // 6 toHess) promoted from coord.cpp to coord.h as AGAMA_DEVICE_INLINE.
    // Motivation: every branch of Multipole::evalCyl routes through
    // toGrad<Sph,Cyl>/toHess<Sph,Cyl> (via transformDerivsSphToCyl in
    // potential_multipole.cpp), so a device kernel could not evaluate a
    // Multipole before this. ProlSph/Axi specializations stay host-only
    // (Stackel action finder / focal-distance finder, both CPU-forever).
    //
    // THE CPU-INVARIANCE GATE: a deterministic pseudo-random sweep comparing
    // coord::toGrad/toHess (the header, AGAMA_DEVICE_INLINE version under
    // test) against the ref_toGrad_*/ref_toHess_* functions defined above
    // main() -- verbatim, frozen copies of the PRE-PROMOTION coord.cpp
    // bodies, never to be "cleaned up" (see the comment on that block).
    //
    // A hard-coded-literal version of this test was tried FIRST and rejected:
    // it pinned 24 literal doubles generated by a separate small TU
    // (local_notes/coord_promote_check/gen_refs.cpp), and 2 of them differed
    // in their last bit from what THIS file's own compilation produced, even
    // under byte-identical compiler flags -- the surrounding code in this
    // much larger TU shifted an FMA-contraction decision at the call site
    // (same class of effect as the fork-vs-upstream BFE FMA finding in
    // findings.md, now visible WITHIN one binary rather than across two
    // installs). A literal pinned outside this TU cannot tell a real
    // regression apart from that kind of harmless codegen drift, and any fix
    // available to a future contributor ("update the literal to match got")
    // would silently paper over a real change just as easily as a spurious
    // one -- a gate that can't tell those apart isn't a gate.
    //
    // This version is immune to that: ref_* and coord::toGrad/toHess are
    // both compiled in THIS TU, so if the compiler merely reschedules or
    // contracts differently, both sides move together and stay equal. Only
    // an actual edit to coord.h's arithmetic (or a diverging edit to the
    // frozen ref_* bodies, which must never happen) can make them disagree.
    // Same input distribution as the original inertness proof
    // (local_notes/coord_promote_check/inertness_check.cpp): a mix of
    // ordinary random values, exact zero, negative zero, and huge/tiny
    // magnitudes, over a fixed-seed std::mt19937_64 so the sweep is
    // reproducible run to run.
    // =====================================================================
    bool ok_coordderiv = true;
    long coordderiv_compared = 0, coordderiv_mismatches = 0;
    {
        // bit-for-bit, not `==` -- the latter treats -0.0 == 0.0, which would
        // silently pass a sign-of-zero regression.
        auto bitEqual = [](double a, double b) {
            uint64_t ia, ib;
            std::memcpy(&ia, &a, 8);
            std::memcpy(&ib, &b, 8);
            return ia == ib;
        };
        auto chk = [&](double got, double want, const char* name) {
            ++coordderiv_compared;
            if (!bitEqual(got, want)) {
                ++coordderiv_mismatches;
                ok_coordderiv = false;
                std::printf("[T2]    %s MISMATCH: got %.17g want %.17g\n", name, got, want);
            }
        };
        // fixed-seed generator: ordinary random values, exact zero, negative
        // zero, and huge/tiny magnitudes -- same recipe as the original
        // inertness harness's nasty(), reseeded here for reproducibility.
        std::mt19937_64 cd_rng(20260725);
        std::uniform_real_distribution<double> cd_normal(-1.0, 1.0);
        std::uniform_real_distribution<double> cd_exp(-300.0, 300.0);
        auto nasty = [&]() -> double {
            switch (cd_rng() % 8) {
                case 0: return 0.0;
                case 1: return -0.0;
                case 2: return cd_normal(cd_rng) * std::pow(10.0, cd_exp(cd_rng));
                default: return cd_normal(cd_rng);
            }
        };

        const int SWEEP_N = 5000;
        for (int i = 0; i < SWEEP_N; ++i) {
            // toGrad<Car,Cyl>
            {
                coord::GradCyl src = {nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Car,coord::Cyl> d = {nasty(), nasty(), nasty(), nasty()};
                coord::GradCar r = coord::toGrad(src, d);
                coord::GradCar rr = ref_toGrad_Car_from_Cyl(src, d);
                chk(r.dx, rr.dx, "toGrad<Car,Cyl> dx"); chk(r.dy, rr.dy, "toGrad<Car,Cyl> dy"); chk(r.dz, rr.dz, "toGrad<Car,Cyl> dz");
            }
            // toGrad<Car,Sph>
            {
                coord::GradSph src = {nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Car,coord::Sph> d = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::GradCar r = coord::toGrad(src, d);
                coord::GradCar rr = ref_toGrad_Car_from_Sph(src, d);
                chk(r.dx, rr.dx, "toGrad<Car,Sph> dx"); chk(r.dy, rr.dy, "toGrad<Car,Sph> dy"); chk(r.dz, rr.dz, "toGrad<Car,Sph> dz");
            }
            // toGrad<Cyl,Car>
            {
                coord::GradCar src = {nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Cyl,coord::Car> d = {nasty(), nasty(), nasty(), nasty()};
                coord::GradCyl r = coord::toGrad(src, d);
                coord::GradCyl rr = ref_toGrad_Cyl_from_Car(src, d);
                chk(r.dR, rr.dR, "toGrad<Cyl,Car> dR"); chk(r.dz, rr.dz, "toGrad<Cyl,Car> dz"); chk(r.dphi, rr.dphi, "toGrad<Cyl,Car> dphi");
            }
            // toGrad<Cyl,Sph>
            {
                coord::GradSph src = {nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Cyl,coord::Sph> d = {nasty(), nasty(), nasty(), nasty()};
                coord::GradCyl r = coord::toGrad(src, d);
                coord::GradCyl rr = ref_toGrad_Cyl_from_Sph(src, d);
                chk(r.dR, rr.dR, "toGrad<Cyl,Sph> dR"); chk(r.dz, rr.dz, "toGrad<Cyl,Sph> dz"); chk(r.dphi, rr.dphi, "toGrad<Cyl,Sph> dphi");
            }
            // toGrad<Sph,Car>
            {
                coord::GradCar src = {nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Sph,coord::Car> d = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::GradSph r = coord::toGrad(src, d);
                coord::GradSph rr = ref_toGrad_Sph_from_Car(src, d);
                chk(r.dr, rr.dr, "toGrad<Sph,Car> dr"); chk(r.dtheta, rr.dtheta, "toGrad<Sph,Car> dtheta"); chk(r.dphi, rr.dphi, "toGrad<Sph,Car> dphi");
            }
            // toGrad<Sph,Cyl>  -- the one Multipole::evalCyl actually calls
            {
                coord::GradCyl src = {nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Sph,coord::Cyl> d = {nasty(), nasty(), nasty(), nasty()};
                coord::GradSph r = coord::toGrad(src, d);
                coord::GradSph rr = ref_toGrad_Sph_from_Cyl(src, d);
                chk(r.dr, rr.dr, "toGrad<Sph,Cyl> dr"); chk(r.dtheta, rr.dtheta, "toGrad<Sph,Cyl> dtheta"); chk(r.dphi, rr.dphi, "toGrad<Sph,Cyl> dphi");
            }
            // toHess<Car,Cyl>
            {
                coord::GradCyl sg = {nasty(), nasty(), nasty()};
                coord::HessCyl sh = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Car,coord::Cyl> d = {nasty(), nasty(), nasty(), nasty()};
                coord::PosDeriv2T<coord::Car,coord::Cyl> d2 = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::HessCar r = coord::toHess(sg, sh, d, d2);
                coord::HessCar rr = ref_toHess_Car_from_Cyl(sg, sh, d, d2);
                chk(r.dx2, rr.dx2, "toHess<Car,Cyl> dx2"); chk(r.dxdy, rr.dxdy, "toHess<Car,Cyl> dxdy"); chk(r.dy2, rr.dy2, "toHess<Car,Cyl> dy2");
                chk(r.dxdz, rr.dxdz, "toHess<Car,Cyl> dxdz"); chk(r.dydz, rr.dydz, "toHess<Car,Cyl> dydz"); chk(r.dz2, rr.dz2, "toHess<Car,Cyl> dz2");
            }
            // toHess<Car,Sph>
            {
                coord::GradSph sg = {nasty(), nasty(), nasty()};
                coord::HessSph sh = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Car,coord::Sph> d = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDeriv2T<coord::Car,coord::Sph> d2 = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty(),
                                                                nasty(), nasty(), nasty(), nasty(), nasty(), nasty(),
                                                                nasty(), nasty(), nasty()};
                coord::HessCar r = coord::toHess(sg, sh, d, d2);
                coord::HessCar rr = ref_toHess_Car_from_Sph(sg, sh, d, d2);
                chk(r.dx2, rr.dx2, "toHess<Car,Sph> dx2"); chk(r.dxdy, rr.dxdy, "toHess<Car,Sph> dxdy"); chk(r.dxdz, rr.dxdz, "toHess<Car,Sph> dxdz");
                chk(r.dy2, rr.dy2, "toHess<Car,Sph> dy2"); chk(r.dydz, rr.dydz, "toHess<Car,Sph> dydz"); chk(r.dz2, rr.dz2, "toHess<Car,Sph> dz2");
            }
            // toHess<Cyl,Car>
            {
                coord::GradCar sg = {nasty(), nasty(), nasty()};
                coord::HessCar sh = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Cyl,coord::Car> d = {nasty(), nasty(), nasty(), nasty()};
                coord::PosDeriv2T<coord::Cyl,coord::Car> d2 = {nasty(), nasty(), nasty(), nasty()};
                coord::HessCyl r = coord::toHess(sg, sh, d, d2);
                coord::HessCyl rr = ref_toHess_Cyl_from_Car(sg, sh, d, d2);
                chk(r.dR2, rr.dR2, "toHess<Cyl,Car> dR2"); chk(r.dRdz, rr.dRdz, "toHess<Cyl,Car> dRdz"); chk(r.dRdphi, rr.dRdphi, "toHess<Cyl,Car> dRdphi");
                chk(r.dz2, rr.dz2, "toHess<Cyl,Car> dz2"); chk(r.dzdphi, rr.dzdphi, "toHess<Cyl,Car> dzdphi"); chk(r.dphi2, rr.dphi2, "toHess<Cyl,Car> dphi2");
            }
            // toHess<Cyl,Sph>
            {
                coord::GradSph sg = {nasty(), nasty(), nasty()};
                coord::HessSph sh = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Cyl,coord::Sph> d = {nasty(), nasty(), nasty(), nasty()};
                coord::PosDeriv2T<coord::Cyl,coord::Sph> d2 = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::HessCyl r = coord::toHess(sg, sh, d, d2);
                coord::HessCyl rr = ref_toHess_Cyl_from_Sph(sg, sh, d, d2);
                chk(r.dR2, rr.dR2, "toHess<Cyl,Sph> dR2"); chk(r.dRdz, rr.dRdz, "toHess<Cyl,Sph> dRdz"); chk(r.dRdphi, rr.dRdphi, "toHess<Cyl,Sph> dRdphi");
                chk(r.dz2, rr.dz2, "toHess<Cyl,Sph> dz2"); chk(r.dzdphi, rr.dzdphi, "toHess<Cyl,Sph> dzdphi"); chk(r.dphi2, rr.dphi2, "toHess<Cyl,Sph> dphi2");
            }
            // toHess<Sph,Car>
            {
                coord::GradCar sg = {nasty(), nasty(), nasty()};
                coord::HessCar sh = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Sph,coord::Car> d = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDeriv2T<coord::Sph,coord::Car> d2 = {nasty(), nasty(), nasty(), nasty(), nasty(),
                                                                nasty(), nasty(), nasty(), nasty(), nasty(),
                                                                nasty(), nasty()};
                coord::HessSph r = coord::toHess(sg, sh, d, d2);
                coord::HessSph rr = ref_toHess_Sph_from_Car(sg, sh, d, d2);
                chk(r.dr2, rr.dr2, "toHess<Sph,Car> dr2"); chk(r.drdtheta, rr.drdtheta, "toHess<Sph,Car> drdtheta"); chk(r.drdphi, rr.drdphi, "toHess<Sph,Car> drdphi");
                chk(r.dtheta2, rr.dtheta2, "toHess<Sph,Car> dtheta2"); chk(r.dthetadphi, rr.dthetadphi, "toHess<Sph,Car> dthetadphi"); chk(r.dphi2, rr.dphi2, "toHess<Sph,Car> dphi2");
            }
            // toHess<Sph,Cyl>  -- the one Multipole::evalCyl actually calls
            {
                coord::GradCyl sg = {nasty(), nasty(), nasty()};
                coord::HessCyl sh = {nasty(), nasty(), nasty(), nasty(), nasty(), nasty()};
                coord::PosDerivT<coord::Sph,coord::Cyl> d = {nasty(), nasty(), nasty(), nasty()};
                coord::PosDeriv2T<coord::Sph,coord::Cyl> d2 = {nasty(), nasty(), nasty(), nasty()};
                coord::HessSph r = coord::toHess(sg, sh, d, d2);
                coord::HessSph rr = ref_toHess_Sph_from_Cyl(sg, sh, d, d2);
                chk(r.dr2, rr.dr2, "toHess<Sph,Cyl> dr2"); chk(r.drdtheta, rr.drdtheta, "toHess<Sph,Cyl> drdtheta"); chk(r.drdphi, rr.drdphi, "toHess<Sph,Cyl> drdphi");
                chk(r.dtheta2, rr.dtheta2, "toHess<Sph,Cyl> dtheta2"); chk(r.dthetadphi, rr.dthetadphi, "toHess<Sph,Cyl> dthetadphi"); chk(r.dphi2, rr.dphi2, "toHess<Sph,Cyl> dphi2");
            }
        }
        std::printf("[T2]    toGrad/toHess Car/Cyl/Sph triangle (12 fns) vs frozen "
            "pre-promotion ref_* bodies, same-TU, %d-iter fixed-seed sweep incl. "
            "exact-zero/-0.0/huge/tiny: %ld values compared, %ld mismatches -> %s\n",
            SWEEP_N, coordderiv_compared, coordderiv_mismatches, ok_coordderiv ? "OK" : "FAIL");
    }

    // =====================================================================
    // Tier 2 (Multipole evaluator prerequisite): math::SphHarmIndicesPod, the
    // device-callable POD mirror of math::SphHarmIndices.
    //
    // SphHarmIndices holds a std::vector<int> lmin_arr, so it cannot cross into
    // a kernel by value. That array is a closed form of (lmax, mmax, sym) whose
    // only inputs are the coord.h symmetry predicates -- all already
    // AGAMA_DEVICE_INLINE -- so the POD restates it in 4 ints and the Multipole
    // descriptor needs no device pointer for the indexing scheme.
    //
    // A restatement is only safe if it is EXACT, hence this exhaustive gate
    // rather than a spot check. The trap it guards: the SphHarmIndices ctor
    // MUTATES sym after validating its arguments (mmax==0 implies z-rotation +
    // x/y-reflection; lmax==0 implies full rotation + z/xyz-reflection), and
    // lmin() reads the mutated value -- which is why the POD may only be built
    // via SphHarmIndices::pod(), never reconstructed from a caller's requested
    // symmetry. The lmax==0 / mmax==0 rows below are what pin that.
    // =====================================================================
    bool ok_shipod = true;
    long shipod_compared = 0, shipod_mismatches = 0;
    {
        static const int SYMS[] = {
            coord::ST_NONE, coord::ST_XREFLECTION, coord::ST_YREFLECTION, coord::ST_ZREFLECTION,
            coord::ST_REFLECTION, coord::ST_ZROTATION, coord::ST_ROTATION,
            coord::ST_TRIAXIAL, coord::ST_BISYMMETRIC, coord::ST_AXISYMMETRIC, coord::ST_SPHERICAL,
            coord::ST_XREFLECTION | coord::ST_YREFLECTION,
            coord::ST_XREFLECTION | coord::ST_ZREFLECTION,
            coord::ST_YREFLECTION | coord::ST_ZREFLECTION,
            coord::ST_REFLECTION  | coord::ST_XREFLECTION,
            coord::ST_REFLECTION  | coord::ST_ZREFLECTION,
            coord::ST_ZROTATION   | coord::ST_ZREFLECTION };
        const int NSYM = (int)(sizeof(SYMS)/sizeof(SYMS[0]));
        // lmax sweep runs past math::LEGENDRE_MMAX (=32) on purpose: the device
        // descriptor builder must reject mmax > LEGENDRE_MMAX, but the INDEXING
        // scheme itself has no such cap and must stay exact above it.
        const int POD_LMAX = 34;
        for(int is=0; is<NSYM && ok_shipod; is++) {
            for(int lmax=0; lmax<=POD_LMAX; lmax++) {
                for(int mmax=0; mmax<=lmax; mmax++) {
                    coord::SymmetryType sym = static_cast<coord::SymmetryType>(SYMS[is]);
                    math::SphHarmIndices    ind(lmax, mmax, sym);
                    math::SphHarmIndicesPod pod = ind.pod();
                    const bool scalars_ok =
                        pod.lmax == ind.lmax && pod.mmax == ind.mmax &&
                        pod.step == ind.step && pod.size() == (int)ind.size() &&
                        pod.mmin() == ind.mmin();
                    shipod_compared += 5;
                    if(!scalars_ok) {
                        ++shipod_mismatches; ok_shipod = false;
                        std::printf("[T2]    SphHarmIndicesPod scalar MISMATCH sym=%d lmax=%d mmax=%d\n",
                            SYMS[is], lmax, mmax);
                    }
                    // probe m beyond +-mmax as well: both must report lmax+1 there
                    for(int m=-mmax-3; m<=mmax+3; m++) {
                        ++shipod_compared;
                        if(pod.lmin(m) != ind.lmin(m)) {
                            ++shipod_mismatches; ok_shipod = false;
                            std::printf("[T2]    SphHarmIndicesPod lmin MISMATCH sym=%d lmax=%d "
                                "mmax=%d m=%d: pod=%d ind=%d\n", SYMS[is], lmax, mmax, m,
                                pod.lmin(m), ind.lmin(m));
                        }
                    }
                    for(int l=0; l<=lmax; l++)
                        for(int m=-l; m<=l; m++) {
                            ++shipod_compared;
                            if(math::SphHarmIndicesPod::index(l,m) !=
                               (int)math::SphHarmIndices::index(l,m)) {
                                ++shipod_mismatches; ok_shipod = false;
                            }
                        }
                }
            }
        }
        std::printf("[T2]    SphHarmIndicesPod vs SphHarmIndices, %d symmetries x lmax<=%d, all m "
            "(incl. out-of-range) + every index(l,m): %ld values compared, %ld mismatches -> %s\n",
            NSYM, POD_LMAX, shipod_compared, shipod_mismatches, ok_shipod ? "OK" : "FAIL");
    }

    // Tier 2 (Multipole evaluator prerequisite): device-callable quintic-spline
    // raw evaluators, evalQuinticSplineRaw (1d) and evalQuinticSpline2dRaw (2d).
    // These are the stateless cores that QuinticSpline::evalDeriv and
    // QuinticSpline2d::evalDeriv now wrap (one source of math), and are what a
    // future MultipoleInterp2d device kernel will call directly per azimuthal
    // harmonic m, in (ln r, tau).
    //
    // (a) CPU-side bit-for-bit gate: the raw evaluator must reproduce the
    //     class evalDeriv() exactly (bit patterns compared via memcpy, so a
    //     literal NaN in the out-of-grid path compares equal to itself instead
    //     of tripping the usual NaN != NaN pitfall of operator==).
    // (b) fp32 instantiation smoke check (CLAUDE.md recipe item 7): both raw
    //     evaluators must actually instantiate for NumT=float and produce a
    //     finite result HERE, in a CPU-only build -- a bare (non-NumT-wrapped)
    //     double literal surviving in evalQuinticSplines's arithmetic would
    //     silently compile (float<-double narrowing is legal) rather than
    //     erroring, so a value-level finiteness check is the only thing that
    //     would catch a broken transcription at this stage; the true
    //     "computed in float, not double-then-truncated" property can only be
    //     confirmed by disassembly or an nvcc build. The bit-for-bit NumT=double
    //     comparison in (a) is the one that actually gates correctness.
    //
    // Data declared at this scope (not nested in a block) because the
    // HAVE_CUDA section far below reuses the same 1d/2d grids and node data
    // for its Serial-vs-Cuda parity check -- same pattern as leg_s/LEG_LMAX
    // above.
    // =====================================================================
    std::vector<double> qx = { -3.0, -2.7, -1.0, -0.2, 0.0, 0.3, 1.5, 2.0, 4.0 };
    std::vector<double> qf(qx.size()), qd(qx.size()), qd2(qx.size());
    for(std::size_t i = 0; i < qx.size(); ++i) {
        qf [i] = std::sin(0.6*qx[i]) + 0.2*qx[i]*qx[i];
        qd [i] = 0.6*std::cos(0.6*qx[i]) + 0.4*qx[i];
        qd2[i] = -0.36*std::sin(0.6*qx[i]) + 0.4;
    }
    std::vector<double> gx = { -2.0, -1.3, -0.4, 0.0, 0.5, 1.7, 2.5 };
    std::vector<double> gy = { -1.5, -1.0, -0.2, 0.1, 0.9, 2.0 };
    math::Matrix<double> f2(gx.size(), gy.size()), fx2(gx.size(), gy.size()),
        fy2(gx.size(), gy.size()), fxy2(gx.size(), gy.size());
    for(std::size_t i = 0; i < gx.size(); ++i)
        for(std::size_t j = 0; j < gy.size(); ++j) {
            f2  (i,j) = std::sin(0.4*gx[i]) * std::cos(0.3*gy[j]);
            fx2 (i,j) =  0.4*std::cos(0.4*gx[i]) * std::cos(0.3*gy[j]);
            fy2 (i,j) = -0.3*std::sin(0.4*gx[i]) * std::sin(0.3*gy[j]);
            fxy2(i,j) = -0.12*std::cos(0.4*gx[i]) * std::sin(0.3*gy[j]);
        }

    auto quinticBitsEq = [](double a, double b) {
        std::uint64_t ba, bb;
        std::memcpy(&ba, &a, 8);
        std::memcpy(&bb, &b, 8);
        return ba == bb;
    };

    bool ok_quintic_raw = true;
    {
        // 1d: QuinticSpline Hermite variant (values + 1st + 2nd derivs at all nodes)
        // on a strongly non-uniform grid -- exercises all three input arrays.
        math::QuinticSpline qs(qx, qf, qd, qd2);
        const double probe1d[] = { qx.front(), qx.back(), qx[3], -0.2, 0.15, -10.0, 100.0, NAN };
        for(double x : probe1d) {
            double v0, d0, d20, d30, v1, d1, d21, d31;
            qs.evalDeriv(x, &v0, &d0, &d20, &d30);
            math::evalQuinticSplineRaw(x, qs.xvalues().data(), qs.fvalues().data(),
                qs.fderivs().data(), qs.fderivs2().data(), (int)qx.size(),
                &v1, &d1, &d21, &d31);
            if(!(quinticBitsEq(v0,v1) && quinticBitsEq(d0,d1) &&
                 quinticBitsEq(d20,d21) && quinticBitsEq(d30,d31))) {
                ok_quintic_raw = false;
                std::printf("[CPU]   evalQuinticSplineRaw MISMATCH at x=%.6g\n", x);
            }
        }

        // 2d: with and without the mixed derivative, including exact corner/edge
        // coincidences (x==xupp / y==yupp) -- exactly what the f_offset trick keys off.
        math::QuinticSpline2d sp_nomix(gx, gy, f2, fx2, fy2);
        math::QuinticSpline2d sp_mix  (gx, gy, f2, fx2, fy2, fxy2);
        const math::QuinticSpline2d* splines2d[] = { &sp_nomix, &sp_mix };
        struct Pt2 { double x, y; };
        const Pt2 probe2d[] = {
            { gx.front(), gy.front() }, { gx.back(), gy.back() },   // both corners
            { gx[2], gy.back() }, { gx.back(), gy[3] },             // one-sided coincidence
            { -0.9, 0.3 }, { 0.0, 0.0 }, { -100.0, -100.0 }, { 100.0, 100.0 }
        };
        for(int s = 0; s < 2; ++s) {
            const math::QuinticSpline2d& sp2d = *splines2d[s];
            for(const Pt2& pt : probe2d) {
                double z0,zx0,zy0,zxx0,zxy0,zyy0, z1,zx1,zy1,zxx1,zxy1,zyy1;
                sp2d.evalDeriv(pt.x, pt.y, &z0,&zx0,&zy0,&zxx0,&zxy0,&zyy0);
                math::evalQuinticSpline2dRaw(pt.x, pt.y,
                    sp2d.xvalues().data(), sp2d.yvalues().data(),
                    (int)gx.size(), (int)gy.size(),
                    sp2d.fvalues().data(), sp2d.dfdx().data(), sp2d.dfdy().data(),
                    sp2d.d2fdx2().data(), sp2d.d2fdxdy().data(), sp2d.d2fdy2().data(),
                    sp2d.d3fdx2dy().data(), sp2d.d3fdxdy2().data(), sp2d.d4fdx2dy2().data(),
                    &z1,&zx1,&zy1,&zxx1,&zxy1,&zyy1);
                if(!(quinticBitsEq(z0,z1) && quinticBitsEq(zx0,zx1) && quinticBitsEq(zy0,zy1) &&
                     quinticBitsEq(zxx0,zxx1) && quinticBitsEq(zxy0,zxy1) && quinticBitsEq(zyy0,zyy1))) {
                    ok_quintic_raw = false;
                    std::printf("[CPU]   evalQuinticSpline2dRaw MISMATCH at (%.6g,%.6g)\n", pt.x, pt.y);
                }
            }
        }
        std::printf("[CPU]   evalQuinticSplineRaw / evalQuinticSpline2dRaw bit-for-bit vs "
            "QuinticSpline/QuinticSpline2d::evalDeriv (on-node/interior/endpoint/out-of-grid/NaN) -> %s\n",
            ok_quintic_raw ? "OK" : "FAIL");

        // fp32 instantiation smoke check
        std::vector<float> qxf(qx.begin(), qx.end()), qff(qf.begin(), qf.end()),
            qdf(qd.begin(), qd.end()), qd2f(qd2.begin(), qd2.end());
        float v32, d32, d232, d332;
        math::evalQuinticSplineRaw<float>(0.15f, qxf.data(), qff.data(), qdf.data(), qd2f.data(),
            (int)qxf.size(), &v32, &d32, &d232, &d332);

        const std::size_t nxg = gx.size(), nyg = gy.size();
        std::vector<float> gxf(gx.begin(), gx.end()), gyf(gy.begin(), gy.end());
        std::vector<float> vf(nxg*nyg), vfx(nxg*nyg), vfy(nxg*nyg), vfxx(nxg*nyg, 0.f),
            vfxy(nxg*nyg), vfyy(nxg*nyg, 0.f), vfxxy(nxg*nyg, 0.f), vfxyy(nxg*nyg, 0.f),
            vfxxyy(nxg*nyg, 0.f);
        for(std::size_t i = 0; i < nxg; ++i)
            for(std::size_t j = 0; j < nyg; ++j) {
                const std::size_t idx = i*nyg + j;
                vf  [idx] = static_cast<float>(f2  (i,j));
                vfx [idx] = static_cast<float>(fx2 (i,j));
                vfy [idx] = static_cast<float>(fy2 (i,j));
                vfxy[idx] = static_cast<float>(fxy2(i,j));
            }
        float z32, zx32, zy32, zxx32, zxy32, zyy32;
        math::evalQuinticSpline2dRaw<float>(0.1f, 0.2f, gxf.data(), gyf.data(),
            (int)nxg, (int)nyg, vf.data(), vfx.data(), vfy.data(), vfxx.data(), vfxy.data(),
            vfyy.data(), vfxxy.data(), vfxyy.data(), vfxxyy.data(),
            &z32, &zx32, &zy32, &zxx32, &zxy32, &zyy32);
        const bool ok_fp32_finite = std::isfinite(v32) && std::isfinite(z32);
        std::printf("[CPU]   evalQuinticSplineRaw<float>/evalQuinticSpline2dRaw<float> "
            "instantiate and evaluate finite -> %s\n", ok_fp32_finite ? "OK" : "FAIL");
        ok_quintic_raw = ok_quintic_raw && ok_fp32_finite;
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

        // ----- (d) device-resident output: integrateOrbitsGPUDevice -----
        // The whole point of the device-output entry point is that it changes WHERE
        // the trajectory lands and nothing else, so the gate is exact equality with
        // the host-output result, not a tolerance. Anything less would let a real
        // divergence (a missed spline upload, a different accuracy clamp, a dropped
        // NAN prefill) hide inside a "close enough" threshold.
        {
            bool ok_dev = true;
            auto check_device_output = [&](auto tag, const char* tname,
                const std::vector<decltype(tag)>& traj_host, double accuracy)
            {
                typedef decltype(tag) T;
                const std::size_t n6 = NORB * TRAJ * 6;
                T* d_traj = NULL;
                if(cudaMalloc(&d_traj, n6 * sizeof(T)) != cudaSuccess) {
                    std::printf("[CUDA]  device-resident orbits %s: cudaMalloc FAILED\n", tname);
                    ok_dev = false;
                    return;
                }
                int rc_d = orbit::integrateOrbitsGPUDevice<T>(pot, NORB, ic.data(),
                    times.data(), TRAJ, accuracy, 100000000, d_traj, /*output_stream*/ 0);
                std::vector<T> back(n6);
                cudaMemcpy(back.data(), d_traj, n6 * sizeof(T), cudaMemcpyDeviceToHost);
                // bit-for-bit, including NAN slots: compare the raw bit patterns so a
                // NAN never compares unequal to itself and mask a real mismatch
                std::size_t ndiff = 0;
                for(std::size_t j = 0; j < n6; j++) {
                    T a = back[j], b = traj_host[j];
                    if(std::memcmp(&a, &b, sizeof(T)) != 0)
                        ndiff++;
                }
                bool ok = rc_d == 0 && ndiff == 0;
                std::printf("[CUDA]  device-resident orbits %s vs host-resident: "
                    "%zu/%zu values differ -> %s\n", tname, ndiff, n6, ok ? "OK (bitwise)" : "FAIL");
                ok_dev = ok_dev && ok;

                // scaleTrajectoryGPUDevice must reproduce the host unit conversion
                // exactly (promote to double, divide, round once) -- see orbit_gpu.h.
                const double LU = 1.234567890123, VU = 0.98765432109;
                int rc_s2 = orbit::scaleTrajectoryGPUDevice<T>(NORB * TRAJ, d_traj, LU, VU);
                std::vector<T> scaled(n6);
                cudaMemcpy(scaled.data(), d_traj, n6 * sizeof(T), cudaMemcpyDeviceToHost);
                std::size_t nsdiff = 0;
                for(std::size_t i = 0; i < NORB * TRAJ; i++) {
                    for(int k = 0; k < 6; k++) {
                        // exactly what the host directToDest loop computes
                        T want = T(double(traj_host[i*6+k]) / (k < 3 ? LU : VU));
                        T got  = scaled[i*6+k];
                        if(std::memcmp(&got, &want, sizeof(T)) != 0)
                            nsdiff++;
                    }
                }
                bool ok_s = rc_s2 == 0 && nsdiff == 0;
                std::printf("[CUDA]  scaleTrajectoryGPUDevice %s vs host unit conversion: "
                    "%zu/%zu values differ -> %s\n", tname, nsdiff, n6,
                    ok_s ? "OK (bitwise)" : "FAIL");
                ok_dev = ok_dev && ok_s;
                cudaFree(d_traj);
            };
            check_device_output(double{}, "fp64", traj_c, ACC);
            check_device_output(float{},  "fp32", traj_f, 1e-5);

            // A NULL destination must be rejected rather than dereferenced. (There is
            // no "wrong device" case to test: this entry point takes no device string
            // -- being CUDA-only is expressed in the signature, not validated at
            // runtime -- so a NULL buffer is the only misuse it can catch.)
            int rc_null = orbit::integrateOrbitsGPUDevice<double>(pot, NORB, ic.data(),
                times.data(), TRAJ, ACC, 100000000, (double*)NULL, 0);
            bool ok_null = rc_null == orbit::ORBIT_GPU_EUNSUPP;
            std::printf("[CUDA]  device-resident orbits, NULL destination rejected: "
                "rc=%d -> %s\n", rc_null, ok_null ? "OK" : "FAIL");
            ok_t3 = ok_t3 && ok_dev && ok_null;
        }
#endif
        if(!ok_t3) {
            std::fprintf(stderr, "FAIL (Tier 3 orbit integration)\n");
            return 1;
        }
    }

    // =====================================================================
    // Tier 1 modifiers: Shifted / Tilted / Rotating / Scaled on the descriptor.
    //
    // A chain of these collapses to one GpuPotXform (a similarity transform of
    // the position plus scalar rescalings of Phi, acc and rho -- see
    // GpuPotXform in potential_composite.h). This block pins that collapse
    // against the CPU virtual eval, which composes the modifiers by nesting
    // actual eval calls, for EVERY subset of the four modifiers, including all
    // the nested combinations. Two things make it worth testing exhaustively
    // rather than one-per-modifier:
    //   - the acceleration transforms with M^TRANSPOSE, and a wrong transpose is
    //     invisible for any chain without a rotation (M is then symmetric);
    //   - the stages must compose outermost-to-innermost, and the wrong order is
    //     invisible unless a translation and a rotation/scale are BOTH present
    //     (translations commute with everything else on their own).
    // The base potential is therefore a triaxial Logarithmic (no symmetry to
    // hide a bad rotation) and every subset with >= 2 modifiers is exercised.
    //
    // Modifiers are constructed directly rather than through the factory so this
    // test needs no INI files; the nesting order matches applyModifiers() in
    // potential_factory.cpp, i.e. Shifted(Tilted(Rotating(Scaled(base)))).
    // =====================================================================
    {
        // constant splines: a single node with zero derivative, exactly what
        // readTimeDependentArray() builds from a bare "x,y,z" parameter string
        const std::vector<double> t0(1, 0.0);
        #define AGAMA_CONST_SPLINE(v) math::CubicSpline(t0, std::vector<double>(1, (v)))
        const double CX = 0.31, CY = -0.22, CZ = 0.47;   // Shifted center
        const double ALPHA = 0.4, BETA = 0.9, GAMMA = -0.3;  // Tilted Euler angles
        const double ANGLE = 0.7;                        // Rotating angle
        const double AMPL = 1.3, LSCALE = 0.8;           // Scaled amplitude, length scale

        double max_rel_all = 0;
        bool ok_all = true;
        int nsubsets = 0;
        for(int mask = 0; mask < 16; mask++) {
            potential::PtrPotential p(new potential::Logarithmic(
                /*v0*/ 1.0, /*coreRadius*/ 0.5, /*axisRatioY*/ 0.7, /*axisRatioZ*/ 0.5));
            // innermost first, so the outermost wrapper ends up outermost --
            // same order as applyModifiers()
            if(mask & 1)
                p.reset(new potential::Scaled<potential::BasePotential>(p,
                    AGAMA_CONST_SPLINE(AMPL), AGAMA_CONST_SPLINE(LSCALE)));
            if(mask & 2)
                p.reset(new potential::Rotating<potential::BasePotential>(p,
                    AGAMA_CONST_SPLINE(ANGLE)));
            if(mask & 4)
                p.reset(new potential::Tilted<potential::BasePotential>(p,
                    ALPHA, BETA, GAMMA));
            if(mask & 8)
                p.reset(new potential::Shifted<potential::BasePotential>(p,
                    AGAMA_CONST_SPLINE(CX), AGAMA_CONST_SPLINE(CY), AGAMA_CONST_SPLINE(CZ)));

            potential::GpuPotDesc<double> desc;
            // every stage is constant here, so the builder must fold the whole
            // chain and emit NO stages -- asserting stageCount == 0 is what makes
            // a regression in CubicSpline::isConstant() visible (it would start
            // emitting per-step stages for a chain that cannot vary)
            if(!potential::buildGpuPotDesc(*p, desc, /*time*/ 0) ||
                desc.nterms != 1 || desc.terms[0].stageCount != 0)
            {
                std::fprintf(stderr,
                    "FAIL (modifier descriptor did not build, mask=%d)\n", mask);
                return 1;
            }
            nsubsets++;
            // Scale-normalized, not pointwise-relative: a triaxial Logarithmic's
            // Phi crosses zero on the sampled shell, and dividing by ~0 there
            // reports a huge error for a perfectly good result (the same trap
            // documented in local_notes/crosscheck_gpu_vs_production.py).
            double max_ad = 0, max_aa = 0, max_dd = 0, ref_p = 0, ref_a = 0, ref_d = 0;
            for(int i = 0; i < 200; i++) {
                const double x = 0.11 + 0.037 * (i % 53) - 0.9,
                             y = -0.43 + 0.029 * (i % 41),
                             z = 0.17 + 0.023 * (i % 31) - 0.4;
                double phi_d, acc_d[3];
                potential::gpu_desc_phi_acc(desc, x, y, z, &phi_d, acc_d);
                const double rho_d = potential::gpu_desc_dens(desc, x, y, z);
                double phi_v;
                coord::GradCar grad;
                p->eval(coord::PosCar(x, y, z), &phi_v, &grad, NULL);
                const double rho_v = p->density(coord::PosCar(x, y, z));
                const double acc_v[3] = { -grad.dx, -grad.dy, -grad.dz };
                max_ad = std::max(max_ad, std::fabs(phi_d - phi_v));
                ref_p  = std::max(ref_p,  std::fabs(phi_v));
                for(int k = 0; k < 3; k++) {
                    max_aa = std::max(max_aa, std::fabs(acc_d[k] - acc_v[k]));
                    ref_a  = std::max(ref_a,  std::fabs(acc_v[k]));
                }
                max_dd = std::max(max_dd, std::fabs(rho_d - rho_v));
                ref_d  = std::max(ref_d,  std::fabs(rho_v));
            }
            const double rel_p = max_ad / std::max(1e-300, ref_p),
                         rel_a = max_aa / std::max(1e-300, ref_a),
                         rel_d = max_dd / std::max(1e-300, ref_d);
            const double rel = std::max(rel_p, std::max(rel_a, rel_d));
            max_rel_all = std::max(max_rel_all, rel);
            // 1e-13: the transform is 13 multiply-adds on top of the leaf, so a
            // correct implementation lands at a few ULP (measured ~4e-16); this
            // leaves three decades of headroom while still catching any real
            // algebra error, which shows up at O(0.1) or larger.
            if(rel > 1e-13) {
                std::fprintf(stderr, "FAIL (modifier descriptor parity, mask=%d: "
                    "phi %.3e, acc %.3e, rho %.3e)\n", mask, rel_p, rel_a, rel_d);
                ok_all = false;
            }
        }
        std::printf("[T1]    modifier descriptor vs virtual eval "
            "(%d Shifted/Tilted/Rotating/Scaled subsets x 200 pts, triaxial Logarithmic, "
            "Phi+acc+rho): max rel err = %.3e, tol = %.1e -> %s\n",
            nsubsets, max_rel_all, 1e-13, ok_all ? "OK" : "FAIL");
        if(!ok_all)
            return 1;

        // ----- a time-VARYING chain: both modes must be right -----
        // Two callers with different needs:
        //  (a) batch eval shares one `time`, so folding the chain at that time is
        //      exact -- build with no spline buffer and check against the CPU eval
        //      AT THAT SAME time (this also proves `time` is actually threaded and
        //      not accepted and dropped);
        //  (b) the orbit kernel sees a different t at every RK stage, so it builds
        //      with a spline buffer and gets GpuModStages it re-evaluates per step
        //      -- check the SAME descriptor against the CPU eval at several
        //      different times, which a folded descriptor could not do.
        {
            std::vector<double> tt(2), vx(2), vy(2), vz(2);
            tt[0] = 0;    tt[1] = 10;
            vx[0] = 0;    vx[1] = 2.0;
            vy[0] = 0;    vy[1] = -1.0;
            vz[0] = 0.5;  vz[1] = 0.5;   // z constant, x and y moving
            potential::PtrPotential base(new potential::Logarithmic(1.0, 0.5, 0.7, 0.5));
            // a moving shift wrapped around a constant scaling: exercises the
            // constant-run flush, i.e. that a folded run keeps its place in the
            // composition order instead of being applied out of sequence
            potential::PtrPotential inner(new potential::Scaled<potential::BasePotential>(
                base, AGAMA_CONST_SPLINE(1.3), AGAMA_CONST_SPLINE(0.8)));
            potential::PtrPotential moving(new potential::Shifted<potential::BasePotential>(
                inner, math::CubicSpline(tt, vx), math::CubicSpline(tt, vy),
                math::CubicSpline(tt, vz)));

            const double TIMES[5] = { -2.0, 0.0, 3.75, 6.25, 12.0 };

            // (a) folded-at-one-time mode
            double worst_fold = 0;
            for(int q = 0; q < 5; q++) {
                potential::GpuPotDesc<double> d;
                if(!potential::buildGpuPotDesc(*moving, d, TIMES[q]) ||
                    d.nstages != 0 || d.terms[0].stageCount != 0)
                {
                    std::fprintf(stderr, "FAIL (folded build at t=%g)\n", TIMES[q]);
                    return 1;
                }
                worst_fold = std::max(worst_fold,
                    desc_vs_virtual(d, *moving, TIMES[q]));
            }

            // (b) stage mode: ONE descriptor, many times
            potential::GpuPotDesc<double> ds;
            std::vector<double> splineData;
            if(!potential::buildGpuPotDesc(*moving, ds, /*time*/ 0, &splineData)) {
                std::fprintf(stderr, "FAIL (stage-mode build)\n");
                return 1;
            }
            ds.splineData = splineData.empty() ? NULL : splineData.data();
            // the Shifted is genuinely time-varying, so at least one stage must
            // exist; the constant Scaled must have been flushed as a CONST stage
            // AFTER it (it is innermost), giving 2 stages in that order
            const bool shape_ok = ds.nstages == 2 &&
                ds.terms[0].stageCount == 2 &&
                ds.stages[0].kind == potential::GPU_MOD_SHIFTED &&
                ds.stages[1].kind == potential::GPU_MOD_CONST &&
                ds.terms[0].hasXform == 0 &&
                !splineData.empty();
            double worst_stage = 0;
            for(int q = 0; q < 5; q++)
                worst_stage = std::max(worst_stage,
                    desc_vs_virtual(ds, *moving, TIMES[q]));

            const bool ok_td = shape_ok && worst_fold <= 1e-13 && worst_stage <= 1e-13;
            std::printf("[T1]    time-varying Shifted(Scaled(Log)) at t = -2..12: "
                "folded-per-time max rel err = %.3e; ONE staged descriptor "
                "re-evaluated per t = %.3e (%d stages: %s); tol = 1.0e-13 -> %s\n",
                worst_fold, worst_stage, ds.nstages,
                shape_ok ? "SHIFTED+CONST as expected" : "UNEXPECTED SHAPE",
                ok_td ? "OK" : "FAIL");
            if(!ok_td) {
                std::fprintf(stderr, "FAIL (time-varying modifier stages)\n");
                return 1;
            }
        }
        #undef AGAMA_CONST_SPLINE
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

    // ----- math::sphHarmArray on Cuda (Tier 0 Legendre helpers) -----
    // One thread per (tau, m) pair; each writes W_l^m and its first two
    // theta-derivatives for l = m..LEG_LMAX into its own slice, exactly as a
    // Multipole eval kernel will. Compared against the Serial reference above.
    device_array<double> d_leg(LEG_N);
    double* dleg = d_leg.data();
    const int    dLMAX = LEG_LMAX, dNM = LEG_NM, dNTAU = LEG_NTAU;
    const std::size_t dSTRIDE = LEG_STRIDE;
    device_array<int> d_ms(LEG_NM);
    d_ms.from_host(LEG_MS, LEG_NM);
    const int* dms = d_ms.data();
    forall(Cuda{}, (std::size_t)LEG_NTAU * LEG_NM, [=] AGAMA_DEVICE (std::size_t k) {
        const int it = (int)(k / dNM), im = (int)(k % dNM);
        const double tau = -1.0 + 1e-12 + (2.0 - 2e-12) * (it + 0.5) / dNTAU;
        double* base = dleg + (((std::size_t)it * dNM + im) * 3) * dSTRIDE;
        math::sphHarmArray(dLMAX, dms[im], tau,
            base, base + dSTRIDE, base + 2 * dSTRIDE);
    });
    std::vector<double> leg_c(LEG_N);
    d_leg.to_host(leg_c.data(), LEG_N);
    // Relative comparison: the second derivatives reach O(1e4) at lmax=12 while the
    // values stay O(1), so an absolute tolerance would be meaningless. Host glibc
    // sqrt/sin/cos vs CUDA plus nvcc's FMA contraction gives a few ULPs per recurrence
    // step; 1e-11 relative is ~1e5 ULPs, loose enough for a 12-step recurrence yet
    // tight enough to catch a wrong table entry or an untaken branch.
    // Only the first (lmax - m + 1) entries of each quantity are written (l = m..lmax);
    // the rest of the stride is padding that sphHarmArray never touches, so it is
    // zero on the host and uninitialized device memory on the GPU -- skip it.
    double max_leg_relerr = 0.0;
    for(int it = 0; it < LEG_NTAU; ++it)
        for(int im = 0; im < LEG_NM; ++im) {
            const std::size_t base = (((std::size_t)it * LEG_NM + im) * 3) * LEG_STRIDE;
            const int nvalid = LEG_LMAX - LEG_MS[im] + 1;
            for(int q = 0; q < 3; ++q)
                for(int j = 0; j < nvalid; ++j) {
                    const std::size_t i = base + (std::size_t)q * LEG_STRIDE + j;
                    const double a = leg_s[i], b = leg_c[i];
                    // |a| can legitimately underflow near |tau|->1; fall back to an
                    // absolute comparison there rather than dividing by ~0
                    const double scale = std::fabs(a) > 1e-300 ? std::fabs(a) : 1.0;
                    const double e = std::fabs(a - b) / scale;
                    if(e > max_leg_relerr) max_leg_relerr = e;
                }
        }
    const double LEG_TOL = 1e-11;
    const bool ok_leg = (max_leg_relerr <= LEG_TOL);

    // -------------------------------------------------------------------
    // Tier 2 commit 1, GPU parity: toGrad<Sph,Cyl> / toHess<Sph,Cyl> Serial
    // vs Cuda over CD_N random inputs, compared at 1e-11 relative -- same
    // structure as the sphHarmArray Serial-vs-Cuda block just above, and the
    // two functions Multipole::evalCyl actually calls (via
    // transformDerivsSphToCyl). Measured 0.000e+00 (exact, not merely within
    // tolerance) on sm_86 -- expected, since these bodies are straight-line
    // multiply-adds with no transcendentals for the device libm to round
    // differently.
    // -------------------------------------------------------------------
    const std::size_t CD_N = 1000;
    std::mt19937_64 cd_rng(424242);
    std::uniform_real_distribution<double> cd_dist(-5.0, 5.0);
    // flat host buffers: GradCyl src (3), PosDerivT<Sph,Cyl> d (4) -> GradSph out (3)
    std::vector<double> cd_src(CD_N * 3), cd_d(CD_N * 4), cd_grad_s(CD_N * 3);
    // flat host buffers: GradCyl srcGrad(3), HessCyl srcHess(6), same d(4),
    // PosDeriv2T<Sph,Cyl> d2(4) -> HessSph out (6)
    std::vector<double> cd_sg(CD_N * 3), cd_sh(CD_N * 6), cd_d2(CD_N * 4), cd_hess_s(CD_N * 6);
    for(std::size_t i = 0; i < CD_N; ++i) {
        for(int k = 0; k < 3; ++k) cd_src[i*3+k] = cd_dist(cd_rng);
        for(int k = 0; k < 4; ++k) cd_d[i*4+k]   = cd_dist(cd_rng);
        for(int k = 0; k < 3; ++k) cd_sg[i*3+k]  = cd_dist(cd_rng);
        for(int k = 0; k < 6; ++k) cd_sh[i*6+k]  = cd_dist(cd_rng);
        for(int k = 0; k < 4; ++k) cd_d2[i*4+k]  = cd_dist(cd_rng);

        coord::GradCyl src = {cd_src[i*3+0], cd_src[i*3+1], cd_src[i*3+2]};
        coord::PosDerivT<coord::Sph,coord::Cyl> d = {cd_d[i*4+0], cd_d[i*4+1], cd_d[i*4+2], cd_d[i*4+3]};
        coord::GradSph r = coord::toGrad(src, d);
        cd_grad_s[i*3+0] = r.dr; cd_grad_s[i*3+1] = r.dtheta; cd_grad_s[i*3+2] = r.dphi;

        coord::GradCyl sg = {cd_sg[i*3+0], cd_sg[i*3+1], cd_sg[i*3+2]};
        coord::HessCyl sh = {cd_sh[i*6+0], cd_sh[i*6+1], cd_sh[i*6+2], cd_sh[i*6+3], cd_sh[i*6+4], cd_sh[i*6+5]};
        coord::PosDeriv2T<coord::Sph,coord::Cyl> d2 = {cd_d2[i*4+0], cd_d2[i*4+1], cd_d2[i*4+2], cd_d2[i*4+3]};
        coord::HessSph rh = coord::toHess(sg, sh, d, d2);
        cd_hess_s[i*6+0] = rh.dr2; cd_hess_s[i*6+1] = rh.dtheta2; cd_hess_s[i*6+2] = rh.dphi2;
        cd_hess_s[i*6+3] = rh.drdtheta; cd_hess_s[i*6+4] = rh.dthetadphi; cd_hess_s[i*6+5] = rh.drdphi;
    }
    device_array<double> d_cd_src(CD_N * 3), d_cd_d(CD_N * 4), d_cd_grad(CD_N * 3);
    device_array<double> d_cd_sg(CD_N * 3), d_cd_sh(CD_N * 6), d_cd_d2(CD_N * 4), d_cd_hess(CD_N * 6);
    d_cd_src.from_host(cd_src.data(), CD_N * 3);
    d_cd_d.from_host(cd_d.data(), CD_N * 4);
    d_cd_sg.from_host(cd_sg.data(), CD_N * 3);
    d_cd_sh.from_host(cd_sh.data(), CD_N * 6);
    d_cd_d2.from_host(cd_d2.data(), CD_N * 4);
    {
        const double *dsrc = d_cd_src.data(), *dd = d_cd_d.data();
        double* dgrad = d_cd_grad.data();
        forall(Cuda{}, CD_N, [=] AGAMA_DEVICE (std::size_t i) {
            coord::GradCyl src = {dsrc[i*3+0], dsrc[i*3+1], dsrc[i*3+2]};
            coord::PosDerivT<coord::Sph,coord::Cyl> d = {dd[i*4+0], dd[i*4+1], dd[i*4+2], dd[i*4+3]};
            coord::GradSph r = coord::toGrad(src, d);
            dgrad[i*3+0] = r.dr; dgrad[i*3+1] = r.dtheta; dgrad[i*3+2] = r.dphi;
        });
        const double *dsg = d_cd_sg.data(), *dsh = d_cd_sh.data(), *dd2 = d_cd_d2.data();
        double* dhess = d_cd_hess.data();
        forall(Cuda{}, CD_N, [=] AGAMA_DEVICE (std::size_t i) {
            coord::GradCyl sg = {dsg[i*3+0], dsg[i*3+1], dsg[i*3+2]};
            coord::HessCyl sh = {dsh[i*6+0], dsh[i*6+1], dsh[i*6+2], dsh[i*6+3], dsh[i*6+4], dsh[i*6+5]};
            coord::PosDerivT<coord::Sph,coord::Cyl> d = {dd[i*4+0], dd[i*4+1], dd[i*4+2], dd[i*4+3]};
            coord::PosDeriv2T<coord::Sph,coord::Cyl> d2 = {dd2[i*4+0], dd2[i*4+1], dd2[i*4+2], dd2[i*4+3]};
            coord::HessSph rh = coord::toHess(sg, sh, d, d2);
            dhess[i*6+0] = rh.dr2; dhess[i*6+1] = rh.dtheta2; dhess[i*6+2] = rh.dphi2;
            dhess[i*6+3] = rh.drdtheta; dhess[i*6+4] = rh.dthetadphi; dhess[i*6+5] = rh.drdphi;
        });
    }
    std::vector<double> cd_grad_c(CD_N * 3), cd_hess_c(CD_N * 6);
    d_cd_grad.to_host(cd_grad_c.data(), CD_N * 3);
    d_cd_hess.to_host(cd_hess_c.data(), CD_N * 6);
    double max_cd_relerr = 0.0;
    for(std::size_t i = 0; i < CD_N * 3; ++i) {
        const double a = cd_grad_s[i], b = cd_grad_c[i];
        const double scale = std::fabs(a) > 1e-300 ? std::fabs(a) : 1.0;
        max_cd_relerr = std::max(max_cd_relerr, std::fabs(a - b) / scale);
    }
    for(std::size_t i = 0; i < CD_N * 6; ++i) {
        const double a = cd_hess_s[i], b = cd_hess_c[i];
        const double scale = std::fabs(a) > 1e-300 ? std::fabs(a) : 1.0;
        max_cd_relerr = std::max(max_cd_relerr, std::fabs(a - b) / scale);
    }
    const double CD_TOL = 1e-11;
    const bool ok_coordderiv_gpu = (max_cd_relerr <= CD_TOL);
    std::printf("[CUDA]  toGrad<Sph,Cyl>/toHess<Sph,Cyl> Serial vs Cuda (N=%zu): "
        "max rel err = %.3e, tol = %.1e -> %s\n",
        CD_N, max_cd_relerr, CD_TOL, ok_coordderiv_gpu ? "OK" : "FAIL");
    // ----- evalQuinticSplineRaw / evalQuinticSpline2dRaw on Cuda -----
    // Measured 0.000e+00 (exact) on sm_86 for BOTH fp64 and fp32 -- the raw
    // evaluators are pure arithmetic over caller-supplied arrays, so host and
    // device agree bit-for-bit; the fp32 tolerance below is headroom, not need.
    // Mirrors the sphHarmArray Cuda block above as closely as
    // possible (device_array upload, forall<Cuda> over ~1000 points, download,
    // compare against a Serial host reference computed through the SAME raw
    // function so this isolates device-vs-host codegen, not wrapper vs raw).
    // Reuses the qx/qf/qd/qd2 (1d) and gx/gy/f2/fx2/fy2/fxy2 (2d) node data
    // defined and bit-for-bit-checked in the [CPU] Tier 2 block above.
    bool ok_quintic_cuda = true;
    {
        // ---- 1d, fp64 ----
        const int NQ1 = 1000;
        std::vector<double> qxs(NQ1);
        for(int i = 0; i < NQ1; ++i)
            qxs[i] = qx.front() - 1.0 + (qx.back() - qx.front() + 2.0) * i / (NQ1 - 1);
        const int nq = (int)qx.size();

        device_array<double> d_qx(qx.size()), d_qf(qx.size()), d_qd(qx.size()), d_qd2(qx.size());
        d_qx.from_host(qx.data(), qx.size());
        d_qf.from_host(qf.data(), qx.size());
        d_qd.from_host(qd.data(), qx.size());
        d_qd2.from_host(qd2.data(), qx.size());
        device_array<double> d_qxs(NQ1);
        d_qxs.from_host(qxs.data(), NQ1);
        device_array<double> d_qv(NQ1), d_qd1(NQ1), d_qd2o(NQ1), d_qd3(NQ1);
        const double *dqx = d_qx.data(), *dqf = d_qf.data(), *dqd = d_qd.data(), *dqd2 = d_qd2.data();
        const double *dqxs = d_qxs.data();
        double *dqv = d_qv.data(), *dqd1 = d_qd1.data(), *dqd2o = d_qd2o.data(), *dqd3 = d_qd3.data();
        forall(Cuda{}, (std::size_t)NQ1, [=] AGAMA_DEVICE (std::size_t i) {
            math::evalQuinticSplineRaw(dqxs[i], dqx, dqf, dqd, dqd2, nq,
                &dqv[i], &dqd1[i], &dqd2o[i], &dqd3[i]);
        });
        std::vector<double> qv_c(NQ1), qd1_c(NQ1), qd2o_c(NQ1), qd3_c(NQ1);
        d_qv.to_host(qv_c.data(), NQ1);
        d_qd1.to_host(qd1_c.data(), NQ1);
        d_qd2o.to_host(qd2o_c.data(), NQ1);
        d_qd3.to_host(qd3_c.data(), NQ1);
        std::vector<double> qv_s(NQ1), qd1_s(NQ1), qd2o_s(NQ1), qd3_s(NQ1);
        for(int i = 0; i < NQ1; ++i)
            math::evalQuinticSplineRaw(qxs[i], qx.data(), qf.data(), qd.data(), qd2.data(), nq,
                &qv_s[i], &qd1_s[i], &qd2o_s[i], &qd3_s[i]);
        double max_q1_relerr = 0.0;
        for(int i = 0; i < NQ1; ++i) {
            const double a[4] = { qv_s[i], qd1_s[i], qd2o_s[i], qd3_s[i] };
            const double b[4] = { qv_c[i], qd1_c[i], qd2o_c[i], qd3_c[i] };
            for(int k = 0; k < 4; ++k) {
                if(a[k] != a[k]) continue;  // both sides NaN in the same (unreachable here) branch
                const double scale = std::fabs(a[k]) > 1e-300 ? std::fabs(a[k]) : 1.0;
                const double e = std::fabs(a[k]-b[k]) / scale;
                if(e > max_q1_relerr) max_q1_relerr = e;
            }
        }
        const double Q1_TOL = 1e-11;
        const bool ok_q1 = max_q1_relerr <= Q1_TOL;

        // ---- 1d, fp32 ----
        std::vector<float> qxf(qx.begin(), qx.end()), qff(qf.begin(), qf.end()),
            qdf(qd.begin(), qd.end()), qd2f(qd2.begin(), qd2.end()), qxsf(qxs.begin(), qxs.end());
        device_array<float> d_qxf(qx.size()), d_qff(qx.size()), d_qdf(qx.size()), d_qd2f(qx.size());
        d_qxf.from_host(qxf.data(), qx.size());
        d_qff.from_host(qff.data(), qx.size());
        d_qdf.from_host(qdf.data(), qx.size());
        d_qd2f.from_host(qd2f.data(), qx.size());
        device_array<float> d_qxsf(NQ1);
        d_qxsf.from_host(qxsf.data(), NQ1);
        device_array<float> d_qvf(NQ1), d_qd1f(NQ1), d_qd2of(NQ1), d_qd3f(NQ1);
        const float *dqxf = d_qxf.data(), *dqff = d_qff.data(), *dqdf = d_qdf.data(), *dqd2fp = d_qd2f.data();
        const float *dqxsf = d_qxsf.data();
        float *dqvf = d_qvf.data(), *dqd1f = d_qd1f.data(), *dqd2of = d_qd2of.data(), *dqd3f = d_qd3f.data();
        forall(Cuda{}, (std::size_t)NQ1, [=] AGAMA_DEVICE (std::size_t i) {
            math::evalQuinticSplineRaw(dqxsf[i], dqxf, dqff, dqdf, dqd2fp, nq,
                &dqvf[i], &dqd1f[i], &dqd2of[i], &dqd3f[i]);
        });
        std::vector<float> qvf_c(NQ1), qd1f_c(NQ1), qd2of_c(NQ1), qd3f_c(NQ1);
        d_qvf.to_host(qvf_c.data(), NQ1);
        d_qd1f.to_host(qd1f_c.data(), NQ1);
        d_qd2of.to_host(qd2of_c.data(), NQ1);
        d_qd3f.to_host(qd3f_c.data(), NQ1);
        std::vector<float> qvf_s(NQ1), qd1f_s(NQ1), qd2of_s(NQ1), qd3f_s(NQ1);
        for(int i = 0; i < NQ1; ++i)
            math::evalQuinticSplineRaw(qxsf[i], qxf.data(), qff.data(), qdf.data(), qd2f.data(), nq,
                &qvf_s[i], &qd1f_s[i], &qd2of_s[i], &qd3f_s[i]);
        float max_q1f_relerr = 0.0f;
        for(int i = 0; i < NQ1; ++i) {
            const float a[4] = { qvf_s[i], qd1f_s[i], qd2of_s[i], qd3f_s[i] };
            const float b[4] = { qvf_c[i], qd1f_c[i], qd2of_c[i], qd3f_c[i] };
            for(int k = 0; k < 4; ++k) {
                if(a[k] != a[k]) continue;
                const float scale = std::fabs(a[k]) > 1e-30f ? std::fabs(a[k]) : 1.0f;
                const float e = std::fabs(a[k]-b[k]) / scale;
                if(e > max_q1f_relerr) max_q1f_relerr = e;
            }
        }
        const float Q1F_TOL = 1e-6f;
        const bool ok_q1f = max_q1f_relerr <= Q1F_TOL;

        // ---- 2d, fp64 ----
        const int NQ2 = 1024;  // 32x32
        std::vector<double> qx2(NQ2), qy2(NQ2);
        for(int i = 0; i < NQ2; ++i) {
            qx2[i] = gx.front() - 1.0 + (gx.back()-gx.front()+2.0) * (i % 32) / 31.0;
            qy2[i] = gy.front() - 1.0 + (gy.back()-gy.front()+2.0) * ((i / 32) % 32) / 31.0;
        }
        const std::size_t nxg = gx.size(), nyg = gy.size();
        std::vector<double> vfval(nxg*nyg), vfx(nxg*nyg), vfy(nxg*nyg), vfxx(nxg*nyg, 0.0),
            vfxy(nxg*nyg), vfyy(nxg*nyg, 0.0), vfxxy(nxg*nyg, 0.0), vfxyy(nxg*nyg, 0.0),
            vfxxyy(nxg*nyg, 0.0);
        for(std::size_t i = 0; i < nxg; ++i)
            for(std::size_t j = 0; j < nyg; ++j) {
                const std::size_t idx = i*nyg + j;
                vfval[idx] = f2(i,j); vfx[idx] = fx2(i,j); vfy[idx] = fy2(i,j); vfxy[idx] = fxy2(i,j);
            }
        device_array<double> d_gx(nxg), d_gy(nyg), d_fval(nxg*nyg), d_fx(nxg*nyg), d_fy(nxg*nyg),
            d_fxx(nxg*nyg), d_fxy(nxg*nyg), d_fyy(nxg*nyg), d_fxxy(nxg*nyg), d_fxyy(nxg*nyg),
            d_fxxyy(nxg*nyg);
        d_gx.from_host(gx.data(), nxg);
        d_gy.from_host(gy.data(), nyg);
        d_fval.from_host(vfval.data(), nxg*nyg);
        d_fx.from_host(vfx.data(), nxg*nyg);
        d_fy.from_host(vfy.data(), nxg*nyg);
        d_fxx.from_host(vfxx.data(), nxg*nyg);
        d_fxy.from_host(vfxy.data(), nxg*nyg);
        d_fyy.from_host(vfyy.data(), nxg*nyg);
        d_fxxy.from_host(vfxxy.data(), nxg*nyg);
        d_fxyy.from_host(vfxyy.data(), nxg*nyg);
        d_fxxyy.from_host(vfxxyy.data(), nxg*nyg);
        device_array<double> d_qx2(NQ2), d_qy2(NQ2);
        d_qx2.from_host(qx2.data(), NQ2);
        d_qy2.from_host(qy2.data(), NQ2);
        device_array<double> d_z(NQ2), d_zx(NQ2), d_zy(NQ2), d_zxx(NQ2), d_zxy(NQ2), d_zyy(NQ2);
        const double *ddgx = d_gx.data(), *ddgy = d_gy.data();
        const double *ddfval = d_fval.data(), *ddfx = d_fx.data(), *ddfy = d_fy.data(),
            *ddfxx = d_fxx.data(), *ddfxy = d_fxy.data(), *ddfyy = d_fyy.data(),
            *ddfxxy = d_fxxy.data(), *ddfxyy = d_fxyy.data(), *ddfxxyy = d_fxxyy.data();
        const double *ddqx2 = d_qx2.data(), *ddqy2 = d_qy2.data();
        double *ddz = d_z.data(), *ddzx = d_zx.data(), *ddzy = d_zy.data(),
            *ddzxx = d_zxx.data(), *ddzxy = d_zxy.data(), *ddzyy = d_zyy.data();
        const int dnxg = (int)nxg, dnyg = (int)nyg;
        forall(Cuda{}, (std::size_t)NQ2, [=] AGAMA_DEVICE (std::size_t i) {
            math::evalQuinticSpline2dRaw(ddqx2[i], ddqy2[i], ddgx, ddgy, dnxg, dnyg,
                ddfval, ddfx, ddfy, ddfxx, ddfxy, ddfyy, ddfxxy, ddfxyy, ddfxxyy,
                &ddz[i], &ddzx[i], &ddzy[i], &ddzxx[i], &ddzxy[i], &ddzyy[i]);
        });
        std::vector<double> z_c(NQ2), zx_c(NQ2), zy_c(NQ2), zxx_c(NQ2), zxy_c(NQ2), zyy_c(NQ2);
        d_z.to_host(z_c.data(), NQ2); d_zx.to_host(zx_c.data(), NQ2); d_zy.to_host(zy_c.data(), NQ2);
        d_zxx.to_host(zxx_c.data(), NQ2); d_zxy.to_host(zxy_c.data(), NQ2); d_zyy.to_host(zyy_c.data(), NQ2);
        std::vector<double> z_s(NQ2), zx_s(NQ2), zy_s(NQ2), zxx_s(NQ2), zxy_s(NQ2), zyy_s(NQ2);
        for(int i = 0; i < NQ2; ++i)
            math::evalQuinticSpline2dRaw(qx2[i], qy2[i], gx.data(), gy.data(), dnxg, dnyg,
                vfval.data(), vfx.data(), vfy.data(), vfxx.data(), vfxy.data(), vfyy.data(),
                vfxxy.data(), vfxyy.data(), vfxxyy.data(),
                &z_s[i], &zx_s[i], &zy_s[i], &zxx_s[i], &zxy_s[i], &zyy_s[i]);
        double max_q2_relerr = 0.0;
        for(int i = 0; i < NQ2; ++i) {
            const double a[6] = { z_s[i], zx_s[i], zy_s[i], zxx_s[i], zxy_s[i], zyy_s[i] };
            const double b[6] = { z_c[i], zx_c[i], zy_c[i], zxx_c[i], zxy_c[i], zyy_c[i] };
            for(int k = 0; k < 6; ++k) {
                if(a[k] != a[k]) continue;
                const double scale = std::fabs(a[k]) > 1e-300 ? std::fabs(a[k]) : 1.0;
                const double e = std::fabs(a[k]-b[k]) / scale;
                if(e > max_q2_relerr) max_q2_relerr = e;
            }
        }
        const double Q2_TOL = 1e-11;
        const bool ok_q2 = max_q2_relerr <= Q2_TOL;

        // ---- 2d, fp32 (values only, mirroring the fp64 case above) ----
        std::vector<float> gxf(gx.begin(), gx.end()), gyf(gy.begin(), gy.end());
        std::vector<float> vfvalf(nxg*nyg), vfxf(nxg*nyg), vfyf(nxg*nyg), vfxxf(nxg*nyg, 0.f),
            vfxyf(nxg*nyg), vfyyf(nxg*nyg, 0.f), vfxxyf(nxg*nyg, 0.f), vfxyyf(nxg*nyg, 0.f),
            vfxxyyf(nxg*nyg, 0.f);
        for(std::size_t i = 0; i < nxg; ++i)
            for(std::size_t j = 0; j < nyg; ++j) {
                const std::size_t idx = i*nyg + j;
                vfvalf[idx] = (float)vfval[idx]; vfxf[idx] = (float)vfx[idx];
                vfyf[idx] = (float)vfy[idx]; vfxyf[idx] = (float)vfxy[idx];
            }
        std::vector<float> qx2f(qx2.begin(), qx2.end()), qy2f(qy2.begin(), qy2.end());
        device_array<float> d_gxf(nxg), d_gyf(nyg), d_fvalf(nxg*nyg), d_fxf(nxg*nyg), d_fyf(nxg*nyg),
            d_fxxf(nxg*nyg), d_fxyf(nxg*nyg), d_fyyf(nxg*nyg), d_fxxyf(nxg*nyg), d_fxyyf(nxg*nyg),
            d_fxxyyf(nxg*nyg);
        d_gxf.from_host(gxf.data(), nxg);
        d_gyf.from_host(gyf.data(), nyg);
        d_fvalf.from_host(vfvalf.data(), nxg*nyg);
        d_fxf.from_host(vfxf.data(), nxg*nyg);
        d_fyf.from_host(vfyf.data(), nxg*nyg);
        d_fxxf.from_host(vfxxf.data(), nxg*nyg);
        d_fxyf.from_host(vfxyf.data(), nxg*nyg);
        d_fyyf.from_host(vfyyf.data(), nxg*nyg);
        d_fxxyf.from_host(vfxxyf.data(), nxg*nyg);
        d_fxyyf.from_host(vfxyyf.data(), nxg*nyg);
        d_fxxyyf.from_host(vfxxyyf.data(), nxg*nyg);
        device_array<float> d_qx2f(NQ2), d_qy2f(NQ2);
        d_qx2f.from_host(qx2f.data(), NQ2);
        d_qy2f.from_host(qy2f.data(), NQ2);
        device_array<float> d_zf(NQ2), d_zxf(NQ2), d_zyf(NQ2), d_zxxf(NQ2), d_zxyf(NQ2), d_zyyf(NQ2);
        const float *ddgxf = d_gxf.data(), *ddgyf = d_gyf.data();
        const float *ddfvalf = d_fvalf.data(), *ddfxf = d_fxf.data(), *ddfyf = d_fyf.data(),
            *ddfxxf = d_fxxf.data(), *ddfxyf = d_fxyf.data(), *ddfyyf = d_fyyf.data(),
            *ddfxxyf = d_fxxyf.data(), *ddfxyyf = d_fxyyf.data(), *ddfxxyyf = d_fxxyyf.data();
        const float *ddqx2f = d_qx2f.data(), *ddqy2f = d_qy2f.data();
        float *ddzf = d_zf.data(), *ddzxf = d_zxf.data(), *ddzyf = d_zyf.data(),
            *ddzxxf = d_zxxf.data(), *ddzxyf = d_zxyf.data(), *ddzyyf = d_zyyf.data();
        forall(Cuda{}, (std::size_t)NQ2, [=] AGAMA_DEVICE (std::size_t i) {
            math::evalQuinticSpline2dRaw(ddqx2f[i], ddqy2f[i], ddgxf, ddgyf, dnxg, dnyg,
                ddfvalf, ddfxf, ddfyf, ddfxxf, ddfxyf, ddfyyf, ddfxxyf, ddfxyyf, ddfxxyyf,
                &ddzf[i], &ddzxf[i], &ddzyf[i], &ddzxxf[i], &ddzxyf[i], &ddzyyf[i]);
        });
        std::vector<float> zf_c(NQ2), zxf_c(NQ2), zyf_c(NQ2), zxxf_c(NQ2), zxyf_c(NQ2), zyyf_c(NQ2);
        d_zf.to_host(zf_c.data(), NQ2); d_zxf.to_host(zxf_c.data(), NQ2); d_zyf.to_host(zyf_c.data(), NQ2);
        d_zxxf.to_host(zxxf_c.data(), NQ2); d_zxyf.to_host(zxyf_c.data(), NQ2); d_zyyf.to_host(zyyf_c.data(), NQ2);
        std::vector<float> zf_s(NQ2), zxf_s(NQ2), zyf_s(NQ2), zxxf_s(NQ2), zxyf_s(NQ2), zyyf_s(NQ2);
        for(int i = 0; i < NQ2; ++i)
            math::evalQuinticSpline2dRaw(qx2f[i], qy2f[i], gxf.data(), gyf.data(), dnxg, dnyg,
                vfvalf.data(), vfxf.data(), vfyf.data(), vfxxf.data(), vfxyf.data(), vfyyf.data(),
                vfxxyf.data(), vfxyyf.data(), vfxxyyf.data(),
                &zf_s[i], &zxf_s[i], &zyf_s[i], &zxxf_s[i], &zxyf_s[i], &zyyf_s[i]);
        float max_q2f_relerr = 0.0f;
        for(int i = 0; i < NQ2; ++i) {
            const float a[6] = { zf_s[i], zxf_s[i], zyf_s[i], zxxf_s[i], zxyf_s[i], zyyf_s[i] };
            const float b[6] = { zf_c[i], zxf_c[i], zyf_c[i], zxxf_c[i], zxyf_c[i], zyyf_c[i] };
            for(int k = 0; k < 6; ++k) {
                if(a[k] != a[k]) continue;
                const float scale = std::fabs(a[k]) > 1e-30f ? std::fabs(a[k]) : 1.0f;
                const float e = std::fabs(a[k]-b[k]) / scale;
                if(e > max_q2f_relerr) max_q2f_relerr = e;
            }
        }
        const float Q2F_TOL = 1e-6f;
        const bool ok_q2f = max_q2f_relerr <= Q2F_TOL;

        ok_quintic_cuda = ok_q1 && ok_q1f && ok_q2 && ok_q2f;
        std::printf("[CUDA]  evalQuinticSplineRaw    Serial vs Cuda (N=%d, fp64): max rel err = %.3e, tol=%.1e -> %s\n",
            NQ1, max_q1_relerr, Q1_TOL, ok_q1 ? "OK" : "FAIL");
        std::printf("[CUDA]  evalQuinticSplineRaw    Serial vs Cuda (N=%d, fp32): max rel err = %.3e, tol=%.1e -> %s\n",
            NQ1, max_q1f_relerr, Q1F_TOL, ok_q1f ? "OK" : "FAIL");
        std::printf("[CUDA]  evalQuinticSpline2dRaw  Serial vs Cuda (N=%d, fp64): max rel err = %.3e, tol=%.1e -> %s\n",
            NQ2, max_q2_relerr, Q2_TOL, ok_q2 ? "OK" : "FAIL");
        std::printf("[CUDA]  evalQuinticSpline2dRaw  Serial vs Cuda (N=%d, fp32): max rel err = %.3e, tol=%.1e -> %s\n",
            NQ2, max_q2f_relerr, Q2F_TOL, ok_q2f ? "OK" : "FAIL");
    }

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
    std::printf("[CUDA]  sphHarmArray Serial vs Cuda (lmax=%d, m=0/1/2/3/%d, %d tau incl. |tau|->1, "
        "W+dW+d2W): max rel err = %.3e, tol = %.1e -> %s\n",
        LEG_LMAX, LEG_LMAX, LEG_NTAU, max_leg_relerr, LEG_TOL, ok_leg ? "OK" : "FAIL");

    if (!(ok_cpu && ok_gpu && ok_trig && ok_coord && ok_leg && ok_legtab && ok_powint && 
          ok_coordderiv && ok_coordderiv_gpu && ok_shipod && ok_quintic_raw && ok_quintic_cuda)) {
        std::fprintf(stderr, "FAIL\n");
        return 1;
    }
    std::printf("PASS (Serial/OpenMP/Cuda agree on forall, reduce, trigMultiAngle, toPos, "
        "sphHarmArray to %.0e, and quintic-spline raw evaluators)\n", TRIG_TOL);

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
    if (!(ok_cpu && ok_legtab && ok_powint && ok_coordderiv && ok_shipod && ok_quintic_raw)) {
        std::fprintf(stderr, "FAIL (CPU only)\n");
        return 1;
    }
    std::printf("[CPU]   trigMultiAngle ran for %zu samples, mmax=%u (no GPU to compare against)\n", NP, MM);
    std::printf("[CPU]   sphHarmArray ran for %d tau x %d orders (no GPU to compare against)\n",
        LEG_NTAU, LEG_NM);
    std::printf("[CPU]   toGrad/toHess Car/Cyl/Sph triangle: %ld values compared, %ld "
        "mismatches vs frozen ref_* bodies -> %s (no GPU to compare Serial-vs-Cuda against)\n",
        coordderiv_compared, coordderiv_mismatches, ok_coordderiv ? "OK" : "FAIL");
    std::printf("PASS (CPU only — HAVE_CUDA not defined)\n");
    return 0;
#endif
}
