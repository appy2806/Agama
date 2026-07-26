"""Tier 1 Python smoke test: pot.potential(xyz, device=..., dtype=...) parity.

For each of the six GPU-migrated analytic potentials (Plummer, Isochrone, NFW,
MiyamotoNagai, Logarithmic, Harmonic), verifies that:

  * The legacy `pot.potential(xyz)` path (no device kwarg) and the new
    `pot.potential(xyz, device=..., dtype=...)` device path produce the same
    Phi values to within ULP-scale relative tolerance, for all four device
    strings:
      - 'cpu'    = OpenMP-parallel across all cores (what AGAMA normally does;
                   'cpu' is the honest legacy-equivalent baseline)
      - 'openmp' = explicit alias of 'cpu'
      - 'serial' = single-thread loop (debugging / baseline); must match
                   'cpu'/'openmp' results EXACTLY (bit-for-bit), since the
                   per-element math is identical and order-independent
      - 'cuda'   = GPU batch path

  * Four Python threads calling pot.potential(xyz, device='cuda') concurrently
    (with different N to exercise scratch-buffer reserve/growth) neither crash
    nor produce wrong values -- the persistent GPU scratch buffers are
    mutex-protected because the binding releases the GIL around the call.

  * Input flexibility — numpy ndarray, list of tuples, list of lists, and
    raw nested Python sequences — all reach the same code path and produce
    the same result.

  * Single-point input (length-3 array or 3-tuple) returns a 0-D scalar.

  * Composite potentials of GPU-capable members dispatch on-device: each member
    kernel accumulates into the same output buffer (zero extra transfers), with
    full parity against the legacy CPU path, for host (numpy) and
    device-resident (CuPy) inputs.

  * device='cuda' on a composite containing a non-GPU-capable member raises
    NotImplementedError naming the first unsupported member type.

  * A malformed __cuda_array_interface__ 'stream' value raises RuntimeError
    instead of aborting the interpreter (exceptions no longer escape through
    Py_BEGIN_ALLOW_THREADS) -- verified in a subprocess so a regression cannot
    kill this test runner.

  * fp32 accuracy across NFW's Pade-vs-closed-form crossover in r/r_s, which the
    parity sweep above is structurally blind to -- see nfw_fp32_crossover_test().

Tolerances:
  fp64: relative ~1e-12 vs max|phi| (1 ULP ~ 1e-16, leaves margin for FMA)
  fp32: relative ~5e-6  vs max|phi| for the potential; 2e-6 for density and 3e-6 for
        force (the force exception is NFW's genuine fp32 floor near its Pade crossover,
        pinned independently by nfw_fp32_crossover_test)

  These are PARITY tolerances -- legacy vs device path for the same precision -- and a
  loose one hides a real accuracy defect rather than a backend difference. The fp32 force
  budget was 5e-6 until an actual 4.7e-4 error in NFW's dPhi/dr was found underneath it
  (fp64-tuned Pade thresholds reused in fp32; fixed by nfw_pade_guard<T>). Before widening
  any tolerance here, check whether the number it is accommodating is a bug.
"""
import os
import subprocess
import sys
import tempfile
import threading
import numpy as np
import agama_migrate as agama


def _check(name, pot, xyz, device, dtype, ref):
    """One (pot, device, dtype) combo: call device path, compare to ref."""
    phi = pot.potential(xyz, device=device, dtype=dtype)
    if phi.dtype != dtype:
        print(f"  FAIL {name:14s} {device:6s} {dtype.__name__:8s} : dtype {phi.dtype} != {dtype}")
        return False
    if phi.shape != ref.shape:
        print(f"  FAIL {name:14s} {device:6s} {dtype.__name__:8s} : shape {phi.shape} != {ref.shape}")
        return False
    max_ref = float(np.max(np.abs(ref))) if ref.size else 1.0
    rel_tol = 5e-6 if dtype == np.float32 else 1e-12
    abs_tol = rel_tol * max_ref
    err = float(np.max(np.abs(phi.astype(np.float64) - ref)))
    ok = err <= abs_tol
    print(f"  {'OK  ' if ok else 'FAIL'} {name:14s} {device:6s} {dtype.__name__:8s} "
          f": max|phi|={max_ref:.3e}  |err|={err:.3e}  tol={abs_tol:.3e}")
    return ok


def threading_smoke_test(pot, n_threads=4, iters=50):
    """Concurrent device='cuda' calls from multiple Python threads.

    The binding releases the GIL around evalPotentialGPU, so threads really do
    run the cuda branch concurrently. Each thread alternates between a small
    (100k) and a large (4M) input to force scratch-buffer reserve() growth
    while other threads' kernels/copies are in flight -- without the mutex in
    potential_gpu.cpp this crashes (use-after-cudaFree) or returns corrupted
    values (interleaved from_host into the shared scratch buffer).
    Every result is compared against a per-size device='cpu' reference.
    Returns True on success."""
    rng = np.random.default_rng(7)
    sizes = [100_000, 4_000_000]
    xyzs = {n: rng.uniform(-5.0, 5.0, size=(n, 3)) for n in sizes}
    refs = {n: pot.potential(xyzs[n], device="cpu", dtype=np.float64) for n in sizes}
    tols = {n: 1e-12 * float(np.max(np.abs(refs[n]))) for n in sizes}
    errors = []

    def worker(tid):
        try:
            for it in range(iters):
                n = sizes[(tid + it) % 2]   # alternate 100k / 4M per iteration
                phi = pot.potential(xyzs[n], device="cuda", dtype=np.float64)
                err = float(np.max(np.abs(phi - refs[n])))
                if err > tols[n]:
                    errors.append(f"thread {tid} iter {it} N={n}: |err|={err:.3e}"
                                  f" > tol={tols[n]:.3e}")
                    return
        except Exception as e:
            errors.append(f"thread {tid}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for msg in errors:
        print(f"  FAIL threading smoke: {msg}")
    ok = not errors
    print(f"  {'OK  ' if ok else 'FAIL'} {n_threads} threads x {iters} iters of "
          f"concurrent device='cuda' calls (N alternating 100k/4M)")
    return ok


def cupy_passthrough_tests(pots, xyz):
    """Phase B: __cuda_array_interface__ passthrough (CuPy in -> CuPy out).

    Device-resident input skips H2D/D2H entirely; the returned array is a
    cupy.ndarray living on the GPU. Covers: fp64/fp32 parity vs the NumPy
    device='cuda' path, dtype inference from the device array (no dtype kwarg),
    explicit matching / mismatched dtype kwarg, single-point (3,) -> 0-d,
    non-contiguous input rejection, and cupy input with device='cpu' rejection.
    Returns True on success (or if skipped for lack of CuPy)."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()   # raises if no usable CUDA device
    except Exception as e:
        print(f"  SKIP CuPy passthrough tests: cupy unavailable ({type(e).__name__}: {e})")
        return True

    all_ok = True
    d_xyz64 = cp.asarray(xyz)                       # fp64 device copy
    xyz32 = xyz.astype(np.float32)
    d_xyz32 = cp.asarray(xyz32)                     # fp32 device copy

    # (a) fp64 parity + (g) returned type is cupy.ndarray
    for name, pot in pots:
        ref = pot.potential(xyz, device='cuda', dtype=np.float64)   # NumPy path
        try:
            phi_d = pot.potential(d_xyz64, device='cuda')           # no dtype kwarg
            is_cp = isinstance(phi_d, cp.ndarray)
            dt_ok = phi_d.dtype == cp.float64
            val_ok = np.allclose(cp.asnumpy(phi_d), ref, rtol=1e-12, atol=0.0)
            ok = is_cp and dt_ok and val_ok
            print(f"  {'OK  ' if ok else 'FAIL'} {name:14s} cupy fp64 in -> cupy out : "
                  f"isinstance={is_cp} dtype={phi_d.dtype} allclose(1e-12)={val_ok}")
        except Exception as e:
            print(f"  FAIL {name:14s} cupy fp64 : {type(e).__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False

    # (b) fp32 parity, dtype inferred from typestr (NOT the fp64 default)
    for name, pot in pots:
        ref32 = pot.potential(xyz32, device='cuda', dtype=np.float32)  # NumPy fp32 path
        try:
            phi_d = pot.potential(d_xyz32, device='cuda')              # no dtype kwarg
            dt_ok = phi_d.dtype == cp.float32
            val_ok = np.allclose(cp.asnumpy(phi_d).astype(np.float64),
                                 ref32.astype(np.float64), rtol=1e-6,
                                 atol=1e-6 * float(np.max(np.abs(ref32))))
            ok = isinstance(phi_d, cp.ndarray) and dt_ok and val_ok
            print(f"  {'OK  ' if ok else 'FAIL'} {name:14s} cupy fp32 in -> cupy out : "
                  f"dtype={phi_d.dtype} (inferred, no dtype kwarg) allclose(1e-6)={val_ok}")
        except Exception as e:
            print(f"  FAIL {name:14s} cupy fp32 : {type(e).__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False

    nfw = pots[2][1]
    ref = nfw.potential(xyz, device='cuda', dtype=np.float64)

    # (c) explicit MATCHING dtype kwarg works; MISMATCHED raises TypeError
    try:
        phi_d = nfw.potential(d_xyz64, device='cuda', dtype=np.float64)
        ok = isinstance(phi_d, cp.ndarray) and phi_d.dtype == cp.float64 \
            and np.allclose(cp.asnumpy(phi_d), ref, rtol=1e-12, atol=0.0)
        print(f"  {'OK  ' if ok else 'FAIL'} NFW cupy fp64 + explicit dtype=float64 (match)")
    except Exception as e:
        print(f"  FAIL NFW cupy fp64 + explicit dtype=float64 : {type(e).__name__}: {e}")
        ok = False
    if not ok:
        all_ok = False
    try:
        nfw.potential(d_xyz64, device='cuda', dtype=np.float32)
        print("  FAIL NFW cupy fp64 + dtype=float32 mismatch : unexpectedly succeeded")
        all_ok = False
    except TypeError as e:
        print(f"  OK   NFW cupy fp64 + dtype=float32 mismatch raised TypeError: {e}")
    except Exception as e:
        print(f"  FAIL NFW cupy fp64 + dtype=float32 mismatch raised wrong type "
              f"{type(e).__name__}: {e}")
        all_ok = False

    # (d) single point (3,) cupy in -> 0-d cupy out
    d_single = cp.asarray(np.array([1.0, 0.5, 0.3]))
    ref_single = float(nfw.potential((1.0, 0.5, 0.3)))
    try:
        phi_s = nfw.potential(d_single, device='cuda')
        shape_ok = isinstance(phi_s, cp.ndarray) and phi_s.shape == ()
        err = abs(float(cp.asnumpy(phi_s)) - ref_single)
        val_ok = err <= 1e-12 * max(abs(ref_single), 1.0)
        ok = shape_ok and val_ok
        print(f"  {'OK  ' if ok else 'FAIL'} NFW cupy single point (3,) -> 0-d : "
              f"shape={phi_s.shape} |err|={err:.3e}")
    except Exception as e:
        print(f"  FAIL NFW cupy single point : {type(e).__name__}: {e}")
        ok = False
    if not ok:
        all_ok = False

    # (e) non-contiguous cupy input raises TypeError
    d_noncontig = cp.asarray(xyz)[:, ::-1]   # negative stride on axis 1
    try:
        nfw.potential(d_noncontig, device='cuda')
        print("  FAIL NFW cupy non-contiguous : unexpectedly succeeded")
        all_ok = False
    except TypeError as e:
        print(f"  OK   NFW cupy non-contiguous raised TypeError: {e}")
    except Exception as e:
        print(f"  FAIL NFW cupy non-contiguous raised wrong type {type(e).__name__}: {e}")
        all_ok = False

    # (f) cupy input with device='cpu' raises TypeError
    try:
        nfw.potential(d_xyz64, device='cpu')
        print("  FAIL NFW cupy device='cpu' : unexpectedly succeeded")
        all_ok = False
    except TypeError as e:
        print(f"  OK   NFW cupy device='cpu' raised TypeError: {e}")
    except Exception as e:
        print(f"  FAIL NFW cupy device='cpu' raised wrong type {type(e).__name__}: {e}")
        all_ok = False

    return all_ok


def _check_fd(name, pot, xyz, device, dtype, ref, op):
    """One (pot, device, dtype) combo for force/density: device path vs legacy ref.

    ref is the legacy CPU result (pot.force(xyz) -> (N,3), pot.density(xyz) -> (N,)).
    Tolerances: rtol 1e-12 (fp64) / 2e-6 (fp32) vs max|ref| -- except fp32 FORCE,
    which uses 3e-6.

    That exception is a real fp32 floor, not slack for a bug. NFW's dPhi/dr is the
    worst case at ~2e-6 pointwise relative near its Pade crossover, because the closed
    form there subtracts two nearly-equal O(1/r_s) terms. It is identical on
    serial/cpu/openmp/cuda, i.e. a property of the shared leaf math, not a backend diff.

    READ THIS BEFORE TIGHTENING IT. This sweep is a PARITY test and is a poor accuracy
    test, because it normalizes by max|ref| (taken at small r) and samples uniform(-5,5)^3,
    which lands in the narrow shell that matters only by accident. Measured for NFW force:
    9.6e-7 with the fp32 guard bug present vs 5.1e-7 with it fixed, against a 1.06e-6
    budget -- i.e. at 3e-6 this sweep does NOT discriminate the bug, and at 2e-6 it would
    discriminate it only by 1.4x on each side, which is a flake, not a gate. The accuracy
    of this leaf is therefore pinned where it can be measured properly, by
    nfw_fp32_crossover_test() below, which separates the two builds by 40x.

    HISTORY: this tolerance was 5e-6 and its comment blamed "~30 ULP" of inherent fp32
    rounding. That was wrong. nfw_eval's Pade crossovers were inherited from upstream,
    which is fp64-only, and were ~20x too small for fp32 -- costing 4.7e-4 in dPhi/dr.
    Fixed by nfw_pade_guard<T> in src/potential_analytic.h. A tolerance widened to fit a
    measurement, with a comment naming the potential responsible, is a bug report."""
    out = getattr(pot, op)(xyz, device=device, dtype=dtype)
    label = f"{name:14s} {op:8s} {device:6s} {dtype.__name__:8s}"
    if out.dtype != dtype:
        print(f"  FAIL {label} : dtype {out.dtype} != {dtype}")
        return False
    if out.shape != ref.shape:
        print(f"  FAIL {label} : shape {out.shape} != {ref.shape}")
        return False
    max_ref = float(np.max(np.abs(ref))) if ref.size else 1.0
    rel_tol = (3e-6 if op == "force" else 2e-6) if dtype == np.float32 else 1e-12
    abs_tol = rel_tol * max_ref
    err = float(np.max(np.abs(out.astype(np.float64) - ref)))
    ok = err <= abs_tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label} "
          f": max|ref|={max_ref:.3e}  |err|={err:.3e}  tol={abs_tol:.3e}")
    return ok


def nfw_fp32_crossover_test():
    """fp32 accuracy across NFW's Pade-vs-closed-form crossover in r/r_s.

    WHY THIS EXISTS. nfw_eval switches from the closed form to a Pade expansion below a
    threshold in r/r_s. Above it, dPhi/dr subtracts two nearly-equal O(1/r_s) terms, so the
    closed form loses ~eps/(r/r_s)^2 relative accuracy; below it the Pade truncation error
    grows. The crossover that balances those two scales as sqrt(eps), so it is ~20x higher
    in fp32 than fp64 -- and AGAMA's constants came from upstream, which has no fp32 path.
    Reusing them cost 4.7e-4 in dPhi/dr and 4.5e-2 in d2Phi/dr2 before nfw_pade_guard<T>.

    The main() parity sweep could not catch that, and this test exists because of how it
    failed: it samples uniform(-5,5)^3 with scaleRadius=1, so it lands in the narrow
    affected shell only by accident, and averages it away against max|ref| taken at small r.
    Here r/r_s is walked deliberately across the crossover on a log grid, and the error is
    POINTWISE relative -- NFW's Phi, |F| and rho are nonzero and monotone over this range,
    so no point can hide behind a larger one elsewhere.

    Reference is the legacy fp64 CPU path, so this measures the fp32 leaf itself rather
    than a backend difference; it is run on every device to confirm exactly that.

    NOT covered: d2Phi/dr2, whose fp32 error was the largest (4.5e-2). The device path
    exposes only potential/force/density (GPUEvalOp in interface_python.cpp), so the fp32
    hessian is unreachable from Python today and the guard fix for it is preventive. If an
    fp32 hessian path is ever added (fp32 variational/Lyapunov integration, fp32 action
    finders), extend this test to it.
    """
    print("\n== NFW fp32 accuracy across the Pade crossover in r/r_s ==")
    pot = agama.Potential(type='NFW', mass=1.0, scaleRadius=1.0)
    rrel = np.logspace(-3, 1, 600)                  # spans both sides of every threshold
    u = np.array([0.4759473, -0.6235637, 0.6199788])
    u /= np.linalg.norm(u)                          # off-axis; NFW is spherical anyway
    xyz = rrel[:, None] * u[None, :]

    # Measured, identical on serial/cpu/cuda, with the guard fix vs with it reverted to
    # the fp64 thresholds (verified by rebuilding both ways, 2026-07-24):
    #                fixed      pre-fix                 tol
    #   potential    1.7e-7     3.2e-6  @r/rs=0.0164    1e-6   -> 5.8x margin / FAILs 3.2x
    #   force        2.0e-6     4.0e-4  @r/rs=0.0164    1e-5   -> 5.0x margin / FAILs 40x
    #   density      2.9e-7     2.9e-7  (no guard, unaffected)  1e-6
    # The pre-fix peaks land exactly on the old crossover, and the post-fix peaks land on
    # the new one (r/rs ~ 0.29-0.32) -- that co-location is the signature to look for if
    # these numbers ever move.
    TOL = {'potential': 1e-6, 'force': 1e-5, 'density': 1e-6}
    all_ok = True
    for device in ("serial", "cpu", "cuda"):
        for op in ("potential", "force", "density"):
            try:
                ref = np.asarray(getattr(pot, op)(xyz), dtype=float)      # legacy fp64 CPU
                got = np.asarray(getattr(pot, op)(xyz, device=device,
                                                  dtype=np.float32), dtype=float)
                if ref.ndim == 2:            # force: compare the radial magnitude
                    ref, got = (np.linalg.norm(ref, axis=1), np.linalg.norm(got, axis=1))
                rel = np.abs(got - ref) / np.abs(ref)
                worst = float(np.max(rel))
                ok = worst <= TOL[op]
                print(f"  {'OK  ' if ok else 'FAIL'} NFW {op:9s} {device:6s} float32 : "
                      f"max pointwise rel={worst:.3e} @r/rs={rrel[int(np.argmax(rel))]:.4g}"
                      f"  tol={TOL[op]:.0e}")
                if not ok:
                    all_ok = False
            except Exception as e:
                print(f"  FAIL NFW {op:9s} {device:6s} float32 : {type(e).__name__}: {e}")
                all_ok = False
    return all_ok


def force_density_cupy_tests(targets, xyz):
    """CuPy in -> CuPy out for force and density: type, shape ((N,3) for force,
    (N,) for density), and parity vs the NumPy device='cuda' path, fp64 + fp32."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        print(f"  SKIP force/density CuPy tests: cupy unavailable ({type(e).__name__}: {e})")
        return True

    all_ok = True
    N = xyz.shape[0]
    for dtype, tag, rtol in ((np.float64, 'fp64', 1e-12), (np.float32, 'fp32', 2e-6)):
        xyz_t = xyz.astype(dtype)
        d_xyz = cp.asarray(xyz_t)
        for name, pot in targets:
            for op, want_shape in (("force", (N, 3)), ("density", (N,))):
                ref = getattr(pot, op)(xyz_t, device='cuda', dtype=dtype)  # NumPy path
                try:
                    out = getattr(pot, op)(d_xyz, device='cuda')           # dtype inferred
                    is_cp = isinstance(out, cp.ndarray)
                    shape_ok = out.shape == want_shape
                    dt_ok = out.dtype == dtype
                    max_ref = float(np.max(np.abs(ref)))
                    err = float(np.max(np.abs(cp.asnumpy(out).astype(np.float64)
                                              - ref.astype(np.float64))))
                    val_ok = err <= rtol * max_ref
                    ok = is_cp and shape_ok and dt_ok and val_ok
                    print(f"  {'OK  ' if ok else 'FAIL'} {name:14s} {op:8s} cupy {tag} : "
                          f"isinstance={is_cp} shape={out.shape} dtype={out.dtype} "
                          f"|err|={err:.3e} (tol {rtol*max_ref:.3e})")
                except Exception as e:
                    print(f"  FAIL {name:14s} {op:8s} cupy {tag} : {type(e).__name__}: {e}")
                    ok = False
                if not ok:
                    all_ok = False
    return all_ok


# ============================================================================
# Tier 2 commit 5b: Multipole on the GPU, through the descriptor path.
# ============================================================================

def _mw_multipole():
    """A GalPot-flavoured MW-like Multipole: an exponential/Sersic-ish disc plus a
    flattened NFW-like halo, expanded to lmax=mmax=8 -- the shape every realistic
    Milky Way model takes, and the one the MultipoleInterp2d branch serves."""
    disc = agama.Density(type='Disk', surfaceDensity=1.0, scaleRadius=3.0,
                         scaleHeight=0.3)
    halo = agama.Density(type='Spheroid', densityNorm=0.5, gamma=1.0, beta=3.0,
                         scaleRadius=15.0, axisRatioZ=0.8)
    return agama.Potential(type='Multipole', density=agama.Density(disc, halo),
                           lmax=8, mmax=8, gridSizeR=30, rmin=0.05, rmax=200.0)


def multipole_targets():
    """Multipole models covering the branch/symmetry space that matters:
    spherical (lmax=0, the Interp1d branch), triaxial lmax=2 (Interp1d plus
    sphHarmTransformInverseDeriv's optimized {0,0}/{2,0}/{2,2} shortcut), triaxial
    lmax=8 (Interp2d, mmax>0 so dPhi/dphi != 0 -- the component a 2-component
    Cyl->Car conversion would silently drop), the MW-like model above, and a
    composite and a Shifted chain wrapping one."""
    sph = agama.Potential(type='Multipole', density='Spheroid', gamma=1, beta=4,
                          scaleRadius=1, lmax=0, mmax=0, gridSizeR=25)
    tri2 = agama.Potential(type='Multipole', density='Spheroid', gamma=1, beta=4,
                           scaleRadius=1, axisRatioY=0.8, axisRatioZ=0.6,
                           lmax=2, mmax=2, gridSizeR=25)
    tri8 = agama.Potential(type='Multipole', density='Spheroid', gamma=1, beta=4,
                           scaleRadius=1, axisRatioY=0.8, axisRatioZ=0.6,
                           lmax=8, mmax=8, gridSizeR=25)
    mw = _mw_multipole()
    nfw = agama.Potential(type='NFW', mass=10.0, scaleRadius=5.0)
    comp = agama.Potential(tri8, nfw)
    shifted = agama.Potential(type='Multipole', density='Spheroid', gamma=1, beta=4,
                              scaleRadius=1, axisRatioY=0.8, axisRatioZ=0.6,
                              lmax=8, mmax=8, gridSizeR=25, center='0.3,-0.2,0.1')
    return [
        ("Mp sph l0",   sph),
        ("Mp tri l2",   tri2),
        ("Mp tri l8",   tri8),
        ("Mp MW l8",    mw),
        ("Mp+NFW comp", comp),
        ("Mp shifted",  shifted),
    ]


def _pointwise_rel(a, b):
    """Max POINTWISE relative difference. Not normalized by max|ref|: that averages
    defects away, which is how a 4.7e-4 error in NFW's dPhi/dr once hid under a 5e-6
    budget (see _check_fd). Elements where the reference underflows to zero are
    compared in absolute terms against the largest |ref| instead."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    scale = np.abs(b)
    floor = np.max(scale) * 1e-300 if scale.size else 1.0
    scale = np.where(scale > 0, scale, max(floor, 1e-300))
    return float(np.max(np.abs(a - b) / scale))


def multipole_parity_tests(targets, xyz):
    """potential / force / density on a Multipole across 4 devices x 2 precisions,
    against the legacy no-kwarg CPU path, with POINTWISE relative tolerances.

    fp64 tolerance 1e-11: the descriptor path re-derives R = sqrt(x^2+y^2) inside the
    kernel, and GCC/nvcc may fuse that multiply-add differently from coord.cpp's copy,
    which is worth a few ULP and, through a steep power-law asymptote, up to ~1e-14
    pointwise. The C++ gate (tests/test_gpu_policy.cpp [T2B]) pins the exact figure at
    8 ULP and pins everything downstream of that conversion at ZERO bitwise
    differences, so this tolerance is not where the real accuracy claim lives.

    THE fp32 REFERENCE IS THE fp32 CPU RESULT, not the fp64 one. This function asks
    "does the GPU backend reproduce the CPU backend at the same precision"; asking
    instead "how close is fp32 to fp64" conflates a backend difference with an
    accuracy property and would report an intrinsic fp32 conditioning limit as a GPU
    bug. Measured while writing this: comparing fp32 density against fp64 gives up to
    0.13 POINTWISE relative on a triaxial r^-4 halo -- identical on
    serial/cpu/openmp/cuda, i.e. a property of the shared leaf math (the in-grid
    density is a cylindrical Laplacian, a sum of four cancelling second derivatives).
    That number is real and belongs in multipole_fp32_budget() below, where it is
    reported and explained, not hidden in a widened parity tolerance."""
    all_ok = True
    for name, pot in targets:
        refs = {
            np.float64: {
                "potential": np.atleast_1d(pot.potential(xyz)),
                "force":     np.atleast_2d(pot.force(xyz)),
                "density":   np.atleast_1d(pot.density(xyz)),
            },
            np.float32: {
                op: getattr(pot, op)(xyz.astype(np.float32), device='cpu',
                                     dtype=np.float32)
                for op in ("potential", "force", "density")
            },
        }
        for op in ("potential", "force", "density"):
            # fp32 DENSITY gets a looser cross-backend tolerance than fp32 potential
            # and force, and the reason is conditioning, not a backend difference:
            # inside the radial grid the density IS the cylindrical Laplacian, whose
            # own fp32-vs-fp64 error is 5e-2 (see multipole_fp32_budget). Demanding
            # 1e-5 agreement between two backends on a quantity that is only good to
            # 5e-2 in that precision is asking the cancellation to round identically
            # under g++ and nvcc. Measured worst case: 1.5e-4, on Shifted(Multipole),
            # where the modifier transform adds one more fp32 rounding before the
            # cancelling sum. It is the same on every device+dtype combination that
            # keeps the transform, i.e. a property of the arithmetic, not of CUDA.
            fp32tol = 1e-3 if op == "density" else 1e-5
            for device in ("cpu", "openmp", "serial", "cuda"):
                for dtype, tol in ((np.float64, 1e-11), (np.float32, fp32tol)):
                    label = f"{name:13s} {op:9s} {device:6s} {dtype.__name__:8s}"
                    try:
                        arg = xyz if dtype == np.float64 else xyz.astype(np.float32)
                        out = getattr(pot, op)(arg, device=device, dtype=dtype)
                        if out.dtype != dtype:
                            print(f"  FAIL {label} : dtype {out.dtype} != {dtype}")
                            all_ok = False
                            continue
                        err = _pointwise_rel(out, refs[dtype][op])
                        ok = err <= tol
                        print(f"  {'OK  ' if ok else 'FAIL'} {label} : "
                              f"max pointwise rel = {err:.3e}  tol = {tol:.1e}")
                        if not ok:
                            all_ok = False
                    except Exception as e:
                        print(f"  FAIL {label} : {type(e).__name__}: {e}")
                        all_ok = False
    return all_ok


def multipole_fp32_budget():
    """THE fp32 ACCURACY BUDGET -- the one thing gating fp32 as a default for Tier 2.

    fp32 vs fp64 on the SAME device path, POINTWISE relative, over a log-spaced radial
    sweep that deliberately straddles the radial grid: inside it the MultipoleInterp2d
    branch runs and never calls sphHarmArray at all (so it entirely avoids the ~2e-2
    near-pole derivative error that fp32 sphHarmArray carries); outside it the
    PowerLawMultipole asymptotes DO call sphHarmArray, so both regimes have to be
    sampled or the reported number is not the budget.

    Reported per quantity and per regime. Pointwise, NOT normalized by max|ref| --
    normalizing by the maximum hides exactly the kind of localized defect this is
    looking for."""
    print("== Multipole fp32-vs-fp64 accuracy budget (pointwise relative) ==")
    models = [
        ("MW-like l8 (disc+halo)", _mw_multipole(), 0.05, 200.0),
        ("triaxial r^-4 halo l8",
         agama.Potential(type='Multipole', density='Spheroid', gamma=1, beta=4,
                         scaleRadius=1, axisRatioY=0.8, axisRatioZ=0.6,
                         lmax=8, mmax=8, gridSizeR=25, rmin=0.02, rmax=100.0),
         0.02, 100.0),
    ]
    all_ok = True
    # a few directions, including near-pole (theta -> 0) and in-plane, at two azimuths
    dirs = []
    for theta in (0.02, 0.3, np.pi / 4, np.pi / 2 - 0.02, np.pi / 2):
        for phi in (0.0, 0.9):
            dirs.append((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                         np.cos(theta)))
    dirs = np.array(dirs)
    # MEASURED budgets, per QUANTITY rather than per regime, because the conditioning
    # is a property of what is being computed:
    #   potential  ~1e-6  -- a plain fp32 spline/series evaluation, near the eps floor;
    #   force      ~1e-3  -- one differentiation of the same interpolant;
    #   density    ~5e-3  -- inside the radial grid this is the cylindrical LAPLACIAN,
    #                        i.e. a sum of four second derivatives that cancel, which
    #                        is the worst-conditioned quantity in the whole path (the
    #                        CPU code even has an explicit eps^(2/3) roundoff cutoff
    #                        for exactly this reason). Outside the grid the closed-form
    #                        PowerLaw density is used instead and the error drops two
    #                        orders of magnitude, which is visible in the table.
    # Headroom over the measured worst case is ~1.5-3x: enough that ordinary drift does
    # not flake the suite, tight enough that a real regression in the spline or harmonic
    # layer trips it. These are the numbers to argue about when deciding whether fp32
    # becomes a default; they are NOT tolerances chosen to make a test pass.
    # DENSITY inside the radial grid is the one quantity fp32 does NOT deliver
    # pointwise, and it is worth being explicit about because it is the answer to
    # "can fp32 be the default": measured up to 1.4e-1 on a triaxial r^-4 halo, five
    # orders of magnitude worse than the potential on the same points. The mechanism
    # is not the spline layer -- it is that rho inside the grid is obtained as the
    # cylindrical LAPLACIAN of Phi, i.e. d2Phi/dR2 + dPhi/dR/R + d2Phi/dz2 +
    # d2Phi/dphi2/R2, four terms that cancel to a small residual for a steeply
    # falling profile. The CPU path itself declares the result zero below an
    # eps^(2/3) relative threshold for exactly this reason; in fp32 that threshold is
    # 2.4e-5 instead of 4e-11, so everything just above it keeps O(1) relative error.
    # Outside the grid the closed-form PowerLaw density is used instead and the error
    # drops by four orders of magnitude, which the table below shows directly.
    #
    # CONCLUSION for the fp32-as-default question: potential (1e-6) and force (1e-3)
    # are fine; in-grid density is not, and an fp32 density consumer must either use
    # fp64 or take the density from a Density object rather than from a potential's
    # Laplacian. That is a property of the CPU algorithm, identical on every backend,
    # not a GPU defect.
    budget = {"potential": 1e-6, "force": 2e-3, "density": 2e-1}
    worst = {}
    for mname, pot, rgrid_lo, rgrid_hi in models:
      print(f"  -- {mname} (radial grid {rgrid_lo:g} .. {rgrid_hi:g})")
      regimes = [
        ("inner asympt (r < rmin)", np.logspace(np.log10(rgrid_lo) - 3,
                                                np.log10(rgrid_lo) - 0.05, 40)),
        ("in grid (Interp2d)",      np.logspace(np.log10(rgrid_lo) + 0.05,
                                                np.log10(rgrid_hi) - 0.05, 120)),
        ("outer asympt (r > rmax)", np.logspace(np.log10(rgrid_hi) + 0.05,
                                                np.log10(rgrid_hi) + 3, 40)),
      ]
      for regime, radii in regimes:
        xyz = (radii[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
        for op in ("potential", "force", "density"):
            ref = getattr(pot, op)(xyz, device='cpu', dtype=np.float64)
            f32 = getattr(pot, op)(xyz.astype(np.float32), device='cpu', dtype=np.float32)
            err = _pointwise_rel(f32, ref)
            worst[(mname, regime, op)] = err
            # locate the worst point, so the number is actionable
            r64 = np.asarray(ref, dtype=np.float64).reshape(len(radii), len(dirs), -1)
            f64 = np.asarray(f32, dtype=np.float64).reshape(len(radii), len(dirs), -1)
            rel = np.abs(f64 - r64) / np.maximum(np.abs(r64), 1e-300)
            i, j = np.unravel_index(int(np.argmax(rel.max(axis=2))), rel.shape[:2])
            print(f"     {regime:26s} {op:9s} : max pointwise rel = {err:.3e}  "
                  f"(worst at r={radii[i]:.4g}, theta={np.arccos(dirs[j][2]):.3f}, "
                  f"budget {budget[op]:.0e})")
    for (mname, regime, op), err in worst.items():
        if err > budget[op]:
            print(f"  FAIL {mname} / {regime} / {op}: {err:.3e} exceeds budget "
                  f"{budget[op]:.1e}")
            all_ok = False
    print(f"  {'OK  ' if all_ok else 'FAIL'} fp32 budget respected in all "
          f"{len(worst)} (model, regime, quantity) cells")
    return all_ok


def multipole_orbit_test():
    """orbit(..., device='cuda') on a Multipole-based MW potential: currently FAILS
    CLOSED to the CPU integrator, and this test pins that contract.

    Batch potential/force/density DO run a Multipole on the GPU. The orbit kernels do
    not yet, for a measured COMPILE-TIME reason documented in orbit_gpu.cpp
    (prepareOrbitBatch): giving the DOP853/DPRKN8 kernels the Multipole scratch also
    inlines the whole four-branch evaluator into a body that already saturates the
    register file, and one ptxas invocation was still running after 27 minutes at
    order 32 / 890 s at order 8, against ~460 s for the whole TU as it ships.

    So the contract to hold here is: either the call raises NotImplementedError, or it
    silently gives the right answer -- never a wrong answer, and never a crash. What
    must NOT happen is a Multipole reaching a kernel with no scratch for it."""
    print("== orbit(device='cuda') on a Multipole-based MW potential ==")
    pot = _mw_multipole()
    rng = np.random.default_rng(7)
    N = 256
    ic = np.empty((N, 6))
    ic[:, 0] = rng.uniform(4.0, 12.0, N)
    ic[:, 1] = rng.uniform(-2.0, 2.0, N)
    ic[:, 2] = rng.uniform(-1.0, 1.0, N)
    vc = np.sqrt(np.maximum(1e-6, -np.sum(ic[:, :3] * pot.force(ic[:, :3]), axis=1)))
    ic[:, 3] = -ic[:, 1] / np.hypot(ic[:, 0], ic[:, 1]) * vc
    ic[:, 4] = ic[:, 0] / np.hypot(ic[:, 0], ic[:, 1]) * vc
    ic[:, 5] = rng.uniform(-0.1, 0.1, N) * vc
    T, NS = 40.0, 33
    all_ok = True

    def energy(w, dt):
        # Reference Phi always in fp64: we are measuring the integrator's drift,
        # not the potential evaluation's precision.
        w64 = np.asarray(w, dtype=np.float64)
        return pot.potential(w64[:, :3]) + 0.5 * np.sum(w64[:, 3:] ** 2, axis=1)

    try:
        tc, wc = agama.orbit(potential=pot, ic=ic, time=T, trajsize=NS,
                             separateTime=True)
        wc = np.asarray(wc)
    except Exception as e:
        print(f"  FAIL CPU reference orbit: {type(e).__name__}: {e}")
        return False

    # All four combinations, because the compile-time problem that gates this is
    # per-KERNEL: method is a host-side switch (DOP853 vs DPRKN8) and dtype selects
    # another instantiation, so c658992 already has four REG:255 kernels in
    # orbit_gpu.cpp. A fix that opens the gate for one of them has not opened it.
    # Trajectory tolerances are loose for fp32 on purpose: these are ADAPTIVE
    # integrators, so the two backends take different step sequences and the
    # trajectory comparison is a smoke test. Energy conservation is the invariant
    # that has to hold regardless, and it is the tighter gate here.
    cases = [
        ('dop853', np.float64, 1e-4, 1e-6),
        ('dop853', np.float32, 5e-2, 5e-3),
        ('dprkn8', np.float64, 1e-4, 1e-6),
        ('dprkn8', np.float32, 5e-2, 5e-3),
    ]
    ran_any = False
    for method, dtype, tol_traj, tol_E in cases:
        label = f"{method} {dtype.__name__}"
        try:
            tg, wg = agama.orbit(potential=pot, ic=ic, time=T, trajsize=NS,
                                 separateTime=True, device='cuda',
                                 method=method, dtype=dtype)
        except NotImplementedError as e:
            # The documented fail-closed state. Acceptable, but say so per case so
            # that a PARTIAL fix (one kernel opened, others not) is visible rather
            # than looking like a uniform pass.
            print(f"  OK   orbit {label:16s} falls back cleanly: "
                  f"NotImplementedError: {str(e)[:78]}")
            continue
        except RuntimeError as e:
            if "without CUDA support" in str(e):
                print("  SKIP orbit device='cuda': library built with HAVE_CUDA=0")
                return True
            print(f"  FAIL orbit {label}: RuntimeError: {e}")
            all_ok = False
            continue
        except Exception as e:
            print(f"  FAIL orbit {label}: {type(e).__name__}: {e}")
            all_ok = False
            continue
        ran_any = True
        wg = np.asarray(wg, dtype=np.float64)
        if wg.shape != wc.shape:
            print(f"  FAIL orbit {label}: shape {wg.shape} != CPU {wc.shape}")
            all_ok = False
            continue
        finite = bool(np.all(np.isfinite(wg)))
        scale = np.max(np.abs(wc), axis=2, keepdims=True)
        rel = float(np.max(np.abs(wg - wc) / np.maximum(scale, 1e-300)))
        okT = finite and rel <= tol_traj
        print(f"  {'OK  ' if okT else 'FAIL'} orbit {label:16s} vs cpu trajectories "
              f"(N={N}, T={T}, {NS} samples): max rel = {rel:.3e}  tol = {tol_traj:.1e}"
              + ("" if finite else "   NON-FINITE VALUES PRESENT"))
        all_ok = all_ok and okT
        e0 = energy(wg[:, 0, :], dtype)
        dE = np.abs((energy(wg[:, -1, :], dtype) - e0) / e0)
        okE = float(np.max(dE)) <= tol_E and finite
        print(f"  {'OK  ' if okE else 'FAIL'} orbit {label:16s} energy conservation: "
              f"max |dE/E| = {float(np.max(dE)):.3e}  tol = {tol_E:.1e}")
        all_ok = all_ok and okE

    if not ran_any:
        print("  INFO no Multipole orbit case runs on the GPU yet -- the fail-closed "
              "gate in prepareOrbitBatch is still in place (compile time, see "
              "orbit_gpu.cpp). This test becomes the acceptance gate when it opens.")
    return all_ok


def multipole_near_axis_test():
    """The near-axis regime, where CPU-vs-GPU parity is EXACT in one branch and
    unachievable-by-construction in the other. Pins the first and documents the
    second, so that neither gets silently traded away.

    IN-GRID (MultipoleInterp2d): GPU must equal CPU BIT FOR BIT on and near the z
    axis, at every R/z. That branch never calls sphHarmArray -- the l-dependence
    lives in the tau grid of the 2D spline -- so there is nothing ill-conditioned
    in it and any drift here is a real regression. Asserted at 0.0, not at a
    tolerance.

    ON THE POWERLAW ASYMPTOTES (r outside the radial grid), F_R near the axis is
    NOT parity-testable and must not be added to any sweep:
      * math_sphharm.h:292 forms dPlm = (l*ct*Plm - (l+m)*Plm1)/st, which for m=0
        subtracts two O(1) quantities to get a result of order st^2 -- a ~13-digit
        cancellation at st=1e-6. One ULP of host-vs-device codegen difference
        becomes ~2e-4 relative, and sphToCylDerivs carries it into F_R undiluted.
      * The CPU is wrong there too, by MORE than the GPU-CPU gap over most of the
        window: F_R/R must tend to a constant as R->0, and the CPU's own value
        drifts 1.65e-4 at R/z=1e-6, 1.04e-3 at 3e-7, and 59% at 1.778e-8, where
        the GPU agrees with the CPU to ~1e-10.
      * Absolute error stays <=3.4e-9 of |F| throughout, because F_R -> 0 on the
        axis: a blowing-up relative error is a vanishing absolute one.
    So a tolerance widened to accommodate it would be accommodating the CPU's
    error, and "fixing" the GPU to match the CPU there fixes nothing physical.
    Full measured record and the upstream EPS discussion in findings.md.
    """
    print("== Multipole near the z axis: exact in-grid, excluded on the asymptotes ==")
    all_ok = True
    pot = agama.Potential(type='Multipole', density='Dehnen', mass=1, scaleRadius=1,
                          gamma=1, axisRatioY=0.5, axisRatioZ=0.2, gridSizeR=40,
                          lmax=8, mmax=8)
    z_in = 1e-2      # comfortably inside the radial grid for this model
    for q in (1e-2, 1e-4, 1e-6, 1e-8, 0.0):
        xyz = np.array([[q * z_in, 0.0, z_in]])
        ref = pot.force(xyz, device='cpu', dtype=np.float64)
        try:
            got = pot.force(xyz, device='cuda', dtype=np.float64)
        except RuntimeError as e:
            if "without CUDA support" in str(e):
                print("  SKIP near-axis in-grid: library built with HAVE_CUDA=0")
                return True
            raise
        exact = bool(np.all(got == ref))
        print(f"  {'OK  ' if exact else 'FAIL'} in-grid R/z={q:<8.1e} force GPU==CPU "
              f"bitwise: {exact}"
              + ("" if exact else f"  maxdiff={float(np.max(np.abs(got-ref))):.3e}"))
        all_ok = all_ok and exact
    return all_ok


def multipole_fail_closed_test():
    """A Multipole shape the device path cannot represent must raise
    NotImplementedError (and leave the CPU path working), never produce a guessed
    answer. The order cap is math::LEGENDRE_MMAX = 32; a Composite with more
    Multipoles than GPU_POT_MAX_MULTIPOLE = 4 likewise falls back."""
    print("== Multipole capability boundary on device='cuda' ==")
    all_ok = True
    xyz = np.array([[1.0, 0.5, 0.3], [3.0, -1.0, 0.7]])
    mps = [agama.Potential(type='Multipole', density='Spheroid', gamma=1, beta=4,
                           scaleRadius=1 + 0.1 * i, axisRatioY=0.8, axisRatioZ=0.6,
                           lmax=8, mmax=8, gridSizeR=25) for i in range(5)]
    # (a) MORE Multipoles than GPU_POT_MAX_MULTIPOLE still WORKS through batch
    #     evaluation, and that is not an accident: try_dispatch recurses into a
    #     Composite MEMBER BY MEMBER, so each Multipole gets its own single-term
    #     descriptor and the 4-slot side table never fills. The cap binds only on
    #     consumers that need ONE descriptor for the whole potential -- the orbit
    #     kernel -- and that case is gated in C++ (tests/test_gpu_policy.cpp [T2B],
    #     which calls buildGpuPotDesc on a 5-Multipole composite and requires it to
    #     be rejected rather than truncated).
    five = agama.Potential(*mps)
    try:
        got = five.potential(xyz, device='cuda')
        ref = five.potential(xyz)
        err = _pointwise_rel(got, ref)
        ok = err <= 1e-11
        print(f"  {'OK  ' if ok else 'FAIL'} 5-Multipole composite on device='cuda' "
              f"(per-member dispatch, side table never exceeds {5} of "
              f"GPU_POT_MAX_MULTIPOLE): max pointwise rel = {err:.3e}")
        all_ok = all_ok and ok
    except RuntimeError as e:
        if "without CUDA support" in str(e):
            print("  SKIP 5-Multipole composite: library built with HAVE_CUDA=0")
        else:
            print(f"  FAIL 5-Multipole composite: RuntimeError: {e}")
            all_ok = False
    except Exception as e:
        print(f"  FAIL 5-Multipole composite: {type(e).__name__}: {e}")
        all_ok = False
    # (b) THE ORDER CAP ITSELF. Two things must hold: the above-cap object is
    #     refused, AND the message says WHY. The bare form ("not supported for
    #     potential type 'Multipole'") is actively misleading here, because
    #     Multipole IS supported -- capability is a property of the instance, not
    #     the type -- so a user reading it would wrongly conclude the whole type is
    #     off-limits rather than lowering lmax. See unsupportedGPUPotentialReason().
    #
    #     TRAP: Multipole::create does NOT hand back the order you asked for.
    #     restrictSphHarmCoefs trims all-zero trailing harmonics and silently
    #     LOWERS the order, so a nearly-spherical density asked for lmax=33 comes
    #     back well below the cap and this test would pass vacuously. Hence the
    #     deliberately lumpy density below (an off-centre Plummer inside a triaxial
    #     Dehnen breaks every symmetry, including m<0), and hence the achieved order
    #     is read BACK out of export() and asserted, not assumed.
    over = agama.Potential(
        type='Multipole',
        density=agama.Density(
            agama.Density(type='Dehnen', mass=1.0, scaleRadius=1.0, gamma=1.0,
                          axisRatioY=0.7, axisRatioZ=0.5),
            agama.Density(type='Plummer', mass=0.3, scaleRadius=0.6,
                          center=[0.8, 0.5, 0.3])),
        lmax=33, mmax=33, gridSizeR=25)
    achieved = None
    with tempfile.NamedTemporaryFile(suffix='.coef', delete=False) as fh:
        coefpath = fh.name
    try:
        over.export(coefpath)
        with open(coefpath) as f:
            for line in f:
                if line.startswith('lmax='):
                    achieved = int(line.split('=')[1])
                    break
    finally:
        os.unlink(coefpath)
    if achieved is None or achieved <= 32:
        print(f"  FAIL above-cap Multipole: achieved lmax={achieved}, which is NOT "
              f"above the cap of 32 -- restrictSphHarmCoefs lowered the order, so this "
              f"case would have tested nothing. Make the density lumpier.")
        all_ok = False
    else:
        try:
            over.potential(xyz, device='cuda')
            print(f"  FAIL Multipole lmax={achieved} was ACCEPTED on device='cuda' "
                  f"(above math::LEGENDRE_MMAX = 32)")
            all_ok = False
        except NotImplementedError as e:
            msg = str(e)
            # The message must let the user act: it has to name the cap, not just
            # the type.  '32' and the achieved order are both required so that a
            # future refactor cannot degrade this back to the generic form.
            informative = '32' in msg and str(achieved) in msg
            print(f"  {'OK  ' if informative else 'FAIL'} Multipole lmax={achieved} -> "
                  f"NotImplementedError naming the order cap: {msg[:160]}")
            all_ok = all_ok and informative
        except RuntimeError as e:
            if "without CUDA support" in str(e):
                print("  SKIP above-cap Multipole: library built with HAVE_CUDA=0")
            else:
                print(f"  FAIL above-cap Multipole: RuntimeError: {e}")
                all_ok = False
        except Exception as e:
            print(f"  FAIL above-cap Multipole: {type(e).__name__}: {e}")
            all_ok = False
        # ... and the above-cap object still evaluates on the CPU, at full order
        try:
            v = over.potential(xyz)
            ok = np.all(np.isfinite(v))
            print(f"  {'OK  ' if ok else 'FAIL'} above-cap Multipole still evaluates on "
                  f"the CPU path (the cap is a DEVICE limit, not a library limit)")
            all_ok = all_ok and ok
        except Exception as e:
            print(f"  FAIL above-cap Multipole CPU path: {type(e).__name__}: {e}")
            all_ok = False
    # (c) THE DESCRIPTOR-CAPACITY CAPS, and the only shape that can reach them.
    #     A bare Composite of Multipoles is unlimited on the GPU (see (a)), so the
    #     caps bind only where ONE descriptor must hold the whole potential: a
    #     modifier wrapping a composite. That is reachable from Python only through
    #     the `potential=<Potential object>` keyword form -- passing dicts
    #     positionally alongside center= is a constructor TypeError, which is easy
    #     to mistake for the code path being unreachable. Both caps must produce an
    #     ACTIONABLE reason, since every member is individually GPU-capable and the
    #     bare type name explains nothing.
    for count, limit_name, ok_count in ((5, 'GPU_POT_MAX_MULTIPOLE', 4),
                                        (17, 'GPU_POT_DESC_MAX_TERMS', 16)):
        if limit_name == 'GPU_POT_MAX_MULTIPOLE':
            members = [agama.Potential(type='Multipole', density='Spheroid', gamma=1,
                                       beta=4, scaleRadius=1 + 0.1 * i, axisRatioY=0.8,
                                       axisRatioZ=0.6, lmax=4, mmax=4, gridSizeR=20)
                       for i in range(count)]
        else:
            members = [agama.Potential(type='NFW', mass=10.0 + i, scaleRadius=5.0 + i)
                       for i in range(count)]
        try:
            over_cap = agama.Potential(potential=agama.Potential(*members),
                                       center=[0.1, 0.2, 0.3])
            under_cap = agama.Potential(potential=agama.Potential(*members[:ok_count]),
                                        center=[0.1, 0.2, 0.3])
        except Exception as e:
            print(f"  FAIL modifier-over-composite construction ({limit_name}): "
                  f"{type(e).__name__}: {e}")
            all_ok = False
            continue
        try:
            under_cap.potential(xyz, device='cuda')
            print(f"  OK   modifier over composite of {ok_count} runs on the GPU "
                  f"(at the {limit_name} limit, not over it)")
        except NotImplementedError as e:
            print(f"  FAIL modifier over composite of {ok_count} was REFUSED, but it is "
                  f"within {limit_name}: {e}")
            all_ok = False
        except RuntimeError as e:
            if "without CUDA support" in str(e):
                print(f"  SKIP {limit_name}: library built with HAVE_CUDA=0")
                continue
            print(f"  FAIL {limit_name} under-cap case: RuntimeError: {e}")
            all_ok = False
            continue
        try:
            over_cap.potential(xyz, device='cuda')
            print(f"  FAIL modifier over composite of {count} was ACCEPTED, exceeding "
                  f"{limit_name}")
            all_ok = False
        except NotImplementedError as e:
            msg = str(e)
            # The reason must reach THROUGH the modifier wrapper. It previously did
            # not: unsupportedGPUPotentialReason recursed into Composite but not
            # into the four modifier wrappers, so this -- the one shape that can
            # actually exhaust the descriptor -- got no explanation at all.
            informative = limit_name in msg and str(count) in msg
            print(f"  {'OK  ' if informative else 'FAIL'} modifier over composite of "
                  f"{count} -> reason names {limit_name}: {msg[-150:]}")
            all_ok = all_ok and informative
        except Exception as e:
            print(f"  FAIL {limit_name} over-cap case: {type(e).__name__}: {e}")
            all_ok = False
    # (d) the OTHER Tier 2 expansion, CylSpline, must still be refused by name --
    #     adding GPU_POT_MULTIPOLE must not have widened capability to BFEs in general
    cs = agama.Potential(type='CylSpline', density='Disk', surfaceDensity=1.0,
                         scaleRadius=2.0, scaleHeight=0.3, mmax=0,
                         gridSizeR=20, gridSizez=20)
    try:
        cs.potential(xyz, device='cuda')
        print("  FAIL CylSpline was ACCEPTED on device='cuda' (capability leaked)")
        all_ok = False
    except NotImplementedError as e:
        named = 'CylSpline' in str(e)
        print(f"  {'OK  ' if named else 'FAIL'} CylSpline -> NotImplementedError, names "
              f"the unsupported type: {str(e)[:100]}")
        all_ok = all_ok and named
    except RuntimeError as e:
        if "without CUDA support" in str(e):
            print("  SKIP CylSpline: library built with HAVE_CUDA=0")
        else:
            print(f"  FAIL CylSpline: RuntimeError: {e}")
            all_ok = False
    except Exception as e:
        print(f"  FAIL CylSpline: {type(e).__name__}: {e}")
        all_ok = False
    # ... and the CPU path for the refused object is unaffected
    try:
        v = cs.potential(xyz)
        ok = np.all(np.isfinite(v))
        print(f"  {'OK  ' if ok else 'FAIL'} refused object still evaluates on the CPU "
              f"path: {v}")
        all_ok = all_ok and ok
    except Exception as e:
        print(f"  FAIL CPU fallback for CylSpline: {type(e).__name__}: {e}")
        all_ok = False
    return all_ok


def main():
    rng = np.random.default_rng(42)
    N = 1024
    xyz = rng.uniform(-5.0, 5.0, size=(N, 3))
    # Multipole point set: the sweep above is centred on unit-scale analytic
    # potentials; the Multipole models use grids out to r ~ 200, and a uniform cube
    # would put most points in one branch. Log-spaced radii x scattered directions
    # instead, so all four branches of the radial dispatch get hit.
    _r = 10.0 ** rng.uniform(-2.0, 2.6, size=N)
    _u = rng.normal(size=(N, 3))
    xyz_mp = _r[:, None] * _u / np.linalg.norm(_u, axis=1)[:, None]

    pots = [
        ("Plummer",       agama.Potential(type='Plummer',       mass=1.0, scaleRadius=1.0)),
        ("Isochrone",     agama.Potential(type='Isochrone',     mass=1.0, scaleRadius=1.0)),
        ("NFW",           agama.Potential(type='NFW',           mass=1.0, scaleRadius=1.0)),
        ("MiyamotoNagai", agama.Potential(type='MiyamotoNagai', mass=1.0, scaleRadius=1.0,
                                          scaleHeight=0.3)),
        ("Logarithmic",   agama.Potential(type='Logarithmic',   v0=1.0, scaleRadius=0.1,
                                          axisRatioY=0.9, axisRatioZ=0.7)),
        ("Harmonic",      agama.Potential(type='Harmonic',      Omega=1.0,
                                          axisRatioY=0.8, axisRatioZ=0.5)),
    ]

    all_ok = True

    # -- Parity across 6 potentials x 4 devices x 2 precisions = 48 cases --
    print("== Parity: 6 potentials x {cpu, openmp, serial, cuda} x {float64, float32} ==")
    for name, pot in pots:
        ref = pot.potential(xyz)
        for device in ("cpu", "openmp", "serial", "cuda"):
            for dtype in (np.float64, np.float32):
                try:
                    if not _check(name, pot, xyz, device, dtype, ref):
                        all_ok = False
                except Exception as e:
                    print(f"  FAIL {name:14s} {device:6s} {dtype.__name__:8s} : "
                          f"{type(e).__name__}: {e}")
                    all_ok = False

    # -- serial must match cpu/openmp EXACTLY (same math, order-independent) --
    print("\n== Exact (bit-for-bit) serial == cpu == openmp ==")
    for name, pot in pots:
        for dtype in (np.float64, np.float32):
            phi_cpu    = pot.potential(xyz, device="cpu",    dtype=dtype)
            phi_openmp = pot.potential(xyz, device="openmp", dtype=dtype)
            phi_serial = pot.potential(xyz, device="serial", dtype=dtype)
            ok = np.array_equal(phi_serial, phi_cpu) and np.array_equal(phi_serial, phi_openmp)
            print(f"  {'OK  ' if ok else 'FAIL'} {name:14s} {dtype.__name__:8s} "
                  f": serial==cpu {np.array_equal(phi_serial, phi_cpu)}, "
                  f"serial==openmp {np.array_equal(phi_serial, phi_openmp)}")
            if not ok:
                all_ok = False

    # -- Input-type flexibility: list-of-tuples, list-of-lists, raw sequences --
    print("\n== Input-type flexibility (matches AGAMA's default input handling) ==")
    nfw = agama.Potential(type='NFW', mass=1.0, scaleRadius=1.0)
    xyz_small = np.array([[1.0, 0.5, 0.3], [0.2, -0.7, 1.1], [2.5, 1.0, -0.5]])
    ref_small = nfw.potential(xyz_small)
    inputs = [
        ("numpy (3,3)",      xyz_small),
        ("list of tuples",   [(1.0, 0.5, 0.3), (0.2, -0.7, 1.1), (2.5, 1.0, -0.5)]),
        ("list of lists",    [[1.0, 0.5, 0.3], [0.2, -0.7, 1.1], [2.5, 1.0, -0.5]]),
        ("nested tuples",    ((1.0, 0.5, 0.3), (0.2, -0.7, 1.1), (2.5, 1.0, -0.5))),
    ]
    for desc, xyz_in in inputs:
        try:
            phi = nfw.potential(xyz_in, device='cuda', dtype=np.float64)
            err = float(np.max(np.abs(phi - ref_small)))
            ok = err <= 1e-12 * float(np.max(np.abs(ref_small)))
            print(f"  {'OK  ' if ok else 'FAIL'} NFW cuda fp64 input={desc:20s} : |err|={err:.3e}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  FAIL NFW cuda fp64 input={desc:20s} : {type(e).__name__}: {e}")
            all_ok = False

    # -- Single-point input (length-3 ndarray, 3-tuple, 3-list) --
    print("\n== Single-point input -> 0-D scalar output ==")
    single_inputs = [
        ("ndarray (3,)",  np.array([1.0, 0.5, 0.3])),
        ("3-tuple",       (1.0, 0.5, 0.3)),
        ("3-list",        [1.0, 0.5, 0.3]),
    ]
    ref_single = nfw.potential((1.0, 0.5, 0.3))
    for desc, inp in single_inputs:
        for dtype in (np.float64, np.float32):
            try:
                phi = nfw.potential(inp, device='cuda', dtype=dtype)
                if phi.shape != ():
                    print(f"  FAIL NFW cuda {dtype.__name__:8s} input={desc:14s} : "
                          f"got shape {phi.shape}, want ()")
                    all_ok = False
                    continue
                tol = 5e-6 if dtype == np.float32 else 1e-12
                err = float(abs(float(phi) - ref_single))
                ok = err <= tol * max(abs(ref_single), 1.0)
                print(f"  {'OK  ' if ok else 'FAIL'} NFW cuda {dtype.__name__:8s} "
                      f"input={desc:14s} : phi={float(phi):.6f}  ref={ref_single:.6f}  |err|={err:.3e}")
                if not ok:
                    all_ok = False
            except Exception as e:
                print(f"  FAIL NFW cuda {dtype.__name__:8s} input={desc:14s} : "
                      f"{type(e).__name__}: {e}")
                all_ok = False

    # -- Composite of GPU-capable members: on-device accumulation parity --
    print("\n== Composite (Plummer+NFW+MiyamotoNagai): device x dtype parity ==")
    composite = agama.Potential(
        dict(type='Plummer',       mass=1.0, scaleRadius=1.0),
        dict(type='NFW',           mass=1.0, scaleRadius=2.0),
        dict(type='MiyamotoNagai', mass=0.5, scaleRadius=3.0, scaleHeight=0.3),
    )
    ref_comp = composite.potential(xyz)          # legacy CPU path
    for device in ("serial", "cpu", "openmp", "cuda"):
        for dtype in (np.float64, np.float32):
            try:
                if not _check("Composite3", composite, xyz, device, dtype, ref_comp):
                    all_ok = False
            except Exception as e:
                print(f"  FAIL Composite3     {device:6s} {dtype.__name__:8s} : "
                      f"{type(e).__name__}: {e}")
                all_ok = False

    # -- Composite: CuPy in -> CuPy out parity vs the NumPy cuda path --
    print("\n== Composite: CuPy device-resident parity ==")
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        cp = None
        print(f"  SKIP composite CuPy test: cupy unavailable ({type(e).__name__}: {e})")
    if cp is not None:
        ref_np = composite.potential(xyz, device='cuda', dtype=np.float64)
        try:
            phi_d = composite.potential(cp.asarray(xyz), device='cuda')
            is_cp = isinstance(phi_d, cp.ndarray)
            val_ok = np.allclose(cp.asnumpy(phi_d), ref_np, rtol=1e-12, atol=0.0)
            ok = is_cp and val_ok
            print(f"  {'OK  ' if ok else 'FAIL'} Composite3 cupy fp64 in -> cupy out : "
                  f"isinstance={is_cp} allclose(1e-12)={val_ok}")
        except Exception as e:
            print(f"  FAIL Composite3 cupy : {type(e).__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False

    # -- Nested composite --
    # The Python factory FLATTENS composite arguments (Potential_initFromTuple
    # in interface_python.cpp unpacks any Composite into its components), so a
    # truly nested Composite is not constructible from Python; the C++
    # can_dispatch/try_dispatch recursion exists for completeness. We verify
    # the flattened result still dispatches correctly on-device.
    print("\n== Nested composite (factory flattens; verify flattened dispatch) ==")
    nested = agama.Potential(
        composite,   # composite of 3 -> unpacked into its members
        agama.Potential(type='Isochrone', mass=0.7, scaleRadius=1.5),
    )
    print(f"  NOTE nested construction produced: {nested}")
    ref_nested = nested.potential(xyz)
    try:
        phi = nested.potential(xyz, device='cuda', dtype=np.float64)
        err = float(np.max(np.abs(phi - ref_nested)))
        ok = err <= 1e-12 * float(np.max(np.abs(ref_nested)))
        print(f"  {'OK  ' if ok else 'FAIL'} nested(=flattened) composite cuda fp64 : |err|={err:.3e}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"  FAIL nested composite cuda : {type(e).__name__}: {e}")
        all_ok = False

    # -- NotImplementedError on a composite with an unsupported member --
    # Non-spherical Dehnen (axisRatioZ != 1) is the genuinely-unsupported member:
    # spherical Dehnen became GPU-capable in Tier 1, but the dispatch gates Dehnen
    # on isSpherical(symmetry()) across all ops (potential/force/density), so a
    # flattened Dehnen still raises NotImplementedError naming 'Dehnen'.
    print("\n== NotImplementedError on composite containing unsupported member ==")
    composite_bad = agama.Potential(
        dict(type='Plummer', mass=1.0, scaleRadius=1.0),
        dict(type='Dehnen',  mass=1.0, scaleRadius=1.0, axisRatioZ=0.7),
    )
    try:
        composite_bad.potential(xyz, device='cuda', dtype=np.float64)
        print("  FAIL Composite+Dehnen cuda : unexpectedly succeeded")
        all_ok = False
    except NotImplementedError as e:
        ok = 'Dehnen' in str(e)
        print(f"  {'OK  ' if ok else 'FAIL'} Composite+Dehnen cuda raised NotImplementedError "
              f"(names unsupported member: {ok}):")
        print(f"         {e}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"  FAIL Composite+Dehnen cuda raised wrong type {type(e).__name__}: {e}")
        all_ok = False

    # -- DiskAnsatz (Tier 1) is GPU-capable but not reachable bare from Python: the
    # only way to build one is agama.Potential(type='Disk', ...), and
    # potential_factory.cpp's GalPot scheme ALWAYS pairs it with a Multipole holding
    # the residual density. Until Tier 2 that Composite{DiskAnsatz, Multipole} could
    # not dispatch at all (can_dispatch requires EVERY member), and this block used to
    # assert that the resulting NotImplementedError named 'Multipole' rather than
    # 'DiskAnsatz'. Tier 2 commit 5b changed the answer: the Multipole now dispatches
    # too, so the whole GalPot-style composite runs on the GPU and this is a
    # positive-parity test -- and incidentally the first end-to-end Python check that
    # DiskAnsatz's device path produces right numbers, not just that it is recognized.
    print("\n== type='Disk' composite (DiskAnsatz + Multipole residual): full GPU dispatch ==")
    disk_composite = agama.Potential(type='Disk', surfaceDensity=1.0, scaleRadius=2.0,
                                     scaleHeight=0.2)
    try:
        ref_dc = disk_composite.potential(xyz)
        for dtype, tol in ((np.float64, 1e-11), (np.float32, 1e-5)):
            arg = xyz if dtype == np.float64 else xyz.astype(np.float32)
            got = disk_composite.potential(arg, device='cuda', dtype=dtype)
            err = _pointwise_rel(got, ref_dc if dtype == np.float64 else
                                 disk_composite.potential(arg, device='cpu', dtype=dtype))
            ok = err <= tol
            print(f"  {'OK  ' if ok else 'FAIL'} Disk composite cuda {dtype.__name__:8s}: "
                  f"max pointwise rel = {err:.3e}  tol = {tol:.1e}")
            if not ok:
                all_ok = False
    except NotImplementedError as e:
        print(f"  FAIL Disk composite cuda still refused: NotImplementedError: {e}")
        all_ok = False
    except Exception as e:
        print(f"  FAIL Disk composite cuda raised {type(e).__name__}: {e}")
        all_ok = False

    # -- Force + density parity: 6 potentials + Composite3, device x dtype --
    # References are the LEGACY CPU paths (pot.force(xyz) -> (N,3),
    # pot.density(xyz) -> (N,)), not the device='cpu' path, so this closes the
    # loop legacy-vs-batch for the new fused Phi+acc and density kernels.
    print("\n== Force & density parity: {6 pots + Composite3} x device x dtype ==")
    fd_targets = pots + [("Composite3", composite)]
    for name, pot in fd_targets:
        for op in ("force", "density"):
            ref = getattr(pot, op)(xyz)          # legacy CPU path
            for device in ("serial", "cpu", "openmp", "cuda"):
                for dtype in (np.float64, np.float32):
                    try:
                        if not _check_fd(name, pot, xyz, device, dtype, ref, op):
                            all_ok = False
                    except Exception as e:
                        print(f"  FAIL {name:14s} {op:8s} {device:6s} "
                              f"{dtype.__name__:8s} : {type(e).__name__}: {e}")
                        all_ok = False

    # -- Force & density single-point shapes: (3,) for force, 0-d for density --
    print("\n== Force & density single-point input shapes ==")
    pt = (1.0, 0.5, 0.3)
    ref_f = np.asarray(nfw.force(pt), dtype=np.float64)   # legacy: (3,)
    ref_d = float(nfw.density(pt))                        # legacy: scalar
    for dtype in (np.float64, np.float32):
        tol = 2e-6 if dtype == np.float32 else 1e-12
        try:
            f = nfw.force(np.array(pt), device='cuda', dtype=dtype)
            shape_ok = f.shape == (3,)
            err = float(np.max(np.abs(f.astype(np.float64) - ref_f)))
            ok = shape_ok and err <= tol * float(np.max(np.abs(ref_f)))
            print(f"  {'OK  ' if ok else 'FAIL'} NFW force   cuda {dtype.__name__:8s} "
                  f"single point : shape={f.shape} (want (3,))  |err|={err:.3e}")
        except Exception as e:
            print(f"  FAIL NFW force   cuda {dtype.__name__:8s} single point : "
                  f"{type(e).__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False
        try:
            d = nfw.density(pt, device='cuda', dtype=dtype)
            shape_ok = d.shape == ()
            err = abs(float(d) - ref_d)
            ok = shape_ok and err <= tol * max(abs(ref_d), 1.0)
            print(f"  {'OK  ' if ok else 'FAIL'} NFW density cuda {dtype.__name__:8s} "
                  f"single point : shape={d.shape} (want ())    |err|={err:.3e}")
        except Exception as e:
            print(f"  FAIL NFW density cuda {dtype.__name__:8s} single point : "
                  f"{type(e).__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False

    # -- NotImplementedError for force/density on unsupported targets --
    print("\n== Force & density NotImplementedError on unsupported targets ==")
    for op in ("force", "density"):
        try:
            getattr(composite_bad, op)(xyz, device='cuda', dtype=np.float64)
            print(f"  FAIL Composite+Dehnen {op} cuda : unexpectedly succeeded")
            all_ok = False
        except NotImplementedError as e:
            ok = 'Dehnen' in str(e)
            print(f"  {'OK  ' if ok else 'FAIL'} Composite+Dehnen {op:8s} cuda raised "
                  f"NotImplementedError (names Dehnen: {ok})")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  FAIL Composite+Dehnen {op} cuda raised wrong type "
                  f"{type(e).__name__}: {e}")
            all_ok = False
    # a Density NOT backed by a Potential object must reject the device kwarg
    # (agama.Density(type='Plummer') is backed by the Plummer POTENTIAL class,
    # so it legitimately dispatches -- verified below; Spheroid is a pure
    # density with no BasePotential base, so it must raise).
    dens_only = agama.Density(type='Spheroid', densityNorm=1.0, scaleRadius=1.0,
                              gamma=1, beta=4)
    try:
        dens_only.density(xyz, device='cuda', dtype=np.float64)
        print("  FAIL Spheroid (pure Density) density cuda : unexpectedly succeeded")
        all_ok = False
    except NotImplementedError as e:
        print(f"  OK   Spheroid (pure Density, non-Potential) density cuda raised "
              f"NotImplementedError: {e}")
    except Exception as e:
        print(f"  FAIL Spheroid density cuda raised wrong type {type(e).__name__}: {e}")
        all_ok = False
    # ...whereas a Density backed by a GPU-migrated Potential dispatches fine
    dens_plummer = agama.Density(type='Plummer', mass=1.0, scaleRadius=1.0)
    try:
        out = dens_plummer.density(xyz, device='cuda', dtype=np.float64)
        ref = dens_plummer.density(xyz)
        err = float(np.max(np.abs(out - ref)))
        ok = err <= 1e-12 * float(np.max(np.abs(ref)))
        print(f"  {'OK  ' if ok else 'FAIL'} Density(type='Plummer') "
              f"(backed by Potential) density cuda : |err|={err:.3e}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"  FAIL Density(type='Plummer') density cuda : {type(e).__name__}: {e}")
        all_ok = False

    # -- Malformed CAI 'stream' raises RuntimeError, does NOT abort --
    # Run in a subprocess: before the exception-safety fix this aborted the
    # whole interpreter (throw escaping Py_BEGIN_ALLOW_THREADS), so a
    # regression here must not be able to kill this test runner.
    print("\n== Malformed CAI 'stream' -> RuntimeError (subprocess) ==")
    snippet = (
        "import numpy as np, agama_migrate as agama\n"
        "import cupy as cp\n"
        "cp.cuda.runtime.getDeviceCount()\n"
        "pot = agama.Potential(type='Plummer', mass=1.0, scaleRadius=1.0)\n"
        "d = cp.asarray(np.random.default_rng(0).uniform(-5, 5, (100, 3)))\n"
        "cai = dict(d.__cuda_array_interface__)\n"
        "cai['stream'] = 0xdeadbeef\n"
        "class Fake:\n"
        "    __cuda_array_interface__ = cai\n"
        "try:\n"
        "    pot.potential(Fake(), device='cuda')\n"
        "    print('NO-EXCEPTION')\n"
        "except RuntimeError as e:\n"
        "    print('RuntimeError:', e)\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", snippet],
                              capture_output=True, text=True, timeout=120)
        out = proc.stdout + proc.stderr
        if "cupy" in out and ("ImportError" in out or "CUDARuntimeError" in out) \
                and proc.returncode != 0:
            print(f"  SKIP garbage-stream subprocess test: cupy unavailable in subprocess")
        else:
            ok = proc.returncode == 0 and "RuntimeError" in proc.stdout
            print(f"  {'OK  ' if ok else 'FAIL'} subprocess returncode={proc.returncode} "
                  f"(0 = no abort), RuntimeError in output: {'RuntimeError' in proc.stdout}")
            if ok:
                print(f"         {proc.stdout.strip().splitlines()[-1]}")
            else:
                print(f"         stdout: {proc.stdout.strip()[:300]}")
                print(f"         stderr: {proc.stderr.strip()[:300]}")
                all_ok = False
    except Exception as e:
        print(f"  FAIL garbage-stream subprocess test: {type(e).__name__}: {e}")
        all_ok = False

    # -- Concurrent Python threads on the shared GPU scratch buffers --
    print("\n== Threading smoke test: concurrent device='cuda' calls ==")
    if not threading_smoke_test(nfw):
        all_ok = False

    # -- Phase B: __cuda_array_interface__ passthrough (CuPy in -> CuPy out) --
    print("\n== CuPy device-resident passthrough (__cuda_array_interface__) ==")
    if not cupy_passthrough_tests(pots, xyz):
        all_ok = False

    # -- Force & density through the CuPy passthrough (incl. (N,3) force shape) --
    print("\n== Force & density: CuPy in -> CuPy out parity ==")
    if not force_density_cupy_tests(pots + [("Composite3", composite)], xyz):
        all_ok = False

    # -- fp32 accuracy where the parity sweep above is structurally blind --
    if not nfw_fp32_crossover_test():
        all_ok = False

    # ---- Tier 2 commit 5b: Multipole through the descriptor path ----
    mp_targets = multipole_targets()
    print("\n== Multipole parity: 6 models x {potential,force,density} x 4 devices "
          "x 2 precisions ==")
    if not multipole_parity_tests(mp_targets, xyz_mp):
        all_ok = False
    print("\n== Multipole: CuPy in -> CuPy out (force & density) ==")
    if not force_density_cupy_tests(mp_targets, xyz_mp):
        all_ok = False
    print()
    if not multipole_fp32_budget():
        all_ok = False
    print()
    if not multipole_near_axis_test():
        all_ok = False
    print()
    if not multipole_fail_closed_test():
        all_ok = False
    print()
    if not multipole_orbit_test():
        all_ok = False

    print("\n" + ("PASS" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
