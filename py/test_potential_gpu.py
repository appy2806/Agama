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

Tolerances:
  fp64: relative ~1e-12 vs max|phi| (1 ULP ~ 1e-16, leaves margin for FMA)
  fp32: relative ~5e-6  vs max|phi| (1 ULP ~ 1e-7)
"""
import subprocess
import sys
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


def main():
    rng = np.random.default_rng(42)
    N = 1024
    xyz = rng.uniform(-5.0, 5.0, size=(N, 3))

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
    print("\n== NotImplementedError on composite containing unsupported member ==")
    composite_bad = agama.Potential(
        dict(type='Plummer', mass=1.0, scaleRadius=1.0),
        dict(type='Dehnen',  mass=1.0, scaleRadius=1.0),
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

    print("\n" + ("PASS" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
