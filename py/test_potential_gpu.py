"""Tier 1 Python smoke test: pot.potential(xyz, device=..., dtype=...) parity.

For each of the six GPU-migrated analytic potentials (Plummer, Isochrone, NFW,
MiyamotoNagai, Logarithmic, Harmonic), verifies that:

  * The legacy `pot.potential(xyz)` path (no device kwarg) and the new
    `pot.potential(xyz, device=..., dtype=...)` device path produce the same
    Phi values to within ULP-scale relative tolerance.

  * Input flexibility — numpy ndarray, list of tuples, list of lists, and
    raw nested Python sequences — all reach the same code path and produce
    the same result.

  * Single-point input (length-3 array or 3-tuple) returns a 0-D scalar.

  * device='cuda' on a non-GPU-capable potential (Composite) raises
    NotImplementedError naming the concrete type.

Tolerances:
  fp64: relative ~1e-12 vs max|phi| (1 ULP ~ 1e-16, leaves margin for FMA)
  fp32: relative ~5e-6  vs max|phi| (1 ULP ~ 1e-7)
"""
import sys
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

    # -- Parity across 6 potentials x 3 devices x 2 precisions = 36 cases --
    print("== Parity: 6 potentials x {cpu, openmp, cuda} x {float64, float32} ==")
    for name, pot in pots:
        ref = pot.potential(xyz)
        for device in ("cpu", "openmp", "cuda"):
            for dtype in (np.float64, np.float32):
                try:
                    if not _check(name, pot, xyz, device, dtype, ref):
                        all_ok = False
                except Exception as e:
                    print(f"  FAIL {name:14s} {device:6s} {dtype.__name__:8s} : "
                          f"{type(e).__name__}: {e}")
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

    # -- NotImplementedError on Composite --
    print("\n== NotImplementedError on non-GPU-capable potential ==")
    composite = agama.Potential(
        agama.Potential(type='Plummer', mass=1.0, scaleRadius=1.0),
        agama.Potential(type='NFW',     mass=1.0, scaleRadius=2.0),
    )
    try:
        composite.potential(xyz, device='cuda', dtype=np.float64)
        print("  FAIL Composite cuda : unexpectedly succeeded")
        all_ok = False
    except NotImplementedError as e:
        print(f"  OK   Composite cuda raised NotImplementedError:")
        print(f"         {e}")
    except Exception as e:
        print(f"  FAIL Composite cuda raised wrong type {type(e).__name__}: {e}")
        all_ok = False

    print("\n" + ("PASS" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
