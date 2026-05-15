"""Tier 1 Python smoke test: pot.potential(xyz, device=..., dtype=...) parity.

Exercises the new device + dtype kwargs on Potential.potential() across all
three execution policies (cpu / openmp / cuda) and both supported precisions
(float32 / float64), for the six analytic potentials that have been migrated
(Plummer, Isochrone, NFW, MiyamotoNagai, Logarithmic, Harmonic).

The reference is the legacy `pot.potential(xyz)` call (no device kwarg) which
goes through the existing OpenMP-parallelized CPU path. Each device/dtype
combination is compared to the legacy reference; relative tolerance vs max|phi|
because some potentials (Harmonic) produce O(1000) values.

Also tests the NotImplementedError path for a non-GPU-capable potential
(Composite is currently not in the dispatch list).
"""
import sys
import numpy as np
import agama_migrate as agama


def _check(pot_name, pot, xyz, device, dtype, ref):
    """Run one device/dtype combo and compare against ref."""
    phi = pot.potential(xyz, device=device, dtype=dtype)
    assert phi.dtype == dtype, f"{pot_name} {device}/{dtype}: got dtype {phi.dtype}"
    assert phi.shape == ref.shape, f"{pot_name} {device}/{dtype}: shape {phi.shape} vs {ref.shape}"
    max_ref = float(np.max(np.abs(ref))) if ref.size else 1.0
    # fp32 carries ~1e-7 relative; fp64 ~1e-13. Be generous.
    rel_tol = 5e-6 if dtype == np.float32 else 1e-12
    abs_tol = rel_tol * max_ref
    err = float(np.max(np.abs(phi.astype(np.float64) - ref)))
    ok = err <= abs_tol
    status = "OK" if ok else "FAIL"
    print(f"  {pot_name:14s}  device={device:6s}  dtype={str(dtype.__name__):8s}  "
          f"max|phi|={max_ref:.3e}  |err|={err:.3e}  tol={abs_tol:.3e}  -> {status}")
    return ok


def main():
    rng = np.random.default_rng(42)
    N = 1024
    xyz = rng.uniform(-5.0, 5.0, size=(N, 3))

    # All six GPU-capable potentials with the same params used in the C++ test.
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
    for pot_name, pot in pots:
        ref = pot.potential(xyz)  # legacy path; reference value
        for device in ("cpu", "openmp", "cuda"):
            for dtype in (np.float64, np.float32):
                try:
                    ok = _check(pot_name, pot, xyz, device, dtype, ref)
                    all_ok = all_ok and ok
                except Exception as e:
                    print(f"  {pot_name:14s}  device={device:6s}  dtype={dtype.__name__:8s}  "
                          f"-> EXCEPTION: {type(e).__name__}: {e}")
                    all_ok = False

    # NotImplementedError path: build a Composite (currently not GPU-capable).
    print("\n-- NotImplementedError path --")
    try:
        composite = agama.Potential(
            agama.Potential(type='Plummer', mass=1.0, scaleRadius=1.0),
            agama.Potential(type='NFW',     mass=1.0, scaleRadius=2.0),
        )
        composite.potential(xyz, device='cuda', dtype=np.float64)
        print("  Composite + device='cuda': unexpectedly succeeded -- FAIL")
        all_ok = False
    except NotImplementedError as e:
        print(f"  Composite + device='cuda' raised NotImplementedError as expected:")
        print(f"    {e}")
    except Exception as e:
        print(f"  Composite + device='cuda' raised wrong exception type {type(e).__name__}: {e}")
        all_ok = False

    # Single-point input (Nx3 -> shape (3,))
    print("\n-- Single-point input --")
    single = np.array([1.0, 0.5, 0.3])
    for pot_name, pot in pots[:2]:
        for device in ("cpu", "cuda"):
            for dtype in (np.float64, np.float32):
                try:
                    phi = pot.potential(single, device=device, dtype=dtype)
                    print(f"  {pot_name:14s}  device={device:6s}  dtype={dtype.__name__:8s}"
                          f"  -> phi = {float(phi):.6f}  (shape {phi.shape}, dtype {phi.dtype})")
                except Exception as e:
                    print(f"  {pot_name:14s}  device={device:6s}  dtype={dtype.__name__:8s}"
                          f"  -> EXCEPTION: {type(e).__name__}: {e}")
                    all_ok = False

    print("\n" + ("PASS" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
