"""Device-resident trajectory output: agama_migrate.orbit(device='cuda', deviceOutput=True).

Tests the new deviceOutput=True kwarg on agama_migrate.orbit(), which keeps the
trajectory buffer on the GPU (a cupy.ndarray, filled by orbit::integrateOrbitsGPUDevice
and, when the unit system is non-trivial, orbit::scaleTrajectoryGPUDevice -- see
src/orbit_gpu.h) instead of copying it back to a NumPy array. Mirrors the
__cuda_array_interface__ passthrough already covered for pot.potential/force/density
in test_potential_gpu.py, adapted to orbit()'s shape: only the trajectory is
device-resident (`ic`/`time`/`timestart` stay host, since they are O(numOrbits) while
the trajectory is O(numOrbits*trajsize*6) -- two orders of magnitude larger), and the
only supported output shape is separateTime=True with a dtype that exactly matches the
integration precision (fp64 dtype + fp64 integration, or fp32 dtype + fp32 integration).

Tests covered:
  1. Rejections (pure argument validation -- do not require CUDA hardware or CuPy):
     separateTime=False, dtype/precision mismatch (default dtype AND explicit
     complex64), der=True, lyapunov=True, deviceOutput=True without device='cuda'
     (device omitted, device='cpu').
  2. Device-resident result == host-resident result, BITWISE, for fp64 and fp32.
  3. Units: under a non-trivial agama_migrate.setUnits(), device-resident ==
     host-resident bitwise, AND the values actually differ from the unitless case
     (guards against a silently-no-op setUnits, which this fork shipped once --
     see agama_migrate_setunits_crossinstall in the project's memory).
  4. Shape/dtype of the returned array: cupy.ndarray, __cuda_array_interface__
     present, expected shape for both the batch and single-orbit (ic.shape==(6,)) cases.
  5. SKIP cleanly (not FAIL) when CuPy is absent, no GPU is present, or this build
     lacks CUDA support -- checked same way as test_orbit_gpu.py/test_potential_gpu.py.
"""
import sys
import numpy as np
import agama_migrate as agama


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bitwise_equal(a, b):
    """True iff a and b have the same shape/dtype and identical bit patterns.

    Deliberately NOT numpy.allclose or a plain numpy.array_equal on the raw float
    values: array_equal on floats returns False for a NaN slot even when both
    sides carry the identical NaN bit pattern (NaN != NaN), which would falsely
    flag the samples-not-reached / NAN-fill convention documented in orbit_gpu.h
    as a mismatch. Viewing as raw bytes compares bit patterns directly and is
    correct for NaNs too.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return np.array_equal(a.view(np.uint8), b.view(np.uint8))


def _make_ics(pot, n=48, seed=7):
    """Near-circular ICs at r=0.5..3, times=15..35 (short, for a fast test)."""
    rng = np.random.default_rng(seed)
    r   = rng.uniform(0.5, 3.0, n)
    phi = rng.uniform(0, 2*np.pi, n)
    z   = rng.uniform(-0.2, 0.2, n)
    pos = np.column_stack([r*np.cos(phi), r*np.sin(phi), z])
    vcirc = np.sqrt(np.maximum(-np.sum(pot.force(pos)*pos, axis=1), 0.0))
    vel = np.column_stack([-vcirc*np.sin(phi), vcirc*np.cos(phi),
                           rng.normal(0, 0.03, n)])
    vel += rng.normal(0, 0.02, (n, 3)) * vcirc[:, None]
    ic = np.column_stack([pos, vel])
    T = rng.uniform(15.0, 35.0, n)
    return ic, T


# ---------------------------------------------------------------------------
# 1. Rejections -- pure argument validation, no CUDA hardware/CuPy required
# ---------------------------------------------------------------------------

def test_rejections(pot, ic, T, trajsize, all_ok):
    print("\n== Rejections (deviceOutput=True); no CUDA hardware required ==")

    def _expect(exc_type, tag, fn):
        nonlocal all_ok
        try:
            fn()
            print(f"  FAIL {tag} : no exception raised")
            all_ok = False
        except exc_type as e:
            print(f"  OK   {tag} : raised {exc_type.__name__}: {e}")
        except Exception as e:
            print(f"  FAIL {tag} : raised {type(e).__name__} instead of "
                  f"{exc_type.__name__}: {e}")
            all_ok = False

    # deviceOutput=True requires device='cuda' -- device omitted / device='cpu'.
    _expect(ValueError, "deviceOutput=True, device omitted",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                dtype=np.float64, separateTime=True,
                                deviceOutput=True, verbose=False))
    _expect(ValueError, "deviceOutput=True, device='cpu'",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cpu', dtype=np.float64, separateTime=True,
                                deviceOutput=True, verbose=False))
    _expect(ValueError, "deviceOutput=True, device='openmp'",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='openmp', dtype=np.float64, separateTime=True,
                                deviceOutput=True, verbose=False))

    # separateTime=False: the deprecated per-orbit object-array format has no
    # device-resident shape.
    _expect(NotImplementedError, "deviceOutput=True + separateTime=False",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', dtype=np.float64, separateTime=False,
                                deviceOutput=True, verbose=False))
    # separateTime omitted defaults to False -- same rejection.
    _expect(NotImplementedError, "deviceOutput=True + separateTime omitted (defaults False)",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', dtype=np.float64,
                                deviceOutput=True, verbose=False))

    # dtype omitted entirely: storage defaults to float32 while integration
    # defaults to fp64 -- the default call is ITSELF a precision mismatch.
    _expect(TypeError, "deviceOutput=True + dtype omitted (float32 storage, fp64 integration)",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', separateTime=True,
                                deviceOutput=True, verbose=False))

    # explicit complex64: not a supported device-resident storage format at all.
    _expect(TypeError, "deviceOutput=True + dtype=complex64",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', dtype=np.complex64, separateTime=True,
                                deviceOutput=True, verbose=False))

    # explicit complex128: same story.
    _expect(TypeError, "deviceOutput=True + dtype=complex128",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', dtype=np.complex128, separateTime=True,
                                deviceOutput=True, verbose=False))

    # der=True / lyapunov=True: rejected generically for any device=... call,
    # deviceOutput or not.
    _expect(NotImplementedError, "deviceOutput=True + der=True",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', dtype=np.float64, separateTime=True,
                                deviceOutput=True, der=True, verbose=False))
    _expect(NotImplementedError, "deviceOutput=True + lyapunov=True",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', dtype=np.float64, separateTime=True,
                                deviceOutput=True, lyapunov=True, verbose=False))

    # deviceOutput must itself be a proper boolean/int 0/1.
    _expect(TypeError, "deviceOutput='banana' (not a bool)",
            lambda: agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                device='cuda', dtype=np.float64, separateTime=True,
                                deviceOutput='banana', verbose=False))

    return all_ok


# ---------------------------------------------------------------------------
# 2. Device-resident == host-resident, bitwise
# ---------------------------------------------------------------------------

def test_device_equals_host(pot, ic, T, trajsize, cp, all_ok):
    print("\n== Device-resident output == host-resident output (bitwise) ==")

    for dtype, label in [(np.float64, "fp64"), (np.float32, "fp32")]:
        ok = True
        try:
            t_host, traj_host = agama.orbit(
                potential=pot, ic=ic, time=T, trajsize=trajsize,
                device='cuda', dtype=dtype, separateTime=True, verbose=False)
            t_dev, traj_dev_cp = agama.orbit(
                potential=pot, ic=ic, time=T, trajsize=trajsize,
                device='cuda', dtype=dtype, separateTime=True,
                deviceOutput=True, verbose=False)
            traj_dev = cp.asnumpy(traj_dev_cp)
            times_ok = _bitwise_equal(t_dev, t_host)
            traj_ok = _bitwise_equal(traj_dev, traj_host)
            ok = times_ok and traj_ok
            print(f"  {'OK  ' if ok else 'FAIL'} {label} : "
                  f"times_bitwise={times_ok} traj_bitwise={traj_ok}")
        except Exception as e:
            print(f"  FAIL {label} device==host : {type(e).__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 3. Units
# ---------------------------------------------------------------------------

def test_units(pot, ic, T, trajsize, cp, all_ok):
    print("\n== Units: device-resident == host-resident under a non-trivial unit system ==")

    # Unitless baseline, so we can later confirm setUnits actually changed something
    # (rather than the units test passing vacuously because setUnits was a no-op --
    # this fork shipped exactly that bug once, see agama_migrate_setunits_crossinstall).
    t_unitless, traj_unitless = agama.orbit(
        potential=pot, ic=ic, time=T, trajsize=trajsize,
        device='cuda', dtype=np.float64, separateTime=True, verbose=False)

    ok = True
    agama.setUnits(length=2.5, velocity=3.0, mass=1e6)
    try:
        t_host, traj_host = agama.orbit(
            potential=pot, ic=ic, time=T, trajsize=trajsize,
            device='cuda', dtype=np.float64, separateTime=True, verbose=False)
        t_dev, traj_dev_cp = agama.orbit(
            potential=pot, ic=ic, time=T, trajsize=trajsize,
            device='cuda', dtype=np.float64, separateTime=True,
            deviceOutput=True, verbose=False)
        traj_dev = cp.asnumpy(traj_dev_cp)

        times_ok = _bitwise_equal(t_dev, t_host)
        traj_ok = _bitwise_equal(traj_dev, traj_host)
        ok = times_ok and traj_ok
        print(f"  {'OK  ' if ok else 'FAIL'} device==host under nontrivial units : "
              f"times_bitwise={times_ok} traj_bitwise={traj_ok}")

        changed = not _bitwise_equal(traj_host, traj_unitless)
        print(f"  {'OK  ' if changed else 'FAIL'} setUnits actually changed the stored "
              f"values relative to the unitless case (non-vacuous units test)")
        ok = ok and changed
    except Exception as e:
        print(f"  FAIL units test : {type(e).__name__}: {e}")
        ok = False
    finally:
        # reset to the default (unitless, G=1) system for any test that runs after this
        agama.setUnits(mass=1, length=1, velocity=1)

    if not ok:
        all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# 4. Shape / dtype of the returned array
# ---------------------------------------------------------------------------

def test_shape_and_dtype(pot, ic, T, trajsize, cp, all_ok):
    print("\n== Shape/dtype of the device-resident result ==")

    norb = ic.shape[0]
    for dtype, cp_dtype in [(np.float64, cp.float64), (np.float32, cp.float32)]:
        ok = True
        try:
            t_dev, traj_dev = agama.orbit(
                potential=pot, ic=ic, time=T, trajsize=trajsize,
                device='cuda', dtype=dtype, separateTime=True,
                deviceOutput=True, verbose=False)
            is_cupy = isinstance(traj_dev, cp.ndarray)
            has_cai = hasattr(traj_dev, '__cuda_array_interface__')
            shape_ok = traj_dev.shape == (norb, trajsize, 6)
            dtype_ok = traj_dev.dtype == cp_dtype
            times_ok = isinstance(t_dev, np.ndarray) and t_dev.shape == (norb, trajsize)
            ok = is_cupy and has_cai and shape_ok and dtype_ok and times_ok
            print(f"  {'OK  ' if ok else 'FAIL'} dtype={dtype.__name__} : "
                  f"isinstance(cupy.ndarray)={is_cupy} has_cai={has_cai} "
                  f"traj.shape={traj_dev.shape} traj.dtype={traj_dev.dtype} "
                  f"times.shape={t_dev.shape}")
        except Exception as e:
            print(f"  FAIL dtype={dtype.__name__} shape/dtype : {type(e).__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False

    # Single-orbit ic (shape (6,)): the device array drops the leading numOrbits
    # dimension, same convention as the host path (PyArray_Return squeeze).
    ok = True
    try:
        ic_single = ic[0]
        T_single = float(T[0]) if hasattr(T, '__len__') else T
        t_dev, traj_dev = agama.orbit(
            potential=pot, ic=ic_single, time=T_single, trajsize=trajsize,
            device='cuda', dtype=np.float64, separateTime=True,
            deviceOutput=True, verbose=False)
        is_cupy = isinstance(traj_dev, cp.ndarray)
        shape_ok = traj_dev.shape == (trajsize, 6)
        times_ok = t_dev.shape == (trajsize,)
        ok = is_cupy and shape_ok and times_ok
        print(f"  {'OK  ' if ok else 'FAIL'} single-orbit ic shape (6,) : "
              f"isinstance(cupy.ndarray)={is_cupy} traj.shape={traj_dev.shape} "
              f"times.shape={t_dev.shape}")
    except Exception as e:
        print(f"  FAIL single-orbit ic shape : {type(e).__name__}: {e}")
        ok = False
    if not ok:
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pot = agama.Potential(
        dict(type='Plummer',       mass=0.1, scaleRadius=0.3),
        dict(type='MiyamotoNagai', mass=0.5, scaleRadius=1.0, scaleHeight=0.3),
        dict(type='NFW',           mass=10.0, scaleRadius=5.0),
    )
    NORB, TRAJSIZE = 48, 16
    ic, T = _make_ics(pot, n=NORB, seed=7)

    all_ok = True

    # Pure argument-validation rejections: independent of CUDA hardware / CuPy,
    # run unconditionally (they must pass even on a CPU-only box).
    all_ok = test_rejections(pot, ic, T, TRAJSIZE, all_ok)

    # Everything below needs an actual device array, so it needs CuPy + a usable
    # CUDA device, AND a HAVE_CUDA=1 build. Skip cleanly (not FAIL) otherwise.
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()   # raises if no usable CUDA device
    except Exception as e:
        print(f"\nSKIP device-resident-output tests: CuPy/GPU unavailable "
              f"({type(e).__name__}: {e})")
        print()
        print("PASS" if all_ok else "FAIL")
        sys.exit(0 if all_ok else 1)

    cuda_build_ok = True
    try:
        agama.orbit(potential=pot, ic=ic[:1], time=5.0, trajsize=4,
                    device='cuda', dtype=np.float64, separateTime=True,
                    deviceOutput=True, verbose=False)
    except RuntimeError as e:
        if 'built without CUDA support' in str(e):
            cuda_build_ok = False
            print("\nSKIP device-resident-output tests: this build lacks CUDA "
                  "support (HAVE_CUDA=0)")
        else:
            raise  # re-raise unexpected RuntimeError

    if cuda_build_ok:
        all_ok = test_device_equals_host(pot, ic, T, TRAJSIZE, cp, all_ok)
        all_ok = test_units(pot, ic, T, TRAJSIZE, cp, all_ok)
        all_ok = test_shape_and_dtype(pot, ic, T, TRAJSIZE, cp, all_ok)

    print()
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
