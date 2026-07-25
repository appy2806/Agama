"""Regression test for the narrow-on-store default-dtype fast path.

Background (see CLAUDE.md / pending_tasks.md "The default-dtype zero-copy
gap"): agama.orbit(device=...) has a zero-copy fast path (`directToDest`)
that lets the batch kernel's D2H copy land straight in the destination NumPy
buffer, but it required the storage dtype width to equal the integration
precision. The plain dtype DEFAULT is float32 storage, while the plain
precision DEFAULT is fp64 integration -- a width mismatch -- so the most
common call (`agama.orbit(..., device=...)`, no `dtype` kwarg) used to take
the slower one-pass `storeTrajectoryGPUImpl` fallback (a Norb*trajsize*6
double intermediate buffer, narrowed to float32 in a second host pass).

The fix templates the trajectory OUTPUT type separately from the
integration type (`integrateOrbitsGPU<T, TOut>`, TOut defaulting to T), so
fp64 integration can narrow each dense-output sample to float32 AT THE
STORE and land the kernel's D2H copy directly in the float32 destination --
`integrateOrbitsGPU<double, float>`. This is enabled in interface_python.cpp
only when the active unit system is trivial (lengthUnit == velocityUnit ==
1): narrowing happens BEFORE any unit conversion, and only under a trivial
unit system is "narrow, then scale-by-1" provably identical to the old
fallback's "scale-by-1, then narrow" (see orbit_gpu.h and the
`narrowStoreDefault` comment in interface_python.cpp for the full argument).
Under a real (non-trivial) unit system, the code must keep taking the
existing fallback -- checked below.

House style: OK/FAIL per check, final PASS/FAIL, import agama_migrate.

Reference construction for check 1 (the key regression check) -- see
test_default_matches_old_fallback_recipe()'s docstring for why comparing
against `dtype=float64` output narrowed in NumPy is a valid, exact stand-in
for what the OLD code (before this change) produced, and for a note on the
actual empirical before/after build diff performed for this change (not
re-run here, since it requires a second built tree; see the commit's report).
"""
import sys
import numpy as np
import agama_migrate as agama


def _make_ics(pot, n=32, seed=42):
    """Near-circular ICs at r=0.5..3, times=20..50 (same recipe as test_orbit_gpu.py)."""
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
    T = rng.uniform(20.0, 50.0, n)
    return ic, T


def _devices(cuda_available):
    return ['cpu', 'openmp', 'serial'] + (['cuda'] if cuda_available else [])


# ---------------------------------------------------------------------------
# Check 1 (required): default call bit-identical to the pre-change output
# ---------------------------------------------------------------------------

def test_default_matches_old_fallback_recipe(pot, ic, T, trajsize, cuda_available, all_ok):
    """Under the default (trivial) unit system, the new narrow-on-store
    default-dtype path must be BITWISE IDENTICAL to what the pre-change code
    produced for the same call.

    Reference construction: dtype=float64 takes the (unaffected)
    directToDest fast path both before and after this change -- storage
    width equals integration precision either way, so its numerical result
    is untouched by this commit. The pre-change default-dtype path
    (storeTrajectoryGPUImpl, before this commit) computed, per component,
    `float(double_value / conv->lengthUnit)`; under the default unit system
    lengthUnit == velocityUnit == 1 exactly, so that reduces to
    `float(double_value)` -- exactly what NumPy's `.astype(np.float32)`
    computes on the same double array (both apply a single IEEE
    round-to-nearest-even narrowing to the identical double value produced
    by the identical integration). So `.astype(np.float32)` of the
    dtype=float64 result reproduces the pre-change default-dtype output
    exactly, without needing to run the old binary here.

    This reasoning was additionally checked empirically for this exact
    change: a tree built from the commit immediately before this one and
    this (post-change) tree were run through the identical default-dtype
    calls below (device in cpu/openmp/serial, both separateTime=True/False)
    and their outputs diffed with np.array_equal -- bit-for-bit identical in
    every case. device='cuda' could not be included in that particular
    build-diff (the pre-change reference build had no CUDA toolchain handy
    at the time), but cuda's default-dtype output is shown INSIDE THIS TEST
    to be bit-for-bit identical to cpu/openmp/serial's (same assertion
    below), and those three were diffed against the real pre-change binary,
    so cuda's result is pinned transitively.

    IMPORTANT: the reference for each device is built with THAT SAME device
    (device=dev, dtype=float64) rather than the no-device legacy CPU
    integrator. The legacy integrator and the Tier-3 batch path are different
    code (see orbit_gpu.cpp's header comment) and are only approximately
    equal in general -- test_orbit_gpu.py's own fp64-parity check uses
    tol_rel=1e-9, not exact equality. Using device=dev keeps this check
    pinned to exactly the one thing this commit changed: the storage step of
    the SAME batch kernel invocation, for the SAME device.
    """
    ok_all = all_ok
    print("\n== narrow-on-store: default dtype == astype(float32) of fp64, bitwise ==")

    results = {}
    for dev in _devices(cuda_available):
        ref_time, ref_traj64 = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                           device=dev, dtype=np.float64, separateTime=True,
                                           verbose=False)
        ref_traj32 = ref_traj64.astype(np.float32)
        out_time, out_traj = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                         device=dev, separateTime=True, verbose=False)
        results[dev] = out_traj
        dtype_ok = out_traj.dtype == np.float32
        match = np.array_equal(out_traj, ref_traj32) and np.array_equal(out_time, ref_time)
        ok = dtype_ok and match
        print(f"  {'OK  ' if ok else 'FAIL'} device={dev!r} : dtype={out_traj.dtype} "
              f"(want float32), bitwise match to astype(float32)-of-fp64={match}")
        if not ok:
            ok_all = False

    # also check the deprecated separateTime=False (per-orbit object array) format
    for dev in _devices(cuda_available):
        ref_legacy = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                 device=dev, dtype=np.float64, verbose=False)
        out_legacy = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                 device=dev, verbose=False)
        match = all(np.array_equal(out_legacy[k][0], ref_legacy[k][0]) and
                    np.array_equal(out_legacy[k][1], ref_legacy[k][1].astype(np.float32))
                    for k in range(len(ic)))
        print(f"  {'OK  ' if match else 'FAIL'} device={dev!r} separateTime=False : "
              f"bitwise match to astype(float32)-of-fp64={match}")
        if not match:
            ok_all = False

    # cross-device consistency (cuda should equal cpu/openmp/serial exactly)
    if cuda_available and len(results) > 1:
        base = results['cpu']
        cross_ok = all(np.array_equal(results[d], base) for d in results if d != 'cpu')
        print(f"  {'OK  ' if cross_ok else 'FAIL'} cuda/cpu/openmp/serial cross-device : "
              f"all bit-identical={cross_ok}")
        if not cross_ok:
            ok_all = False

    return ok_all


# ---------------------------------------------------------------------------
# Check 2 (required): non-trivial unit system still bitwise-correct
# ---------------------------------------------------------------------------

def test_nontrivial_units_still_fallback(pot, ic, T, trajsize, cuda_available, all_ok):
    """Under a NON-trivial unit system, narrowStoreDefault must be false (the
    code keeps taking the storeTrajectoryGPUImpl fallback), and the result
    must still be bitwise correct there -- i.e. bit-identical to the
    pre-existing (unaffected) recipe `float(double_value / lengthUnit)`
    applied to the fp64 integration result.

    Also asserts the unit system actually took effect (so this check cannot
    pass vacuously if setUnits were silently a no-op, which is a real bug
    class in this codebase -- see agama_migrate_setunits_crossinstall in
    project memory): the same numeric `ic`/`time` arrays, reinterpreted
    under the new unit system, integrate a physically different scenario
    (the same potential object's internal representation is unaffected by a
    later setUnits() call, but the ic/time conversion to internal units and
    the output conversion back both use the freshly active factors), so the
    returned trajectory must differ from the trivial-unit-system baseline.
    """
    ok_all = all_ok
    print("\n== narrow-on-store: non-trivial units keep the fallback, bitwise-correct ==")
    dev0 = _devices(cuda_available)[0]

    baseline = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                           device=dev0, dtype=np.float64, separateTime=True, verbose=False)

    agama.setUnits(length=2.0, velocity=3.0, mass=4e6)
    try:
        # non-vacuousness: the non-trivial unit system must actually change the
        # returned (user-unit) values relative to the trivial-unit baseline
        # (same device, same dtype=float64 call, only the active unit system differs)
        changed_ref = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                  device=dev0, dtype=np.float64, separateTime=True, verbose=False)
        changed = not np.array_equal(changed_ref[1], baseline[1])
        print(f"  {'OK  ' if changed else 'FAIL'} setUnits actually changed the result "
              f"(non-vacuous check): changed={changed}")
        if not changed:
            ok_all = False

        # Per-device reference (device=dev, dtype=float64) -- the SAME device and
        # method, still an unaffected combination (directToDest, storage width ==
        # integration precision), just under the now-active non-trivial units.
        for dev in _devices(cuda_available):
            ref_time, ref_traj64 = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                               device=dev, dtype=np.float64, separateTime=True,
                                               verbose=False)
            ref_traj32 = ref_traj64.astype(np.float32)
            out_time, out_traj = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                             device=dev, separateTime=True, verbose=False)
            dtype_ok = out_traj.dtype == np.float32
            match = np.array_equal(out_traj, ref_traj32) and np.array_equal(out_time, ref_time)
            ok = dtype_ok and match
            print(f"  {'OK  ' if ok else 'FAIL'} device={dev!r} non-trivial units : "
                  f"bitwise match to astype(float32)-of-fp64={match}")
            if not ok:
                ok_all = False
    finally:
        agama.setUnits()  # reset to defaults for subsequent tests

    return ok_all


# ---------------------------------------------------------------------------
# Check 3 (required): explicit float32/float64 dtype unchanged
# ---------------------------------------------------------------------------

def test_explicit_dtype_unchanged(pot, ic, T, trajsize, cuda_available, all_ok):
    """Explicit dtype=float32 (fp32 integration, directToDest<float,float>)
    and dtype=float64 (fp64 integration, directToDest<double,double>) are
    untouched by this change -- neither is the new default+trivial-unit
    combination (narrowStoreDefault requires dtype absent/float32 AND fp64
    integration; an explicit dtype makes deviceFp32 track that dtype exactly,
    so directToDest's pre-existing condition is what fires either way).

    The host policies (cpu/openmp/serial) are all g++-compiled and run the
    identical instantiation, so dtype=float64 must be exactly bit-identical
    across them (matching test_orbit_gpu.py's test_serial_cpu_openmp_identical).
    cuda is compiled by nvcc and is allowed to differ at the ~1e-9 relative
    level for fp64 (FMA contraction -- see project findings.md, "Fork vs
    upstream: FMA, not math"; test_orbit_gpu.py's own fp64 parity check uses
    the same tol_rel=1e-9), so it is checked with a loose tolerance, not
    exact equality -- this is pre-existing behaviour this commit does not
    touch (the dtype=float64/float32 combinations were directToDest before
    this change too, and still are)."""
    ok_all = all_ok
    print("\n== explicit dtype=float32/float64 unchanged ==")
    devices = _devices(cuda_available)
    host_devices = [d for d in devices if d != 'cuda']

    for dtype in (np.float64, np.float32):
        host_out = {}
        for dev in host_devices:
            out = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                              device=dev, dtype=dtype, separateTime=True, verbose=False)
            host_out[dev] = out
        ref = host_out[host_devices[0]]
        for dev in host_devices:
            out = host_out[dev]
            match = np.array_equal(out[1], ref[1]) and np.array_equal(out[0], ref[0])
            ok = match and out[1].dtype == dtype
            print(f"  {'OK  ' if ok else 'FAIL'} device={dev!r} dtype={dtype.__name__} : "
                  f"bitwise vs {host_devices[0]!r}={match}")
            if not ok:
                ok_all = False
        if cuda_available:
            # fp64: FMA-contraction-level agreement (1e-9, matching test_orbit_gpu.py's
            # test_fp64_parity). fp32: integration itself accumulates much more error
            # over adaptive steps regardless of device (matching test_orbit_gpu.py's
            # test_fp32_integration tol_rel=1e-3) -- this is not something this commit
            # changes, so a loose tolerance is the right check here, not exact equality.
            tol = 1e-9 if dtype == np.float64 else 1e-3
            out_cuda = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                   device='cuda', dtype=dtype, separateTime=True, verbose=False)
            rel_err = float(np.max(np.abs(out_cuda[1].astype(np.float64) - ref[1].astype(np.float64))) /
                            max(float(np.max(np.abs(ref[1]))), 1e-30))
            ok = out_cuda[1].dtype == dtype and rel_err < tol
            print(f"  {'OK  ' if ok else 'FAIL'} device='cuda' dtype={dtype.__name__} : "
                  f"dtype ok, max rel err vs cpu = {rel_err:.3e} (tol {tol:.1e})")
            if not ok:
                ok_all = False
    return ok_all


# ---------------------------------------------------------------------------
# Check 4 (required): NaN slots still NaN in the fp32 destination
# ---------------------------------------------------------------------------

def test_nan_fill_on_step_limit(pot, cuda_available, all_ok):
    """Force the step limit with a tiny maxNumSteps so some trajectory samples
    are never reached, and confirm they are still NaN in the fp32
    destination under the new narrow-on-store default-dtype path (TOut(NAN)
    must still read as NaN for TOut=float)."""
    ok_all = all_ok
    print("\n== narrow-on-store: NaN fill on step-limit still NaN in fp32 dest ==")
    rng = np.random.default_rng(7)
    n = 8
    r = rng.uniform(0.5, 3.0, n)
    phi = rng.uniform(0, 2*np.pi, n)
    pos = np.column_stack([r*np.cos(phi), r*np.sin(phi), np.zeros(n)])
    vcirc = np.sqrt(np.maximum(-np.sum(pot.force(pos)*pos, axis=1), 0.0))
    vel = np.column_stack([-vcirc*np.sin(phi), vcirc*np.cos(phi), np.zeros(n)])
    ic = np.column_stack([pos, vel])
    T = np.full(n, 200.0)   # long enough that a tiny maxNumSteps cannot finish
    trajsize = 64

    for dev in _devices(cuda_available):
        out_time, out_traj = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                         device=dev, separateTime=True,
                                         maxNumSteps=3, verbose=False)
        has_nan = np.isnan(out_traj).any()
        dtype_ok = out_traj.dtype == np.float32
        # the first sample (initial conditions) must always be real, never NaN
        first_ok = not np.isnan(out_traj[:, 0, :]).any()
        ok = has_nan and dtype_ok and first_ok
        print(f"  {'OK  ' if ok else 'FAIL'} device={dev!r} : has_nan={has_nan}, "
              f"dtype={out_traj.dtype}, sample0 finite={first_ok}")
        if not ok:
            ok_all = False
    return ok_all


# ---------------------------------------------------------------------------
# Check 5 (required): separateTime=False and complex dtypes route as before
# ---------------------------------------------------------------------------

def test_separatetime_false_and_complex_unaffected(pot, ic, T, trajsize, cuda_available, all_ok):
    """separateTime=False (deprecated per-orbit object-array format) and the
    complex64/complex128 storage dtypes never satisfy narrowStoreDefault
    (which requires separateTime!=0 and dtype==NPY_FLOAT), so they must be
    completely unaffected by this change: basic functional smoke checks."""
    ok_all = all_ok
    print("\n== separateTime=False / complex dtypes unaffected ==")
    dev = 'cuda' if cuda_available else 'cpu'

    # separateTime=False, default dtype
    out = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                      device=dev, verbose=False)
    ok = len(out) == len(ic) and out[0][1].dtype == np.float32 and out[0][1].shape[1] == 6
    print(f"  {'OK  ' if ok else 'FAIL'} device={dev!r} separateTime=False, default dtype")
    if not ok:
        ok_all = False

    for cdtype, ncols in [(np.complex64, 3), (np.complex128, 3)]:
        out_c = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                            device=dev, dtype=cdtype, separateTime=True, verbose=False)
        traj_c = out_c[1]
        ok = traj_c.dtype == cdtype and traj_c.shape[-1] == ncols and np.isfinite(traj_c).all()
        print(f"  {'OK  ' if ok else 'FAIL'} device={dev!r} dtype={cdtype.__name__} : "
              f"dtype/shape/finite={ok}")
        if not ok:
            ok_all = False
    return ok_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pot = agama.Potential(
        dict(type='Plummer',       mass=0.1, scaleRadius=0.3),
        dict(type='MiyamotoNagai', mass=0.5, scaleRadius=1.0, scaleHeight=0.3),
        dict(type='NFW',           mass=10.0, scaleRadius=5.0),
    )
    NORB, TRAJSIZE = 32, 32
    ic, T = _make_ics(pot, n=NORB, seed=42)

    cuda_available = True
    try:
        agama.orbit(potential=pot, ic=ic[:1], time=5.0, trajsize=4,
                    device='cuda', dtype=np.float64, verbose=False)
    except RuntimeError as e:
        if 'built without CUDA support' in str(e):
            cuda_available = False
            print("NOTE: CUDA not available in this build; cuda checks will be omitted.")
        else:
            raise

    all_ok = True
    all_ok = test_default_matches_old_fallback_recipe(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_nontrivial_units_still_fallback(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_explicit_dtype_unchanged(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_nan_fill_on_step_limit(pot, cuda_available, all_ok)
    all_ok = test_separatetime_false_and_complex_unaffected(pot, ic, T, TRAJSIZE, cuda_available, all_ok)

    print()
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
