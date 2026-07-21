"""Tier 3 Python smoke test: agama_migrate.orbit(device=..., dtype=...) parity.

Tests the batch orbit integration path exposed via the `device=` kwarg on
agama_migrate.orbit(), which dispatches to integrateOrbitsGPU<T> under the
Serial / OpenMP / Cuda backend.  The legacy path (no device kwarg) is the
reference in all correctness checks.

Composite potential: Plummer bulge (M=0.1, rs=0.3) + MiyamotoNagai disk
(M=0.5, rs=1.0, rz=0.3) + NFW halo (M=10, rs=5).  ICs: 32 near-circular
orbits at r=0.5..3, T=20..50, trajsize=32.

Tests covered:
  1. fp64 parity: each of {cpu, openmp, serial, cuda} vs legacy, tol 1e-9.
  2. Default-dtype (no dtype kwarg): device output == legacy at float32.
  3. fp32 integration (dtype=np.float32, device='cuda'): loose parity +
     energy conservation |dE/E| < 1e-2.
  4. Per-orbit time array (different T per orbit).
  5. Negative-time (backward integration).
  6. timestart shift.
  7. Single-orbit ic shape: (6,) returns (times, traj) pair directly.
  8. Error cases: exact exception types for all rejected combinations.
  9. serial == cpu == openmp bit-for-bit.
  10. cuda: SKIP cleanly if the build lacks CUDA support.
  11. Path A thread concurrency: results bit-identical solo vs concurrent.
  12. Mixed-precision concurrency: fp64 + fp32 threads vs solo.
  13. Exception safety under concurrency: failing thread does not corrupt valid ones.
"""
import sys
import numpy as np
import concurrent.futures
import agama_migrate as agama

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _energy(pot, traj):
    """Specific energy of each snapshot row; traj shape (M, 6)."""
    pos = traj[:, :3]
    vel = traj[:, 3:]
    return pot.potential(pos) + 0.5 * np.sum(vel**2, axis=1)


def _compare_trajectories(dev_orbits, ref_orbits, tol_rel, tag, all_ok,
                           require_exact_times=True, max_report=3):
    """Compare device-path trajectory output against a reference.

    dev_orbits / ref_orbits: (N,2) object arrays as returned by agama.orbit().
    tol_rel: maximum allowed relative error (0 means exact fp comparison).

    Relative error is computed per orbit as:
        max_abs_err / max(max(|ref|), max(|dev|), 1e-30)
    This mirrors crosscheck_orbits.py and avoids inflated relative errors when
    individual trajectory components pass through zero (e.g. z-velocity).

    Returns updated all_ok.
    """
    N = len(dev_orbits)
    worst_err = 0.0
    times_ok = True
    for k in range(N):
        t_dev, tr_dev = dev_orbits[k]
        t_ref, tr_ref = ref_orbits[k]
        if require_exact_times and not np.array_equal(t_dev, t_ref):
            if times_ok:  # report only first failure
                print(f"  FAIL {tag} : orbit {k} time arrays differ "
                      f"(dev[0]={t_dev[0]:.6g} ref[0]={t_ref[0]:.6g})")
            times_ok = False
            all_ok = False
        tr_dev64 = tr_dev.astype(np.float64)
        tr_ref64 = tr_ref.astype(np.float64)
        if tol_rel == 0.0:
            if not np.array_equal(tr_dev64, tr_ref64):
                all_ok = False
                worst_err = 1.0  # flag
        else:
            # Per-orbit scale: max of both arrays so near-zero ref values
            # don't inflate the relative error artificially.
            scale = max(float(np.max(np.abs(tr_ref64))),
                        float(np.max(np.abs(tr_dev64))),
                        1e-30)
            err = float(np.max(np.abs(tr_dev64 - tr_ref64))) / scale
            if err > worst_err:
                worst_err = err
    if tol_rel == 0.0:
        ok = (worst_err == 0.0) and times_ok
        print(f"  {'OK  ' if ok else 'FAIL'} {tag} : "
              f"bit-for-bit match ({N} orbits)")
    else:
        ok = (worst_err <= tol_rel) and times_ok
        print(f"  {'OK  ' if ok else 'FAIL'} {tag} : "
              f"max rel err = {worst_err:.3e}  (tol {tol_rel:.1e}, {N} orbits)")
    if not ok:
        all_ok = False
    return all_ok


def _make_ics(pot, n=32, seed=42):
    """Near-circular ICs at r=0.5..3, times=20..50."""
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


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

def test_fp64_parity(pot, ic, T, trajsize, cuda_available, all_ok):
    """fp64 parity: device={cpu,openmp,serial,cuda} vs legacy, tol 1e-9."""
    print("\n== fp64 parity: {cpu, openmp, serial, cuda} vs legacy ==")

    # Reference: legacy path, explicit fp64.
    ref = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                      dtype=np.float64, verbose=False)

    devices = ["cpu", "openmp", "serial"]
    if cuda_available:
        devices.append("cuda")

    for dev in devices:
        try:
            out = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                              device=dev, dtype=np.float64, verbose=False)
            # time arrays must be exactly equal
            times_match = all(np.array_equal(out[k][0], ref[k][0])
                              for k in range(len(ic)))
            if not times_match:
                print(f"  FAIL device={dev!r} fp64 : time arrays differ from legacy")
                all_ok = False
                continue
            all_ok = _compare_trajectories(out, ref, tol_rel=1e-9,
                                           tag=f"device={dev!r} fp64",
                                           all_ok=all_ok)
        except Exception as e:
            print(f"  FAIL device={dev!r} fp64 : {type(e).__name__}: {e}")
            all_ok = False

    if not cuda_available:
        print(f"  SKIP device='cuda' fp64 : no CUDA support in this build")

    return all_ok


def test_dprkn8_parity(pot, ic, T, trajsize, cuda_available, all_ok):
    """DPRKN8 method on the device path: device={cpu,openmp,serial,cuda} with
    method='dprkn8' must match the legacy CPU DPRKN8 integrator. The batch cores
    share the exact same dprkn8_step/dense as the legacy OdeStepperDPRKN8 and the
    same 10*accuracy^0.9 rescaling, so serial/cpu/openmp agree with legacy to
    round-off (composite force summation order differs by ULPs, as for DOP853);
    cuda agrees within the same fp64 tolerance."""
    print("\n== DPRKN8 parity: {cpu, openmp, serial, cuda} vs legacy DPRKN8 ==")

    # Reference: legacy CPU path, DPRKN8, explicit fp64.
    ref = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                      method='dprkn8', dtype=np.float64, verbose=False)

    devices = ["cpu", "openmp", "serial"]
    if cuda_available:
        devices.append("cuda")

    for dev in devices:
        try:
            out = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                              method='dprkn8', device=dev, dtype=np.float64,
                              verbose=False)
            times_match = all(np.array_equal(out[k][0], ref[k][0])
                              for k in range(len(ic)))
            if not times_match:
                print(f"  FAIL device={dev!r} dprkn8 : time arrays differ from legacy")
                all_ok = False
                continue
            all_ok = _compare_trajectories(out, ref, tol_rel=1e-9,
                                           tag=f"device={dev!r} dprkn8",
                                           all_ok=all_ok)
        except Exception as e:
            print(f"  FAIL device={dev!r} dprkn8 : {type(e).__name__}: {e}")
            all_ok = False

    if not cuda_available:
        print(f"  SKIP device='cuda' dprkn8 : no CUDA support in this build")

    return all_ok


def test_default_dtype_parity(pot, ic, T, trajsize, cuda_available, all_ok):
    """Default-dtype: device output must equal legacy at float32, bit-for-bit.

    Legacy default is dtype=np.float32 (fp64 integration, fp32 storage).
    Device path with no dtype kwarg should produce the same fp64 integration
    stored as fp32 -- bit-for-bit identical to the legacy result.
    """
    print("\n== Default-dtype (no dtype kwarg): device output == legacy float32 ==")

    # Legacy reference -- no dtype kwarg, so default float32 storage.
    ref = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                      verbose=False)

    devices = ["cpu", "openmp", "serial"]
    if cuda_available:
        devices.append("cuda")

    for dev in devices:
        try:
            out = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                              device=dev, verbose=False)
            # Check storage dtype is float32
            dt = out[0][1].dtype
            dtype_ok = dt == np.float32
            # Check bit-for-bit equality with legacy
            match = all(np.array_equal(out[k][0], ref[k][0]) and
                        np.array_equal(out[k][1], ref[k][1])
                        for k in range(len(ic)))
            ok = dtype_ok and match
            print(f"  {'OK  ' if ok else 'FAIL'} device={dev!r} default dtype : "
                  f"storage={dt} (want float32), bit-for-bit={match}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  FAIL device={dev!r} default dtype : {type(e).__name__}: {e}")
            all_ok = False

    if not cuda_available:
        print(f"  SKIP device='cuda' default dtype : no CUDA support in this build")

    return all_ok


def test_fp32_integration(pot, ic, T, trajsize, cuda_available, all_ok):
    """fp32 integration (dtype=np.float32, device='cuda'): loose parity + energy.

    Expected: max rel err vs fp64 legacy < 1e-3; energy conservation |dE/E| < 1e-2.
    """
    print("\n== fp32 integration (dtype=np.float32, device='cuda') ==")

    if not cuda_available:
        print("  SKIP : no CUDA support in this build")
        return all_ok

    ref = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                      dtype=np.float64, verbose=False)
    try:
        out = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                          device='cuda', dtype=np.float32, verbose=False)
        # Storage dtype must be float32.
        dt = out[0][1].dtype
        dtype_ok = dt == np.float32
        if not dtype_ok:
            print(f"  FAIL fp32 integration : storage dtype {dt} != float32")
            return False

        # Loose parity vs fp64 reference.
        all_ok = _compare_trajectories(out, ref, tol_rel=1e-3,
                                       tag="fp32 integration vs fp64 ref",
                                       all_ok=all_ok,
                                       require_exact_times=True)

        # Energy conservation.  pot.potential(1d xyz) -> scalar float.
        max_dE = 0.0
        for k in range(len(ic)):
            ic_k = ic[k]
            traj = out[k][1].astype(np.float64)
            E0 = float(pot.potential(tuple(ic_k[:3]))) + 0.5*float(np.sum(ic_k[3:]**2))
            E1 = float(pot.potential(tuple(traj[-1, :3]))) + 0.5*float(np.sum(traj[-1, 3:]**2))
            dE = abs((E1 - E0) / E0) if E0 != 0 else abs(E1 - E0)
            if dE > max_dE:
                max_dE = dE
        ok = max_dE < 1e-2
        print(f"  {'OK  ' if ok else 'FAIL'} fp32 energy conservation : "
              f"max |dE/E| = {max_dE:.3e}  (tol 1e-2)")
        if not ok:
            all_ok = False

    except Exception as e:
        print(f"  FAIL fp32 integration : {type(e).__name__}: {e}")
        all_ok = False

    return all_ok


def test_per_orbit_time(pot, ic, T, trajsize, cuda_available, all_ok):
    """Per-orbit time array: different T per orbit (length-N array)."""
    print("\n== Per-orbit time array (different T per orbit) ==")

    # Reference: legacy with the same per-orbit T array.
    ref = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                      dtype=np.float64, verbose=False)

    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")

    for dev in devices:
        try:
            out = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                              device=dev, dtype=np.float64, verbose=False)
            all_ok = _compare_trajectories(out, ref, tol_rel=1e-9,
                                           tag=f"per-orbit time, device={dev!r}",
                                           all_ok=all_ok)
        except Exception as e:
            print(f"  FAIL per-orbit time, device={dev!r} : {type(e).__name__}: {e}")
            all_ok = False

    if not cuda_available:
        print("  SKIP device='cuda' per-orbit time : no CUDA support in this build")

    return all_ok


def test_negative_time(pot, ic, trajsize, cuda_available, all_ok):
    """Negative time (backward integration): compare device vs legacy at -T=10."""
    print("\n== Negative time (backward integration) ==")

    T_neg = -10.0
    ref = agama.orbit(potential=pot, ic=ic[:8], time=T_neg, trajsize=trajsize,
                      dtype=np.float64, verbose=False)

    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")

    for dev in devices:
        try:
            out = agama.orbit(potential=pot, ic=ic[:8], time=T_neg,
                              trajsize=trajsize, device=dev, dtype=np.float64,
                              verbose=False)
            all_ok = _compare_trajectories(out, ref, tol_rel=1e-9,
                                           tag=f"negative time, device={dev!r}",
                                           all_ok=all_ok)
        except Exception as e:
            print(f"  FAIL negative time, device={dev!r} : {type(e).__name__}: {e}")
            all_ok = False

    if not cuda_available:
        print("  SKIP device='cuda' negative time : no CUDA support in this build")

    return all_ok


def test_timestart(pot, ic, T, trajsize, cuda_available, all_ok):
    """timestart shift: time arrays shifted, trajectories match timestart=0 run.

    Static potential -> the trajectory positions are identical regardless of
    timestart; only the time axis is shifted.
    """
    print("\n== timestart shift (static potential) ==")

    T_fixed = 25.0
    ts = 100.0  # timestart offset

    # timestart=0 reference (device path, fp64).
    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")

    for dev in devices:
        try:
            ref = agama.orbit(potential=pot, ic=ic[:8], time=T_fixed,
                              trajsize=trajsize, device=dev, dtype=np.float64,
                              verbose=False)
            out = agama.orbit(potential=pot, ic=ic[:8], time=T_fixed,
                              timestart=ts, trajsize=trajsize, device=dev,
                              dtype=np.float64, verbose=False)

            # Time arrays should be shifted by exactly ts.
            times_shifted_ok = True
            traj_match = True
            for k in range(8):
                t_shift = out[k][0] - ts
                if not np.allclose(t_shift, ref[k][0], rtol=1e-12, atol=0):
                    times_shifted_ok = False
                if not np.allclose(out[k][1].astype(np.float64),
                                   ref[k][1].astype(np.float64),
                                   rtol=1e-12, atol=0):
                    traj_match = False

            ok = times_shifted_ok and traj_match
            print(f"  {'OK  ' if ok else 'FAIL'} timestart={ts}, device={dev!r} : "
                  f"times_shifted={times_shifted_ok}, traj_match={traj_match}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  FAIL timestart, device={dev!r} : {type(e).__name__}: {e}")
            all_ok = False

    if not cuda_available:
        print("  SKIP device='cuda' timestart : no CUDA support in this build")

    return all_ok


def test_single_orbit_shape(pot, ic, T, trajsize, cuda_available, all_ok):
    """Single-orbit ic (shape (6,)) returns (times, traj) pair directly, not (1,2)."""
    print("\n== Single-orbit ic shape: (6,) -> (times, traj) pair ==")

    T_single = 20.0
    ic_single = ic[0]  # shape (6,)

    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")

    for dev in devices:
        try:
            out = agama.orbit(potential=pot, ic=ic_single, time=T_single,
                              trajsize=trajsize, device=dev, dtype=np.float64,
                              verbose=False)
            # For a single orbit, agama returns the pair directly, not a (1,2) array.
            is_pair = (isinstance(out, (tuple, list, np.ndarray)) and
                       len(out) == 2)
            times_arr = out[0]
            traj_arr  = out[1]
            shape_ok = (isinstance(times_arr, np.ndarray) and
                        times_arr.ndim == 1 and len(times_arr) == trajsize and
                        isinstance(traj_arr, np.ndarray) and
                        traj_arr.ndim == 2 and traj_arr.shape == (trajsize, 6))
            ok = is_pair and shape_ok
            print(f"  {'OK  ' if ok else 'FAIL'} single orbit, device={dev!r} : "
                  f"is_pair={is_pair}, times.shape={times_arr.shape}, "
                  f"traj.shape={traj_arr.shape}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  FAIL single orbit, device={dev!r} : {type(e).__name__}: {e}")
            all_ok = False

    if not cuda_available:
        print("  SKIP device='cuda' single orbit : no CUDA support in this build")

    return all_ok


def test_serial_cpu_openmp_identical(pot, ic, T, trajsize, all_ok):
    """serial == cpu == openmp: same code, deterministic per orbit -> bit-for-bit."""
    print("\n== serial == cpu == openmp : bit-for-bit identical ==")

    try:
        out_serial = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                 device='serial', dtype=np.float64, verbose=False)
        out_cpu    = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                 device='cpu',    dtype=np.float64, verbose=False)
        out_openmp = agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                                 device='openmp', dtype=np.float64, verbose=False)

        serial_eq_cpu = all(np.array_equal(out_serial[k][0], out_cpu[k][0]) and
                            np.array_equal(out_serial[k][1], out_cpu[k][1])
                            for k in range(len(ic)))
        serial_eq_omp = all(np.array_equal(out_serial[k][0], out_openmp[k][0]) and
                            np.array_equal(out_serial[k][1], out_openmp[k][1])
                            for k in range(len(ic)))
        ok = serial_eq_cpu and serial_eq_omp
        print(f"  {'OK  ' if ok else 'FAIL'} serial==cpu={serial_eq_cpu}, "
              f"serial==openmp={serial_eq_omp}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"  FAIL serial/cpu/openmp identity : {type(e).__name__}: {e}")
        all_ok = False

    return all_ok


def test_error_cases(pot, ic, T, trajsize, cuda_available, all_ok):
    """All NotImplementedError / ValueError / TypeError cases."""
    print("\n== Error cases (exception type and content) ==")

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

    ic1 = ic[:4]
    T1 = float(T[0]) if hasattr(T, '__len__') else T

    # Unknown device string -> ValueError.
    _expect(ValueError, "unknown device='banana'",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=trajsize,
                                device='banana', verbose=False))

    # targets + device -> NotImplementedError.
    tgt = agama.Target(type='KinemShell', degree=0,
                       gridr=np.logspace(-1, 1, 5))
    _expect(NotImplementedError, "device + targets",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=trajsize,
                                device='cpu', targets=tgt, verbose=False))

    # der=True + device -> NotImplementedError.
    _expect(NotImplementedError, "device + der=True",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=trajsize,
                                device='cpu', der=True, verbose=False))

    # lyapunov=True + device -> NotImplementedError.
    _expect(NotImplementedError, "device + lyapunov=True",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=trajsize,
                                device='cpu', lyapunov=True, verbose=False))

    # Omega != 0 + device -> NotImplementedError.
    _expect(NotImplementedError, "device + Omega!=0",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=trajsize,
                                device='cpu', Omega=1.0, verbose=False))

    # method!='dop853' + device -> NotImplementedError.
    _expect(NotImplementedError, "device + method='hermite'",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=trajsize,
                                device='cpu', method='hermite', verbose=False))

    # dtype=object + device -> NotImplementedError.
    _expect(NotImplementedError, "device + dtype=object",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=trajsize,
                                device='cpu', dtype=object, verbose=False))

    # trajsize absent + device -> NotImplementedError.
    _expect(NotImplementedError, "device + no trajsize",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1,
                                device='cpu', verbose=False))

    # trajsize=0 (natural timestep) + device -> NotImplementedError.
    _expect(NotImplementedError, "device + trajsize=0",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=0,
                                device='cpu', verbose=False))

    # Per-orbit differing trajsize + device -> NotImplementedError.
    ts_arr = np.array([trajsize, trajsize, trajsize, trajsize + 1])
    _expect(NotImplementedError, "device + differing per-orbit trajsize",
            lambda: agama.orbit(potential=pot, ic=ic1, time=T1, trajsize=ts_arr,
                                device='cpu', verbose=False))

    # Unsupported potential -> NotImplementedError naming 'Dehnen'.
    # Spherical Dehnen became GPU-capable in Tier 1; a non-spherical Dehnen
    # (axisRatioZ != 1) is still rejected (buildGpuPotDesc gates on sphericity).
    dehnen = agama.Potential(type='Dehnen', mass=1.0, scaleRadius=1.0, axisRatioZ=0.7)
    try:
        agama.orbit(potential=dehnen, ic=ic1, time=T1, trajsize=trajsize,
                    device='cpu', verbose=False)
        print("  FAIL device + Dehnen : no exception raised")
        all_ok = False
    except NotImplementedError as e:
        ok = 'Dehnen' in str(e)
        print(f"  {'OK  ' if ok else 'FAIL'} device + Dehnen : "
              f"raised NotImplementedError (names Dehnen: {ok}): {e}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"  FAIL device + Dehnen : raised {type(e).__name__}: {e}")
        all_ok = False

    # Composite containing a non-spherical (unsupported) Dehnen -> NotImplementedError
    # naming the component.
    pot_bad = agama.Potential(
        dict(type='Plummer', mass=1.0, scaleRadius=1.0),
        dict(type='Dehnen',  mass=1.0, scaleRadius=1.0, axisRatioZ=0.7),
    )
    try:
        agama.orbit(potential=pot_bad, ic=ic1, time=T1, trajsize=trajsize,
                    device='cpu', verbose=False)
        print("  FAIL device + Composite(Plummer+Dehnen) : no exception raised")
        all_ok = False
    except NotImplementedError as e:
        ok = 'Dehnen' in str(e)
        print(f"  {'OK  ' if ok else 'FAIL'} device + Composite(Plummer+Dehnen) : "
              f"raised NotImplementedError (names Dehnen: {ok}): {e}")
        if not ok:
            all_ok = False
    except Exception as e:
        print(f"  FAIL device + Composite(Plummer+Dehnen) : raised {type(e).__name__}: {e}")
        all_ok = False

    return all_ok


def test_two_single_potentials(cuda_available, all_ok):
    """Parity on two individual GPU-capable potentials beyond the composite."""
    print("\n== Two single potentials: Plummer fp64, NFW fp64 parity ==")

    single_pots = [
        ("Plummer", agama.Potential(type='Plummer', mass=1.0, scaleRadius=1.0)),
        ("NFW",     agama.Potential(type='NFW',     mass=1.0, scaleRadius=2.0)),
    ]
    rng = np.random.default_rng(77)
    ic_s = rng.standard_normal((16, 6)) * np.array([1, 1, 0.2, 0.3, 0.3, 0.1])
    T_s  = 15.0
    ts   = 8

    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")

    for name, pot_s in single_pots:
        ref_s = agama.orbit(potential=pot_s, ic=ic_s, time=T_s, trajsize=ts,
                            dtype=np.float64, verbose=False)
        for dev in devices:
            try:
                out_s = agama.orbit(potential=pot_s, ic=ic_s, time=T_s, trajsize=ts,
                                    device=dev, dtype=np.float64, verbose=False)
                all_ok = _compare_trajectories(out_s, ref_s, tol_rel=1e-9,
                                               tag=f"{name} device={dev!r} fp64",
                                               all_ok=all_ok)
            except Exception as e:
                print(f"  FAIL {name} device={dev!r} fp64 : {type(e).__name__}: {e}")
                all_ok = False

    if not cuda_available:
        print("  SKIP device='cuda' single pots : no CUDA support in this build")

    return all_ok


# ---------------------------------------------------------------------------
# Path A thread concurrency tests (CUDA-only)
# ---------------------------------------------------------------------------

def _orbit_call(args):
    """Worker thunk for ThreadPoolExecutor: unpack args, run agama.orbit, return result."""
    pot, ic, T, trajsize, dtype = args
    return agama.orbit(potential=pot, ic=ic, time=T, trajsize=trajsize,
                       device='cuda', dtype=dtype, verbose=False)


def test_path_a_concurrency_parity(pot, cuda_available, all_ok):
    """Path A thread concurrency: concurrent cuda calls are bit-identical to solo.

    Builds 4 distinct IC sets of 2000 orbits each (T=50, trajsize=32, fp64).
    Computes each result solo (sequential), then re-runs all 4 concurrently via
    ThreadPoolExecutor(4).  Repeats the concurrent round 3 times -- stream
    scheduling varies run to run, correctness must hold every time.
    """
    print("\n== Path A thread concurrency: concurrent == solo (fp64, 4 IC sets) ==")

    if not cuda_available:
        print("  SKIP : no CUDA support in this build")
        return all_ok

    NORB_CONC = 2000
    TRAJSIZE_CONC = 32
    T_CONC = 50.0

    # Build 4 distinct IC sets with different seeds.
    ic_sets = []
    for seed in [101, 202, 303, 404]:
        ic_s, _ = _make_ics(pot, n=NORB_CONC, seed=seed)
        ic_sets.append(ic_s)

    # Solo (sequential) reference for each IC set.
    solo_results = []
    for ic_s in ic_sets:
        res = agama.orbit(potential=pot, ic=ic_s, time=T_CONC,
                          trajsize=TRAJSIZE_CONC, device='cuda',
                          dtype=np.float64, verbose=False)
        solo_results.append(res)

    print(f"  Solo baselines computed: {len(ic_sets)} IC sets x {NORB_CONC} orbits each")

    # Run concurrent rounds.
    N_ROUNDS = 3
    all_rounds_ok = True
    for rnd in range(N_ROUNDS):
        work_args = [(pot, ic_sets[i], T_CONC, TRAJSIZE_CONC, np.float64)
                     for i in range(len(ic_sets))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(_orbit_call, a) for a in work_args]
            conc_results = [f.result() for f in futures]

        round_ok = True
        for i, (conc_res, solo_res) in enumerate(zip(conc_results, solo_results)):
            for k in range(NORB_CONC):
                t_conc, tr_conc = conc_res[k]
                t_solo, tr_solo = solo_res[k]
                if not np.array_equal(t_conc, t_solo):
                    print(f"  FAIL round {rnd+1} IC set {i} orbit {k}: "
                          f"time arrays differ")
                    round_ok = False
                    all_rounds_ok = False
                    break
                if not np.array_equal(tr_conc, tr_solo):
                    diff = np.abs(tr_conc.astype(np.float64)
                                  - tr_solo.astype(np.float64))
                    print(f"  FAIL round {rnd+1} IC set {i} orbit {k}: "
                          f"traj arrays differ, max abs diff = {diff.max():.6e}")
                    round_ok = False
                    all_rounds_ok = False
                    break
            if not round_ok:
                break

        status = "OK  " if round_ok else "FAIL"
        print(f"  {status} round {rnd+1}/{N_ROUNDS}: "
              f"4 concurrent cuda calls bit-identical to solo "
              f"({len(ic_sets)} x {NORB_CONC} orbits)")

    if not all_rounds_ok:
        all_ok = False

    return all_ok


def test_path_a_mixed_precision_concurrency(pot, cuda_available, all_ok):
    """Mixed-precision concurrency: 2 fp64 + 2 fp32 threads concurrent vs solo.

    Each thread's result must be bit-exactly equal to its solo counterpart.
    fp64 threads use the same IC sets as the concurrency parity test; fp32
    threads use two fresh IC sets.
    """
    print("\n== Path A mixed-precision concurrency: 2xfp64 + 2xfp32 threads ==")

    if not cuda_available:
        print("  SKIP : no CUDA support in this build")
        return all_ok

    NORB_CONC = 2000
    TRAJSIZE_CONC = 32
    T_CONC = 50.0

    # 2 fp64 IC sets + 2 fp32 IC sets.
    ic_fp64 = []
    for seed in [501, 602]:
        ic_s, _ = _make_ics(pot, n=NORB_CONC, seed=seed)
        ic_fp64.append(ic_s)

    ic_fp32 = []
    for seed in [703, 804]:
        ic_s, _ = _make_ics(pot, n=NORB_CONC, seed=seed)
        ic_fp32.append(ic_s)

    # Solo references.
    solo_fp64 = [
        agama.orbit(potential=pot, ic=ic_s, time=T_CONC, trajsize=TRAJSIZE_CONC,
                    device='cuda', dtype=np.float64, verbose=False)
        for ic_s in ic_fp64
    ]
    solo_fp32 = [
        agama.orbit(potential=pot, ic=ic_s, time=T_CONC, trajsize=TRAJSIZE_CONC,
                    device='cuda', dtype=np.float32, verbose=False)
        for ic_s in ic_fp32
    ]

    # Build work list: [fp64_0, fp64_1, fp32_0, fp32_1].
    dtypes = [np.float64, np.float64, np.float32, np.float32]
    ic_all = ic_fp64 + ic_fp32
    solo_all = solo_fp64 + solo_fp32
    work_args = [(pot, ic_all[i], T_CONC, TRAJSIZE_CONC, dtypes[i])
                 for i in range(4)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_orbit_call, a) for a in work_args]
        conc_results = [f.result() for f in futures]

    labels = ["fp64-0", "fp64-1", "fp32-0", "fp32-1"]
    section_ok = True
    for i, (conc_res, solo_res, label) in enumerate(
            zip(conc_results, solo_all, labels)):
        thread_ok = True
        for k in range(NORB_CONC):
            t_conc, tr_conc = conc_res[k]
            t_solo, tr_solo = solo_res[k]
            if not np.array_equal(t_conc, t_solo):
                print(f"  FAIL thread {label} orbit {k}: time arrays differ")
                thread_ok = False
                break
            if not np.array_equal(tr_conc, tr_solo):
                diff = np.abs(tr_conc.astype(np.float64)
                              - tr_solo.astype(np.float64))
                print(f"  FAIL thread {label} orbit {k}: "
                      f"traj arrays differ, max abs diff = {diff.max():.6e}")
                thread_ok = False
                break
        status = "OK  " if thread_ok else "FAIL"
        print(f"  {status} thread {label}: "
              f"concurrent result bit-identical to solo ({NORB_CONC} orbits)")
        if not thread_ok:
            section_ok = False

    if not section_ok:
        all_ok = False

    return all_ok


def test_path_a_exception_safety(pot, cuda_available, all_ok):
    """Exception safety under concurrency: one bad thread must not corrupt valid ones.

    4 threads run concurrently.  Thread 0 uses an unsupported (non-spherical)
    Dehnen potential and must raise exactly NotImplementedError.  Threads 1-3 run
    valid cuda calls and their results must be bit-identical to solo runs.
    """
    print("\n== Path A exception safety: 1 bad thread + 3 valid threads concurrent ==")

    if not cuda_available:
        print("  SKIP : no CUDA support in this build")
        return all_ok

    NORB_CONC = 2000
    TRAJSIZE_CONC = 32
    T_CONC = 50.0

    dehnen = agama.Potential(type='Dehnen', mass=1.0, scaleRadius=1.0, axisRatioZ=0.7)

    ic_valid = []
    for seed in [901, 1002, 1103]:
        ic_s, _ = _make_ics(pot, n=NORB_CONC, seed=seed)
        ic_valid.append(ic_s)

    # Dummy IC for the bad thread (small; it should raise before integrating).
    ic_bad, _ = _make_ics(pot, n=4, seed=999)

    # Solo references for the 3 valid threads.
    solo_valid = [
        agama.orbit(potential=pot, ic=ic_s, time=T_CONC, trajsize=TRAJSIZE_CONC,
                    device='cuda', dtype=np.float64, verbose=False)
        for ic_s in ic_valid
    ]

    # Submit: thread 0 = bad (Dehnen), threads 1-3 = valid.
    def bad_call():
        return agama.orbit(potential=dehnen, ic=ic_bad, time=T_CONC,
                           trajsize=TRAJSIZE_CONC, device='cuda',
                           dtype=np.float64, verbose=False)

    good_args = [(pot, ic_valid[j], T_CONC, TRAJSIZE_CONC, np.float64)
                 for j in range(3)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_bad = ex.submit(bad_call)
        f_good = [ex.submit(_orbit_call, a) for a in good_args]
        # Collect good results first (they should succeed).
        good_results = []
        good_exc = None
        for f in f_good:
            try:
                good_results.append(f.result())
            except Exception as e:
                good_exc = e
                good_results.append(None)
        # Now collect bad result.
        bad_exc = None
        try:
            f_bad.result()
            bad_exc = None  # no exception -- unexpected
        except Exception as e:
            bad_exc = e

    # Check bad thread raised NotImplementedError.
    bad_ok = isinstance(bad_exc, NotImplementedError)
    print(f"  {'OK  ' if bad_ok else 'FAIL'} bad thread raised "
          f"{type(bad_exc).__name__ if bad_exc is not None else 'nothing'}"
          f"{' (want NotImplementedError)' if not bad_ok else ''}"
          f": {bad_exc}")
    if not bad_ok:
        all_ok = False

    # Check good threads not corrupted.
    if good_exc is not None:
        print(f"  FAIL valid thread raised unexpected {type(good_exc).__name__}: "
              f"{good_exc}")
        all_ok = False
    else:
        labels = ["valid-0", "valid-1", "valid-2"]
        for j, (conc_res, solo_res, label) in enumerate(
                zip(good_results, solo_valid, labels)):
            if conc_res is None:
                print(f"  FAIL thread {label}: no result (exception was raised)")
                all_ok = False
                continue
            thread_ok = True
            for k in range(NORB_CONC):
                t_conc, tr_conc = conc_res[k]
                t_solo, tr_solo = solo_res[k]
                if not np.array_equal(t_conc, t_solo):
                    print(f"  FAIL thread {label} orbit {k}: time arrays differ")
                    thread_ok = False
                    break
                if not np.array_equal(tr_conc, tr_solo):
                    diff = np.abs(tr_conc.astype(np.float64)
                                  - tr_solo.astype(np.float64))
                    print(f"  FAIL thread {label} orbit {k}: "
                          f"traj arrays differ, max abs diff = {diff.max():.6e}")
                    thread_ok = False
                    break
            status = "OK  " if thread_ok else "FAIL"
            print(f"  {status} thread {label}: "
                  f"result bit-identical to solo despite bad sibling thread "
                  f"({NORB_CONC} orbits)")
            if not thread_ok:
                all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    # Composite potential matching crosscheck_orbits.py for comparability.
    pot = agama.Potential(
        dict(type='Plummer',       mass=0.1, scaleRadius=0.3),
        dict(type='MiyamotoNagai', mass=0.5, scaleRadius=1.0, scaleHeight=0.3),
        dict(type='NFW',           mass=10.0, scaleRadius=5.0),
    )

    NORB    = 32
    TRAJSIZE = 32
    ic, T = _make_ics(pot, n=NORB, seed=42)

    # Probe for CUDA support by attempting a tiny orbit with device='cuda';
    # mirrors how test_potential_gpu.py probes for GPU availability.
    cuda_available = True
    ic_probe = ic[:1]
    try:
        agama.orbit(potential=pot, ic=ic_probe, time=5.0, trajsize=4,
                    device='cuda', dtype=np.float64, verbose=False)
    except RuntimeError as e:
        if 'built without CUDA support' in str(e):
            cuda_available = False
            print("NOTE: CUDA not available in this build; cuda tests will SKIP.")
        else:
            raise  # re-raise unexpected RuntimeError

    all_ok = True

    all_ok = test_fp64_parity(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_dprkn8_parity(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_default_dtype_parity(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_fp32_integration(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_per_orbit_time(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_negative_time(pot, ic, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_timestart(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_single_orbit_shape(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_serial_cpu_openmp_identical(pot, ic, T, TRAJSIZE, all_ok)
    all_ok = test_error_cases(pot, ic, T, TRAJSIZE, cuda_available, all_ok)
    all_ok = test_two_single_potentials(cuda_available, all_ok)
    all_ok = test_path_a_concurrency_parity(pot, cuda_available, all_ok)
    all_ok = test_path_a_mixed_precision_concurrency(pot, cuda_available, all_ok)
    all_ok = test_path_a_exception_safety(pot, cuda_available, all_ok)

    # Count OK/FAIL lines for a summary matching house style.
    # We can't count them directly (tests report inline), so derive from all_ok.
    # Print final status in the house "N OK / M FAIL" style, followed by PASS/FAIL.
    print()
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
