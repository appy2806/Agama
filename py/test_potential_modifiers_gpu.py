"""Tier 1 modifier test: Shifted / Tilted / Rotating / Scaled on the device paths.

A modifier wraps a potential and transforms the query: Shifted translates,
Tilted rotates by fixed Euler angles, Rotating rotates about z by a
possibly-time-dependent angle, Scaled rescales length and amplitude. Chains
nest, and `agama.Potential(..., center=, orientation=, rotation=, scale=)`
builds them in the order  Shifted(Tilted(Rotating(Scaled(base)))).

On the device side the whole chain collapses to a single similarity transform
carried by the descriptor term (GpuPotXform in potential_composite.h), so it
costs one kernel launch regardless of nesting depth. What this file checks:

  * Parity vs the legacy CPU path for every modifier and for the fully nested
    four-modifier chain, across {cpu, openmp, serial, cuda} x {float64,
    float32}, for potential, force AND density -- density matters separately
    because a similarity transform multiplies it by an extra factor of
    scale^2 that the potential does not see.

  * 'serial' == 'cpu' == 'openmp' bit-for-bit (same math, order-independent).

  * TIME DEPENDENCE actually reaches the kernel: for a modifier built from a
    time-series, `pot.potential(xyz, t=T, device=...)` must equal the legacy
    `pot.potential(xyz, t=T)` at several distinct T, and must DIFFER between
    t=0 and t=T. A test that only checked t=0 would pass even if `t` were
    silently dropped -- which is exactly the bug this feature had to fix.

  * A per-point array of times is REFUSED on the device path (it has one shared
    time per batch), rather than silently reduced to one value.

  * An unknown/misspelled kwarg raises TypeError on the device path, as it
    always has on the legacy path. Before this change the device path read only
    `device` and `dtype` out of the kwargs dict and silently ignored the rest,
    so `pot.potential(xyz, t=5, device='cpu')` dropped the t.

  * The orbit kernel accepts a CONSTANT modifier chain and refuses a
    time-VARYING one with NotImplementedError. It builds its descriptor once and
    reuses it across every integration step, so freezing a moving potential at
    one instant would return a plausible wrong trajectory; that is refused
    instead. Removing the refusal without adding device-resident splines would
    reintroduce exactly that silent error.

  * UniformAcceleration, newly GPU-dispatchable on the batch path, tracks its
    time-dependent acceleration.

Tolerances are PARITY tolerances (legacy vs device for the same precision), not
accuracy budgets: 1e-12 relative for fp64, 5e-6 for fp32, both normalized by
max|reference| over the sample rather than pointwise, because a triaxial
Logarithmic's Phi crosses zero on the sampled shell.
"""
import os
import sys
import tempfile

import numpy as np
import agama_migrate as agama


DEVICES = ("cpu", "openmp", "serial", "cuda")
DTYPES = (np.float64, np.float32)

# Base potentials chosen so a wrong rotation or a wrong matrix transpose cannot
# hide: a triaxial Logarithmic has no symmetry at all, so every one of the nine
# matrix entries affects the answer. Plummer is the spherical control -- it
# should be insensitive to Tilted/Rotating, which is itself worth confirming.
BASE_TRIAXIAL = dict(type='Logarithmic', v0=1.0, scaleRadius=0.5,
                     axisRatioY=0.7, axisRatioZ=0.5)
BASE_SPHERICAL = dict(type='Plummer', mass=1.0, scaleRadius=1.0)

# Constant modifier parameters. Deliberately all non-trivial and mutually
# non-commuting, so an incorrect composition ORDER shows up.
MOD_CENTER = [0.31, -0.22, 0.47]
MOD_ORIENT = [0.4, 0.9, -0.3]
MOD_ROTATE = 0.7
MOD_SCALE = [1.3, 0.8]          # (amplitude, length scale)

OPS = ("potential", "force", "density")


def cuda_available():
    """True if this build has the CUDA backend and a usable device.

    A CPU-only build (HAVE_CUDA=0) raises RuntimeError for device='cuda', which
    is correct behaviour and must not be reported as a modifier failure -- but
    it also must not silently reduce this file's coverage without saying so.
    """
    pot = agama.Potential(**BASE_SPHERICAL)
    try:
        pot.potential(np.zeros((4, 3)), device="cuda")
        return True
    except Exception:
        return False


def _call(pot, xyz, op, **kw):
    return getattr(pot, op)(xyz, **kw)


def _relerr(got, ref):
    """Scale-normalized error: max|got-ref| / max|ref| over the whole sample.

    Not pointwise-relative: Logarithmic's Phi crosses zero on the sampled shell,
    where a pointwise ratio reports a huge error for a perfectly good result.
    """
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    scale = float(np.max(np.abs(ref))) if ref.size else 1.0
    return float(np.max(np.abs(got - ref))) / max(scale, 1e-300)


def parity_sweep(label, pot, xyz, devices, time=None):
    """One potential x 3 ops x devices x 2 dtypes against the legacy CPU path."""
    ok = True
    tkw = {} if time is None else {"t": time}
    for op in OPS:
        ref = _call(pot, xyz, op, **tkw)
        for device in devices:
            for dtype in DTYPES:
                tol = 5e-6 if dtype == np.float32 else 1e-12
                try:
                    got = _call(pot, xyz, op, device=device, dtype=dtype, **tkw)
                except Exception as e:
                    print(f"  FAIL {label:26s} {op:9s} {device:6s} "
                          f"{dtype.__name__:8s} : {type(e).__name__}: {e}")
                    ok = False
                    continue
                if got.shape != np.asarray(ref).shape:
                    print(f"  FAIL {label:26s} {op:9s} {device:6s} {dtype.__name__:8s} "
                          f": shape {got.shape} != {np.asarray(ref).shape}")
                    ok = False
                    continue
                err = _relerr(got, ref)
                good = err <= tol
                ok = ok and good
                print(f"  {'OK  ' if good else 'FAIL'} {label:26s} {op:9s} {device:6s} "
                      f"{dtype.__name__:8s} : rel={err:.3e} tol={tol:.1e}")
    return ok


def exact_serial_equals_cpu(label, pot, xyz, time=None):
    """serial / cpu / openmp differ only in loop scheduling, so results must be
    bit-for-bit identical -- the per-element math is the same and the reduction
    is element-wise, with no summation-order freedom."""
    ok = True
    tkw = {} if time is None else {"t": time}
    for op in OPS:
        for dtype in DTYPES:
            a = _call(pot, xyz, op, device="cpu", dtype=dtype, **tkw)
            b = _call(pot, xyz, op, device="openmp", dtype=dtype, **tkw)
            c = _call(pot, xyz, op, device="serial", dtype=dtype, **tkw)
            good = np.array_equal(a, b) and np.array_equal(a, c)
            ok = ok and good
            print(f"  {'OK  ' if good else 'FAIL'} {label:26s} {op:9s} "
                  f"{dtype.__name__:8s} : serial==cpu==openmp bit-for-bit")
    return ok


def constant_modifier_tests(xyz, devices):
    """Every modifier alone, plus the fully nested chain, plus a modified member
    inside a composite."""
    cases = [
        ("Shifted(Log)",   dict(BASE_TRIAXIAL, center=MOD_CENTER)),
        ("Tilted(Log)",    dict(BASE_TRIAXIAL, orientation=MOD_ORIENT)),
        ("Rotating(Log)",  dict(BASE_TRIAXIAL, rotation=MOD_ROTATE)),
        ("Scaled(Log)",    dict(BASE_TRIAXIAL, scale=MOD_SCALE)),
        # all four at once -> Shifted(Tilted(Rotating(Scaled(Log)))). This is the
        # case that pins the composition order: translations commute with
        # everything on their own, so order errors only surface when a shift and
        # a rotation/scale are both present.
        ("all4(Log)",      dict(BASE_TRIAXIAL, center=MOD_CENTER, orientation=MOD_ORIENT,
                                rotation=MOD_ROTATE, scale=MOD_SCALE)),
        ("all4(Plummer)",  dict(BASE_SPHERICAL, center=MOD_CENTER, orientation=MOD_ORIENT,
                                rotation=MOD_ROTATE, scale=MOD_SCALE)),
    ]
    ok = True
    for label, params in cases:
        pot = agama.Potential(**params)
        if not parity_sweep(label, pot, xyz, devices):
            ok = False
    print()
    for label, params in cases[:2]:
        pot = agama.Potential(**params)
        if not exact_serial_equals_cpu(label, pot, xyz):
            ok = False

    # A modified member alongside an unmodified one: the unmodified member keeps
    # its fast per-class kernel while the modified one goes through the
    # descriptor kernel, and both accumulate into the same output buffer.
    print()
    comp = agama.Potential(
        dict(type='Plummer', mass=0.5, scaleRadius=0.4),
        dict(type='NFW', mass=8.0, scaleRadius=4.0, center=MOD_CENTER),
    )
    print(f"  (composite is: {comp})")
    if not parity_sweep("Composite[Plum,Shift-NFW]", comp, xyz, devices):
        ok = False
    return ok


def time_dependent_tests(xyz, devices):
    """The part a t=0-only test cannot see: `t` must reach the kernel."""
    ok = True
    # center moves, amplitude grows, rotation angle ramps -- each as a 2-row
    # time series [t, values...], which readTimeDependentArray turns into a
    # genuine (non-constant) cubic spline.
    cases = [
        ("moving Shifted", dict(BASE_TRIAXIAL,
                               center=[[0.0, 0.0, 0.0, 0.0], [10.0, 2.0, -1.0, 0.5]])),
        ("growing Scaled", dict(BASE_TRIAXIAL,
                               scale=[[0.0, 1.0, 1.0], [10.0, 2.0, 1.4]])),
        ("spinning Rotating", dict(BASE_TRIAXIAL,
                                  rotation=[[0.0, 0.0], [10.0, 1.9]])),
        ("moving+growing", dict(BASE_TRIAXIAL,
                                center=[[0.0, 0.0, 0.0, 0.0], [10.0, 2.0, -1.0, 0.5]],
                                scale=[[0.0, 1.0, 1.0], [10.0, 2.0, 1.4]])),
    ]
    for label, params in cases:
        pot = agama.Potential(**params)
        # (a) the device answer must track the legacy answer at each t
        for t in (0.0, 2.5, 6.25, 10.0):
            if not parity_sweep(f"{label} t={t}", pot, xyz, devices, time=t):
                ok = False
        # (b) and the answer must actually CHANGE with t, or (a) proves nothing
        for device in devices:
            a = pot.potential(xyz, t=0.0, device=device)
            b = pot.potential(xyz, t=10.0, device=device)
            spread = float(np.max(np.abs(np.asarray(b) - np.asarray(a))))
            good = spread > 1e-6
            ok = ok and good
            print(f"  {'OK  ' if good else 'FAIL'} {label:26s} {device:6s} "
                  f": Phi(t=10) differs from Phi(t=0) by {spread:.3e} (must be > 1e-6)")
        print()

    # UniformAcceleration: time-dependent by nature, newly registered on the
    # batch path. Built from a file of [time ax ay az] rows.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("0\t0\t0\t0\n10\t0.3\t-0.2\t0.1\n")
        accel_file = f.name
    try:
        ua = agama.Potential(type='UniformAcceleration', file=accel_file)
        print(f"  (built: {ua})")
        for t in (0.0, 5.0, 10.0):
            if not parity_sweep(f"UniformAccel t={t}", ua, xyz, devices, time=t):
                ok = False
        for device in devices:
            a = ua.potential(xyz, t=0.0, device=device)
            b = ua.potential(xyz, t=10.0, device=device)
            spread = float(np.max(np.abs(np.asarray(b) - np.asarray(a))))
            good = spread > 1e-6
            ok = ok and good
            print(f"  {'OK  ' if good else 'FAIL'} {'UniformAccel':26s} {device:6s} "
                  f": Phi(t=10) differs from Phi(t=0) by {spread:.3e}")
    except Exception as e:
        print(f"  FAIL UniformAcceleration: {type(e).__name__}: {e}")
        ok = False
    finally:
        os.unlink(accel_file)
    return ok


def kwarg_hygiene_tests(xyz):
    """The device path must reject what the legacy path rejects.

    Regression guard for a real bug: the device entry point used to read only
    `device` and `dtype` straight out of the kwargs dict and never check for
    leftovers, so a dropped `t` or a misspelled `dtype` passed silently.
    """
    pot = agama.Potential(**BASE_SPHERICAL)
    ok = True

    # (a) a misspelled kwarg must raise, on the device path as on the legacy one
    for kw, where in ((dict(dtpye=np.float32, device="cpu"), "device path"),
                      (dict(dtpye=np.float32), "legacy path")):
        try:
            pot.potential(xyz, **kw)
            print(f"  FAIL misspelled kwarg on the {where} was accepted silently")
            ok = False
        except TypeError as e:
            print(f"  OK   misspelled kwarg on the {where} raises TypeError: {e}")
        except Exception as e:
            print(f"  FAIL misspelled kwarg on the {where} raised "
                  f"{type(e).__name__} (expected TypeError): {e}")
            ok = False

    # (b) a per-point array of times cannot be honoured by a path that shares one
    # time across the batch, so it must be refused rather than quietly truncated
    try:
        pot.potential(xyz, t=np.linspace(0.0, 1.0, len(xyz)), device="cpu")
        print("  FAIL per-point t array on the device path was accepted "
              "(it can only have used one of the times)")
        ok = False
    except NotImplementedError as e:
        print(f"  OK   per-point t array on the device path raises "
              f"NotImplementedError: {str(e)[:90]}")
    except Exception as e:
        print(f"  FAIL per-point t array raised {type(e).__name__} "
              f"(expected NotImplementedError): {e}")
        ok = False

    # (c) a scalar t is fine and, for a time-independent potential, changes nothing
    try:
        a = pot.potential(xyz, device="cpu")
        b = pot.potential(xyz, t=3.0, device="cpu")
        good = np.array_equal(a, b)
        ok = ok and good
        print(f"  {'OK  ' if good else 'FAIL'} scalar t is accepted and is a no-op "
              f"for a time-independent potential")
    except Exception as e:
        print(f"  FAIL scalar t on the device path: {type(e).__name__}: {e}")
        ok = False
    return ok


def orbit_tests(cuda):
    """The orbit kernel builds one descriptor and reuses it for every step, so a
    constant modifier is fine and a time-varying one must be refused."""
    ok = True
    devices = ["cpu", "serial"] + (["cuda"] if cuda else [])
    rng = np.random.default_rng(11)
    nic = 24
    ic = np.column_stack([
        rng.uniform(1.0, 3.0, nic), rng.uniform(-1.0, 1.0, nic),
        rng.uniform(-0.5, 0.5, nic),
        rng.uniform(-0.3, 0.3, nic), rng.uniform(0.4, 0.8, nic),
        rng.uniform(-0.2, 0.2, nic),
    ])

    # (a) a CONSTANT modifier chain: the descriptor is valid at every time, so
    # the device orbit must reproduce the legacy CPU orbit
    pot = agama.Potential(dict(BASE_SPHERICAL, center=MOD_CENTER,
                               orientation=MOD_ORIENT, scale=MOD_SCALE))
    print(f"  (constant-modifier potential: {pot})")
    ref = agama.orbit(potential=pot, ic=ic, time=20.0, trajsize=17,
                      dtype=np.float64)
    for device in devices:
        try:
            got = agama.orbit(potential=pot, ic=ic, time=20.0, trajsize=17,
                              dtype=np.float64, device=device)
        except Exception as e:
            print(f"  FAIL orbit, constant modifiers, device={device:6s}: "
                  f"{type(e).__name__}: {e}")
            ok = False
            continue
        worst = 0.0
        for i in range(nic):
            r_ref, r_got = np.asarray(ref[i][1]), np.asarray(got[i][1])
            scale = max(float(np.max(np.abs(r_ref))), 1e-300)
            worst = max(worst, float(np.max(np.abs(r_got - r_ref))) / scale)
        tol = 1e-10
        good = worst <= tol
        ok = ok and good
        print(f"  {'OK  ' if good else 'FAIL'} orbit, constant modifiers, "
              f"device={device:6s}: max rel err vs legacy = {worst:.3e} tol={tol:.1e}")

    # (b) a TIME-VARYING modifier must be refused with a message that says so.
    # If this ever starts passing, device-resident modifier splines have landed
    # and this check should become a parity check instead of a refusal check.
    moving = agama.Potential(dict(BASE_SPHERICAL,
                                  center=[[0.0, 0.0, 0.0, 0.0], [20.0, 3.0, -1.0, 0.0]]))
    for device in devices:
        try:
            agama.orbit(potential=moving, ic=ic, time=20.0, trajsize=17,
                        dtype=np.float64, device=device)
            print(f"  FAIL orbit, time-varying modifier, device={device:6s}: "
                  f"accepted (would have integrated a frozen snapshot)")
            ok = False
        except NotImplementedError as e:
            msg = str(e)
            good = "time" in msg.lower()
            ok = ok and good
            print(f"  {'OK  ' if good else 'FAIL'} orbit, time-varying modifier, "
                  f"device={device:6s}: NotImplementedError: {msg[:110]}")
        except Exception as e:
            print(f"  FAIL orbit, time-varying modifier, device={device:6s}: "
                  f"{type(e).__name__} (expected NotImplementedError): {e}")
            ok = False

    # (c) the same time-varying potential still integrates on the legacy path,
    # so the refusal above is a device-path limitation, not a broken potential
    try:
        agama.orbit(potential=moving, ic=ic, time=20.0, trajsize=17, dtype=np.float64)
        print("  OK   orbit, time-varying modifier, legacy path: integrates fine")
    except Exception as e:
        print(f"  FAIL orbit, time-varying modifier, legacy path: "
              f"{type(e).__name__}: {e}")
        ok = False
    return ok


def main():
    rng = np.random.default_rng(4242)
    xyz = rng.uniform(-3.0, 3.0, size=(2048, 3))

    cuda = cuda_available()
    devices = tuple(d for d in DEVICES if d != "cuda" or cuda)
    print(f"CUDA backend: {'available' if cuda else 'NOT built -- cuda cases skipped'}")
    print(f"devices exercised: {', '.join(devices)}\n")

    all_ok = True

    print("== Constant modifier chains: parity vs the legacy CPU path ==")
    if not constant_modifier_tests(xyz, devices):
        all_ok = False

    print("\n== Time-dependent modifiers: `t` must reach the kernel ==")
    if not time_dependent_tests(xyz, devices):
        all_ok = False

    print("\n== Keyword-argument hygiene on the device path ==")
    if not kwarg_hygiene_tests(xyz):
        all_ok = False

    print("\n== Orbit integration with modifiers ==")
    if not orbit_tests(cuda):
        all_ok = False

    print("\n" + ("PASS" if all_ok else "FAIL")
          + ("" if cuda else "  (CPU-only build: device='cuda' cases were skipped)"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
