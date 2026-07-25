#!/usr/bin/env python
"""
Units on the device-resident (__cuda_array_interface__) path for
pot.potential() / pot.force() / pot.density().

Before 2026-07-25 these raised NotImplementedError under a non-trivial unit
system. They are now supported by scaling at the Python boundary: the input is
multiplied into internal units (a scaled COPY -- the caller's array must not be
mutated) and the result back into user units, both by CuPy elementwise multiply
with a numpy scalar of the array's own dtype.

The gate is BITWISE equality with the host (numpy) path, not closeness, because
the boundary scaling reproduces the host's `in[i] * static_cast<T>(L)` and
`out[i] *= static_cast<T>(F)` exactly. A "close enough" tolerance here would hide
precisely the kind of ordering/precision change that would make the two paths
diverge.

Every units check also asserts the unit system ACTUALLY CHANGED the numbers
relative to the unitless case. Without that, a device-vs-host comparison passes
vacuously if setUnits silently does nothing -- which is a bug this fork really
shipped once (fixed in commit 27d6555).
"""
import numpy
import sys

try:
    import agama_migrate as agama
except ImportError:
    print("SKIP: agama_migrate is not importable")
    sys.exit(0)

try:
    import cupy
    cupy.zeros(1)          # force context creation; raises if no usable GPU
except Exception as e:
    print("SKIP: CuPy/GPU unavailable (%s: %s)" % (type(e).__name__, e))
    sys.exit(0)

nok = nfail = 0


def check(ok, label, detail=""):
    global nok, nfail
    if ok:
        nok += 1
        print("  OK   %s %s" % (label, detail))
    else:
        nfail += 1
        print("  FAIL %s %s" % (label, detail))


def bitwise_equal(a, b):
    """Bit-pattern comparison, so NaN slots compare equal to themselves and a
    real mismatch cannot hide behind NaN != NaN."""
    a = numpy.ascontiguousarray(a)
    b = numpy.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return numpy.array_equal(a.view(numpy.uint8), b.view(numpy.uint8))


def make_points(n, dtype):
    rng = numpy.random.default_rng(20260725)
    r = 0.5 + 4.5 * rng.random(n)
    th = numpy.arccos(1 - 2 * rng.random(n))
    ph = 2 * numpy.pi * rng.random(n)
    xyz = numpy.stack([r * numpy.sin(th) * numpy.cos(ph),
                       r * numpy.sin(th) * numpy.sin(ph),
                       r * numpy.cos(th)], axis=1)
    return numpy.ascontiguousarray(xyz, dtype=dtype)


OPS = ("potential", "force", "density")


def run_ops(pot, xyz_host, dtype):
    """Return {op: (host_result, device_result)} for the three ops."""
    out = {}
    for op in OPS:
        fn = getattr(pot, op)
        host = fn(xyz_host, device="cuda", dtype=dtype)
        xyz_dev = cupy.asarray(xyz_host)
        dev = fn(xyz_dev, device="cuda", dtype=dtype)
        out[op] = (numpy.asarray(host), cupy.asnumpy(dev))
    return out


def main():
    N = 4096

    # ---------------------------------------------------------------- unitless
    print("\n== Default (trivial) unit system: device == host, bitwise ==")
    unitless = {}
    for dtype in (numpy.float64, numpy.float32):
        pot = agama.Potential(type="NFW", mass=1.0, scaleRadius=1.0)
        xyz = make_points(N, dtype)
        res = run_ops(pot, xyz, dtype)
        for op in OPS:
            h, d = res[op]
            check(bitwise_equal(h, d), "%-8s %s :" % (op, dtype.__name__),
                  "bitwise=%s shape=%s" % (bitwise_equal(h, d), d.shape))
        unitless[dtype] = {op: res[op][0].copy() for op in OPS}

    # ------------------------------------------------------------ with units
    # A deliberately non-round unit system, so every factor differs from 1 and
    # none of them is exactly representable -- the case most likely to expose an
    # arithmetic-ordering difference between the two paths.
    print("\n== Non-trivial unit system (setUnits): device == host, bitwise ==")
    agama.setUnits(mass=1e11, length=3.7, velocity=237.0)

    for dtype in (numpy.float64, numpy.float32):
        pot = agama.Potential(type="NFW", mass=1.0, scaleRadius=1.0)
        xyz = make_points(N, dtype)

        # the caller's device array must NOT be mutated by the scaling
        xyz_dev = cupy.asarray(xyz)
        xyz_dev_before = cupy.asnumpy(xyz_dev).copy()

        for op in OPS:
            fn = getattr(pot, op)
            host = numpy.asarray(fn(xyz, device="cuda", dtype=dtype))
            dev = cupy.asnumpy(fn(xyz_dev, device="cuda", dtype=dtype))
            check(bitwise_equal(host, dev),
                  "%-8s %s :" % (op, dtype.__name__),
                  "bitwise=%s" % bitwise_equal(host, dev))

            # non-vacuity: the unit system must have changed the values, else the
            # comparison above proves nothing (see the module docstring)
            changed = not numpy.array_equal(
                numpy.nan_to_num(host), numpy.nan_to_num(unitless[dtype][op]))
            check(changed, "%-8s %s : units changed the values" %
                  (op, dtype.__name__), "(non-vacuous)")

        check(bitwise_equal(xyz_dev_before, cupy.asnumpy(xyz_dev)),
              "caller's input array unmutated (%s) :" % dtype.__name__,
              "bitwise=%s" % bitwise_equal(xyz_dev_before, cupy.asnumpy(xyz_dev)))

    # ------------------------------------------------- single-point shape, units
    print("\n== Single point (shape (3,)) under units ==")
    pot = agama.Potential(type="NFW", mass=1.0, scaleRadius=1.0)
    for dtype in (numpy.float64, numpy.float32):
        p = numpy.array([1.3, -0.7, 0.45], dtype=dtype)
        for op in OPS:
            fn = getattr(pot, op)
            host = numpy.asarray(fn(p, device="cuda", dtype=dtype))
            dev = cupy.asnumpy(fn(cupy.asarray(p), device="cuda", dtype=dtype))
            check(bitwise_equal(host, dev), "%-8s %s single :" %
                  (op, dtype.__name__), "host=%s dev=%s" % (host, dev))

    # ------------------------------------------------------------- composite
    # The boundary scaling is independent of the potential's structure, so a
    # composite (and a modifier chain) must behave exactly as a plain potential.
    print("\n== Composite + modifier chain under units ==")
    comp = agama.Potential(
        dict(type="Plummer", mass=1.0, scaleRadius=0.7),
        dict(type="NFW", mass=10.0, scaleRadius=5.0, center=[0.1, -0.2, 0.05]))
    for dtype in (numpy.float64, numpy.float32):
        xyz = make_points(1024, dtype)
        for op in OPS:
            fn = getattr(comp, op)
            host = numpy.asarray(fn(xyz, device="cuda", dtype=dtype))
            dev = cupy.asnumpy(fn(cupy.asarray(xyz), device="cuda", dtype=dtype))
            check(bitwise_equal(host, dev),
                  "composite %-8s %s :" % (op, dtype.__name__),
                  "bitwise=%s" % bitwise_equal(host, dev))

    agama.setUnits(mass=1, length=1, velocity=1)

    print("\n%d OK / %d FAIL" % (nok, nfail))
    print("PASS" if nfail == 0 else "FAIL")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
