"""Tier 1 timing benchmark: pot.potential(xyz, device=..., dtype=...).

Runs each GPU-migrated analytic potential through:
  * device='cpu'  (the legacy OpenMP-parallelized path that AGAMA has always used,
                   reached when no device kwarg is given; we test it via device='cpu'
                   for symmetry with the GPU path even though the kernel is identical)
  * device='cuda' (the new GPU path via potential_gpu.cpp)
each at both fp64 (np.float64) and fp32 (np.float32), at a sweep of N values.

The reported "best of K trials" excludes a warm-up call so per-process startup cost
(CUDA context init, JIT, first-touch allocation) is not counted. H2D and D2H copies
are included in the GPU timing because they are part of the user-observed cost.
"""
import time
import numpy as np
import agama_migrate as agama


def time_ms_min(fn, trials=5):
    """min over `trials` runs of `fn`, in milliseconds. Discards one warm-up call."""
    fn()
    best = float("inf")
    for _ in range(trials):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if dt < best:
            best = dt
    return best * 1e3


def bench_one(name, pot, xyz, dtype):
    """Time fp{32,64} potential eval on cpu vs cuda. Returns (t_cpu_ms, t_gpu_ms)."""
    t_cpu = time_ms_min(lambda: pot.potential(xyz, device="cpu",  dtype=dtype))
    t_gpu = time_ms_min(lambda: pot.potential(xyz, device="cuda", dtype=dtype))
    return t_cpu, t_gpu


def main():
    rng = np.random.default_rng(42)
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
    Ns = [1 << 14, 1 << 18, 1 << 20, 1 << 22]   # 16K, 262K, 1M, 4M

    header = (f"{'Potential':14s} {'N':>8s}   "
              f"{'fp64 cpu':>10s} {'fp64 gpu':>10s} {'gpu/cpu':>8s}   "
              f"{'fp32 cpu':>10s} {'fp32 gpu':>10s} {'gpu/cpu':>8s}")
    print(header)
    print("-" * len(header))

    for N in Ns:
        xyz = rng.uniform(-5.0, 5.0, size=(N, 3))
        for name, pot in pots:
            t_cpu_64, t_gpu_64 = bench_one(name, pot, xyz, np.float64)
            t_cpu_32, t_gpu_32 = bench_one(name, pot, xyz, np.float32)
            print(f"{name:14s} {N:>8d}   "
                  f"{t_cpu_64:>8.3f}ms {t_gpu_64:>8.3f}ms {t_cpu_64/t_gpu_64:>6.1f}x   "
                  f"{t_cpu_32:>8.3f}ms {t_gpu_32:>8.3f}ms {t_cpu_32/t_gpu_32:>6.1f}x")
        print()

    # Reminder: input flexibility — numpy and list-of-tuples both work
    print("Input-type sanity check (NFW, N=4):")
    nfw = pots[2][1]
    for desc, xyz_in in [
        ("numpy (4,3)", np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])),
        ("list of tuples", [(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(1.,1.,1.)]),
    ]:
        phi = nfw.potential(xyz_in, device='cuda', dtype=np.float64)
        print(f"  {desc:18s} -> phi = {phi}  (dtype {phi.dtype})")


if __name__ == "__main__":
    main()
