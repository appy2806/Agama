"""Tier 3 timing benchmark: agama_migrate.orbit(device=..., dtype=...).

Same composite potential as crosscheck_orbits.py so numbers are comparable
to the recorded 16-core legacy baseline (~31k orbits/s):

  Composite: Plummer (M=0.1, rs=0.3) + MiyamotoNagai (M=0.5, rs=1.0, rz=0.3)
             + NFW (M=10, rs=5)

Rows reported:
  - legacy (no device kwarg, 16-core OpenMP loop) fp64
  - device='cpu'    (batch OpenMP)          fp64
  - device='cuda'   fp64
  - device='cuda'   fp32 (dtype=np.float32) -- fp32 integration

T=100, trajsize=64.  N in {5000, 50000}.

Timing: best of 3 runs; first run discarded as warm-up for cuda (so that
CUDA context init, first-touch cudaMalloc costs are not counted).  H2D/D2H
copies ARE included because they are part of the user-visible cost.
"""
import time
import numpy as np
import agama_migrate as agama


def time_s_min(fn, trials=3):
    """Best of `trials` runs of fn(), in seconds. Discards one warm-up call."""
    fn()  # warm-up (evicts first-call malloc/JIT overhead)
    best = float("inf")
    for _ in range(trials):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if dt < best:
            best = dt
    return best


def make_ics(pot, n, seed=7):
    """Near-circular ICs at r=0.5..3."""
    rng = np.random.default_rng(seed)
    r   = rng.uniform(0.5, 3.0, n)
    phi = rng.uniform(0, 2*np.pi, n)
    z   = rng.uniform(-0.3, 0.3, n)
    pos = np.column_stack([r*np.cos(phi), r*np.sin(phi), z])
    vcirc = np.sqrt(np.maximum(-np.sum(pot.force(pos)*pos, axis=1), 0.0))
    vel = np.column_stack([-vcirc*np.sin(phi), vcirc*np.cos(phi),
                           rng.normal(0, 0.05, n)])
    vel += rng.normal(0, 0.03, (n, 3)) * vcirc[:, None]
    return np.column_stack([pos, vel])


def probe_cuda(pot):
    """Return True if device='cuda' is available in this build."""
    ic_probe = np.array([[1.0, 0.0, 0.0, 0.0, 0.3, 0.0]])
    try:
        agama.orbit(potential=pot, ic=ic_probe, time=1.0, trajsize=2,
                    device='cuda', dtype=np.float64, verbose=False)
        return True
    except RuntimeError as e:
        if 'built without CUDA support' in str(e):
            return False
        raise


def main():
    pot = agama.Potential(
        dict(type='Plummer',       mass=0.1, scaleRadius=0.3),
        dict(type='MiyamotoNagai', mass=0.5, scaleRadius=1.0, scaleHeight=0.3),
        dict(type='NFW',           mass=10.0, scaleRadius=5.0),
    )

    T        = 100.0
    TRAJSIZE = 64
    Ns       = [5000, 50000]

    cuda_available = probe_cuda(pot)
    if not cuda_available:
        print("NOTE: CUDA not available in this build; cuda rows will show n/a.")

    # Table header.
    hdr = (f"{'N':>7s}  {'backend':22s}  {'dtype':6s}  "
           f"{'time(s)':>9s}  {'orbits/s':>10s}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for N in Ns:
        ic = make_ics(pot, N, seed=7)

        rows = []

        # Legacy (no device kwarg): 16-core OpenMP, fp64 -- the baseline to beat.
        t = time_s_min(lambda: agama.orbit(potential=pot, ic=ic, time=T,
                                           trajsize=TRAJSIZE, dtype=np.float64,
                                           verbose=False))
        rows.append(("legacy (no device)", "float64", t))

        # Batch CPU / OpenMP path.
        t = time_s_min(lambda: agama.orbit(potential=pot, ic=ic, time=T,
                                           trajsize=TRAJSIZE, device='cpu',
                                           dtype=np.float64, verbose=False))
        rows.append(("device='cpu' batch", "float64", t))

        # CUDA fp64.
        if cuda_available:
            t = time_s_min(lambda: agama.orbit(potential=pot, ic=ic, time=T,
                                               trajsize=TRAJSIZE, device='cuda',
                                               dtype=np.float64, verbose=False))
            rows.append(("device='cuda'", "float64", t))
        else:
            rows.append(("device='cuda'", "float64", None))

        # CUDA fp32 integration.
        if cuda_available:
            t = time_s_min(lambda: agama.orbit(potential=pot, ic=ic, time=T,
                                               trajsize=TRAJSIZE, device='cuda',
                                               dtype=np.float32, verbose=False))
            rows.append(("device='cuda'", "float32", t))
        else:
            rows.append(("device='cuda'", "float32", None))

        for backend, dtype_str, elapsed in rows:
            if elapsed is None:
                print(f"{N:>7d}  {backend:22s}  {dtype_str:6s}  "
                      f"{'n/a':>9s}  {'n/a':>10s}")
            else:
                orbs_per_s = N / elapsed
                print(f"{N:>7d}  {backend:22s}  {dtype_str:6s}  "
                      f"{elapsed:>9.3f}  {orbs_per_s:>10.0f}")
        print(sep)


if __name__ == "__main__":
    main()
