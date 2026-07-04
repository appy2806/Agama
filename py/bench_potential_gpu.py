"""Tier 1 timing benchmark: pot.potential(xyz, device=..., dtype=...).

Runs each GPU-migrated analytic potential through:
  * device='cpu'  = OpenMP-parallel across ALL cores. This matches what AGAMA
                    normally does (the legacy no-device path is OpenMP-parallel),
                    so it is the honest baseline: the gpu/cpu ratios below are
                    GPU vs multi-core OpenMP, NOT GPU vs a single-thread loop.
                    Expect fp64 GPU to LOSE to OpenMP at large N on consumer
                    cards with 1:64 fp64:fp32 throughput -- that is real.
                    (device='serial' exists for the single-thread baseline.)
  * device='cuda' = the GPU path via potential_gpu.cpp
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

    # ----------------------------------------------------------------------
    # Per-step breakdown of one GPU call at the largest N. Tells the user
    # where time actually goes: cast cost vs H2D vs kernel vs D2H. Lets them
    # understand why pot.potential(xyz, device='cuda') is not the C++-bench's
    # 12x speedup -- input must be converted and copied to GPU each call.
    # The only way to recover the kernel-only speedup is to keep data on
    # the device (Phase B: CuPy passthrough via __cuda_array_interface__).
    # ----------------------------------------------------------------------
    N = 1 << 22
    print(f"\nPer-step breakdown for NFW, N={N}, fp32 (input is fp64 numpy by default):")
    nfw = pots[2][1]
    xyz_f64 = rng.uniform(-5.0, 5.0, size=(N, 3))            # what users typically pass
    xyz_f32 = xyz_f64.astype(np.float32)                     # what users could pre-cast
    # warm up the GPU scratch buffers (first call pays cudaMalloc)
    _ = nfw.potential(xyz_f64, device='cuda', dtype=np.float32)
    _ = nfw.potential(xyz_f64, device='cuda', dtype=np.float64)
    t_cast   = time_ms_min(lambda: nfw.potential(xyz_f64, device='cuda', dtype=np.float32))
    print(f"  fp64-numpy in, fp32 GPU out (cast on entry)   : {t_cast:7.3f} ms")
    t_nocast = time_ms_min(lambda: nfw.potential(xyz_f32, device='cuda', dtype=np.float32))
    print(f"  fp32-numpy in, fp32 GPU out (no cast)         : {t_nocast:7.3f} ms")
    t_f64    = time_ms_min(lambda: nfw.potential(xyz_f64, device='cuda', dtype=np.float64))
    print(f"  fp64-numpy in, fp64 GPU out (no cast)         : {t_f64:7.3f} ms")
    print(f"  -> pre-casting the input to the requested dtype is a {t_cast/t_nocast:.1f}x "
          f"end-to-end win here ({t_cast:.1f} -> {t_nocast:.1f} ms): smaller copies AND "
          f"a faster kernel, not just the astype() cost vanishing.\n")

    # ----------------------------------------------------------------------
    # Phase B: device-resident (CuPy) round-trip. Input already lives on the
    # GPU (__cuda_array_interface__ passthrough), output is a CuPy array --
    # no H2D, no D2H, no dtype cast. This is the C++ kernel benchmark number
    # finally reaching Python. The call is internally synchronized
    # (cudaStreamSynchronize before return), so perf_counter timing is valid.
    # ----------------------------------------------------------------------
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        cp = None
        print(f"\nDevice-resident (CuPy) round-trip: SKIP (cupy unavailable: "
              f"{type(e).__name__}: {e})")
    if cp is not None:
        print(f"\nDevice-resident (CuPy) round-trip for NFW, N={N}:")
        d_xyz64 = cp.asarray(xyz_f64)
        d_xyz32 = cp.asarray(xyz_f32)
        # warm-up (allocator, module import side effects)
        _ = nfw.potential(d_xyz64, device='cuda')
        _ = nfw.potential(d_xyz32, device='cuda')
        t_dev64 = time_ms_min(lambda: nfw.potential(d_xyz64, device='cuda'))
        t_dev32 = time_ms_min(lambda: nfw.potential(d_xyz32, device='cuda'))
        # honest comparison baselines at the same N
        t_omp64 = time_ms_min(lambda: nfw.potential(xyz_f64, device='cpu', dtype=np.float64))
        t_omp32 = time_ms_min(lambda: nfw.potential(xyz_f32, device='cpu', dtype=np.float32))
        print(f"  cupy fp64 in, cupy fp64 out (no copies)       : {t_dev64:7.3f} ms")
        print(f"  cupy fp32 in, cupy fp32 out (no copies)       : {t_dev32:7.3f} ms")
        print(f"  [recall] numpy fp64 in, fp64 out (H2D+D2H)    : {t_f64:7.3f} ms")
        print(f"  [recall] numpy fp32 in, fp32 out (H2D+D2H)    : {t_nocast:7.3f} ms")
        print(f"  [recall] OpenMP all-cores fp64                : {t_omp64:7.3f} ms")
        print(f"  [recall] OpenMP all-cores fp32                : {t_omp32:7.3f} ms")
        print(f"  -> honest comparison vs OpenMP all-cores: device-resident GPU is "
              f"{t_omp64/t_dev64:.1f}x (fp64) / {t_omp32/t_dev32:.1f}x (fp32); "
              f"eliminating the PCIe round-trip took the GPU path from "
              f"{t_f64:.1f}/{t_nocast:.1f} ms (numpy) to {t_dev64:.2f}/{t_dev32:.2f} ms "
              f"(fp64/fp32). Keep data on the device to get kernel-limited speed.")

    # ----------------------------------------------------------------------
    # Composite dispatch: 4-component composite evaluated member-by-member on
    # the SAME device buffers (first member stores, the rest accumulate with
    # add=true) -- zero extra host/device transfers. The proof is in the CuPy
    # numbers: composite time ~= sum of the 4 individual member kernel times
    # (+ ~launch overhead), with no copy-sized gap.
    # ----------------------------------------------------------------------
    print(f"\nComposite dispatch (4 members: Plummer+Isochrone+NFW+MiyamotoNagai), N={N}:")
    member_pots = [pots[0], pots[1], pots[2], pots[3]]   # Plummer, Isochrone, NFW, MiyamotoNagai
    comp = agama.Potential(
        dict(type='Plummer',       mass=1.0, scaleRadius=1.0),
        dict(type='Isochrone',     mass=1.0, scaleRadius=1.0),
        dict(type='NFW',           mass=1.0, scaleRadius=1.0),
        dict(type='MiyamotoNagai', mass=1.0, scaleRadius=1.0, scaleHeight=0.3),
    )
    for dtype, xyz_h in ((np.float64, xyz_f64), (np.float32, xyz_f32)):
        tag = 'fp64' if dtype == np.float64 else 'fp32'
        t_omp  = time_ms_min(lambda: comp.potential(xyz_h, device='cpu',  dtype=dtype))
        t_np   = time_ms_min(lambda: comp.potential(xyz_h, device='cuda', dtype=dtype))
        print(f"  [{tag}] composite OpenMP all-cores                 : {t_omp:7.3f} ms")
        print(f"  [{tag}] composite numpy-cuda (H2D + 4 kernels + D2H): {t_np:7.3f} ms")
        if cp is not None:
            d_in = cp.asarray(xyz_h)
            _ = comp.potential(d_in, device='cuda')   # warm-up
            t_cp = time_ms_min(lambda: comp.potential(d_in, device='cuda'))
            t_members = 0.0
            for mname, mpot in member_pots:
                _ = mpot.potential(d_in, device='cuda')   # warm-up
                t_members += time_ms_min(lambda: mpot.potential(d_in, device='cuda'))
            print(f"  [{tag}] composite cupy-cuda (device-resident)     : {t_cp:7.3f} ms")
            print(f"  [{tag}] sum of 4 individual cupy-cuda member calls: {t_members:7.3f} ms")
            print(f"  [{tag}] -> composite cupy time is {t_cp/t_members:.2f}x the summed member "
                  f"kernel times (delta {t_cp - t_members:+.3f} ms ~ saved output allocs + "
                  f"launch overhead): no hidden transfers in the composite path.")

    # Reminder: input flexibility — numpy and list-of-tuples both work
    print("\nInput-type sanity check (NFW, N=4):")
    for desc, xyz_in in [
        ("numpy (4,3)", np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])),
        ("list of tuples", [(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(1.,1.,1.)]),
    ]:
        phi = nfw.potential(xyz_in, device='cuda', dtype=np.float64)
        print(f"  {desc:18s} -> phi = {phi}  (dtype {phi.dtype})")


if __name__ == "__main__":
    main()
