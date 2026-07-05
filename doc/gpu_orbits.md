# GPU orbit integration from Python — usage patterns

This document explains how to use `agama.orbit(..., device=...)` (Tier 3 of the
CPU/GPU unification) efficiently, and — just as important — which invocation
patterns do *not* help. The physics and outputs are identical across all
patterns; only throughput differs.

**Status:** the batch backends (`device='cpu'/'openmp'/'serial'/'cuda'`) are
implemented and tested. Per-call CUDA streams ("Path A") landed 2026-07-05,
making the threaded pattern in §3 fast — each `agama.orbit(...)` call from a
different Python thread runs on its own `cudaStream_t`, and kernels co-execute
on the GPU.

---

## 1. The basics

```python
import numpy as np
import agama            # (or agama_migrate during the migration)

pot = agama.Potential(
    dict(type='Plummer',       mass=0.1, scaleRadius=0.3),
    dict(type='MiyamotoNagai', mass=0.5, scaleRadius=1.0, scaleHeight=0.3),
    dict(type='NFW',           mass=10.0, scaleRadius=5.0))

# N x 6 initial conditions, one batch call, all orbits integrated on the GPU:
times, trajs = zip(*agama.orbit(potential=pot, ic=ic, time=100.0, trajsize=64,
                                device='cuda'))
```

- `device` absent → the untouched legacy CPU path (bit-for-bit identical to
  upstream AGAMA).
- `device='cuda'` with no `dtype` → **fp64 integration**, float32 trajectory
  storage — the same defaults as the legacy path (verified bit-for-bit at
  float32 precision).
- Precision opt-in: `dtype=np.float32` switches the *integration itself* to
  fp32. On consumer GPUs (fp64 throttled 1:64) this is the fast path. Pair it
  with an explicit `accuracy=1e-5`:

```python
agama.orbit(potential=pot, ic=ic, time=100.0, trajsize=64,
            device='cuda', dtype=np.float32, accuracy=1e-5)
```

  fp32 quality: energy conservation |dE/E| ~ 1e-5 over T=100 in a realistic
  composite — adequate for particle-spray / stream statistics. Requests for
  accuracy below ~10 ULP of the working precision are floored automatically.

- Per-orbit `time` and `timestart` arrays are supported — a particle-spray
  release directly maps onto one call:

```python
# 5000 particles released at t_start in [0, 20], all integrated to t_end = 100
agama.orbit(potential=pot, ic=ic, time=100.0 - t_start, timestart=t_start,
            trajsize=64, device='cuda', dtype=np.float32, accuracy=1e-5)
```

Unsupported combinations under `device=` raise `NotImplementedError`
(targets, `der`, `lyapunov`, `Omega != 0`, non-DOP853 `method`,
`dtype=object`, `trajsize=0`, per-orbit trajsize) — omit `device` to use the
legacy path for those. Potentials must be GPU-capable (the analytic types or a
Composite of them); others raise `NotImplementedError` naming the offending
component.

---

## 2. Rule #1: one big call beats many small calls

GPU throughput is set by how many orbit-threads are resident at once. A single
5K-orbit call under-fills the card; concatenating your work into one call is
always the simplest win and **works today**:

```python
ic_all      = np.concatenate([ic_1, ..., ic_10])        # 10 sprays -> 50K orbits
time_all    = np.concatenate([t_1,  ..., t_10])
tstart_all  = np.concatenate([ts_1, ..., ts_10])
res = agama.orbit(potential=pot, ic=ic_all, time=time_all, timestart=tstart_all,
                  trajsize=64, device='cuda', dtype=np.float32, accuracy=1e-5)
```

A serial `for` loop of small calls gains nothing from any concurrency
mechanism — there is never more than one kernel in flight:

```python
for s in range(10):                          # ANTI-PATTERN on any backend
    out.append(agama.orbit(..., device='cuda'))
```

---

## 3. Concurrent calls from ONE script: Python threads (Path A)

When the calls cannot be concatenated — e.g. an MCMC sampler whose likelihood
generates a spray per walker — use a **thread** pool. The C extension releases
the GIL for the duration of each orbit call, so plain Python threads give real
concurrency:

```python
from concurrent.futures import ThreadPoolExecutor

pot = agama.Potential(...)          # ONE shared potential object; read-only
                                    # during orbit calls, safe across threads

def run_spray(seed):
    ic, t_start = make_spray(seed)  # each job: its own ~5K-particle spray
    return agama.orbit(potential=pot, ic=ic,
                       time=100.0 - t_start, timestart=t_start,
                       trajsize=64, device='cuda',
                       dtype=np.float32, accuracy=1e-5)

with ThreadPoolExecutor(max_workers=3) as pool:
    # match to device capacity: ~2-3 concurrent 5K-orbit fp64 calls on a 48-SM card
    results = list(pool.map(run_spray, range(10)))   # up to 3 calls in flight
```

Each in-flight call runs on its own CUDA stream and the hardware scheduler
co-executes the kernels — up to 3 × 5K-orbit concurrent calls saturate the
card automatically on this RTX 3080 Laptop (48 SMs, register-heavy DOP853), at
roughly the throughput of the equivalent one-big-batch call.
**No code changes are needed in this snippet** — it was already the correct way
to write concurrent calls; Path A simply makes it fast.

**How many threads?** Match your pool size to device capacity. On this RTX 3080
Laptop, capacity is ~2–3 concurrent 5K-orbit fp64 calls before the warp queue
backs up; 4 C++ threads beat the batched 20K call (214 ms vs 310 ms), a good
practical knee. On larger cards (L40, A100 with more SMs or fast fp64) capacity
is higher — re-measure at deployment. **Important:** fp32 kernels are much
shorter (~16 ms for 5K at accuracy=1e-5) and benefit little from concurrency
(2 threads 1.12×, ≥3 threads degrade to 0.2–0.7×) — for fp32, concatenate
ICs into one batch instead. Results are bit-identical whether a call ran alone
or concurrently — streams change scheduling, never arithmetic.

Sampler integration is one line: hand `emcee`/`dynesty`/... a *thread* pool
(`multiprocessing.pool.ThreadPool` or the executor above), **not** a process
pool. Threads beyond card saturation queue harmlessly.

---

## 4. Multiple independent scripts: read this before batch-launching

Streams (and therefore Path A) exist *inside one process*. Separate Python
scripts get separate CUDA **contexts**, and the driver **time-slices**
contexts — their kernels take turns on the GPU and never co-execute:

- **A single script using §2 or §3 already saturates the GPU.** There is no
  spare capacity for a second script to claim; concurrent scripts just divide
  the same card and add context-switch overhead (measured on the WSL2 dev box:
  4 concurrent processes ran **2.4× slower in aggregate** than the same work
  issued sequentially from one process).
- Running "N scripts, each with a ThreadPoolExecutor" therefore does NOT
  multiply throughput: each script saturates the card *during its own time
  slices*, and total throughput stays at (slightly below) the single-script
  ceiling while every script's latency inflates ~N×.
- **On the WSL2 laptop:** run GPU scripts one at a time (a trivial job queue /
  `for f in *.py; do python $f; done` is the right tool). NVIDIA MPS is not
  available under WSL2.
- **On a native-Linux deployment box (L40/A100):** start NVIDIA MPS
  (`nvidia-cuda-mps-control -d`) and the many-scripts pattern becomes
  efficient — MPS funnels all processes into one shared context, so their
  kernels co-schedule like streams. No AGAMA or script changes required.

| Pattern | WSL2 laptop | Linux box + MPS |
|---|---|---|
| one script, one big batched call (§2) | **saturates** | saturates |
| one script, thread pool (§3, with Path A) | **saturates** | saturates |
| one script, serial loop of small calls | under-fills — batch it | under-fills — batch it |
| many scripts in parallel | slower than sequential — don't | efficient |

---

## 5. What throughput to expect (RTX 3080 Laptop, 16-core host, 2026-07)

Composite bulge+disk+halo, T=100, 64 samples/orbit:

| configuration | orbits/s |
|---|---|
| legacy CPU loop (fp64, 16 cores) | ~31k |
| `device='cpu'` batch fp64 | ~71k |
| `device='cuda'` fp64, 5K orbits | ~30k (card under-filled) |
| `device='cuda'` fp64, 50K orbits | ~60k |
| `device='cuda'` fp32 `accuracy=1e-5`, 5K orbits | ~220k |
| `device='cuda'` fp32 `accuracy=1e-5`, 20–50K orbits | ~250–310k |

Consumer cards run fp64 at 1/64 rate — on a datacenter card (A100/H100, 1:2)
the fp64 rows behave like the fp32 rows. Heavier potentials (Multipole /
CylSpline, Tier 2) raise the arithmetic per orbit-step and shift the GPU
advantage to smaller batch sizes.
