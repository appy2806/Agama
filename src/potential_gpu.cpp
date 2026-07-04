/** \file    potential_gpu.cpp
    \brief   GPU-routed dispatch for batch potential evaluation. Compiled by nvcc
             (CUDA_TUS in Makefile.list) when HAVE_CUDA=1; compiles cleanly under
             g++ when HAVE_CUDA=0 with the "cuda" branch returning POT_GPU_ENOTBUILT.
    \author  GPU-unification fork
*/
#include "potential_gpu.h"
#include "potential_analytic.h"
#include "potential_base.h"
#include "potential_composite.h"
#include "gpu_policy.h"
#include <cstring>
#include <string>
#include <stdexcept>
#ifdef HAVE_CUDA
#include <mutex>
#endif

namespace potential {

namespace {

#ifdef HAVE_CUDA
/** Process-wide persistent GPU scratch buffers for the Python entry path.
    Lazily allocated via the C++11 function-static-init guarantee; grown
    (never shrunk) by reserve() to the largest N seen so far. Reused across
    every call into evalPotentialGPU<T> with device="cuda" -- eliminates the
    per-call cudaMalloc / cudaFree pair (~1-2 ms each) we previously paid.

    Intentional leak: the device_array lives behind a raw `new` and is never
    deleted (same pattern as detail::get_reduce_scratch in gpu_policy.h), so
    its destructor cannot run during static-destruction order at process exit,
    after the CUDA context may already have been torn down. The driver
    reclaims the memory on context destruction.

    Thread-safety: the Python binding releases the GIL around evalPotentialGPU
    (Py_BEGIN_ALLOW_THREADS in interface_python.cpp), so multiple Python
    threads CAN be inside the cuda branch concurrently. All access to these
    shared buffers is therefore serialized by gpu_scratch_mutex() in
    evalPotentialGPU: without it, one thread's reserve() could cudaFree a
    buffer another thread's kernel is still reading, and concurrent from_host
    calls would corrupt each other's inputs. The path is fully synchronous
    (blocking memcpys + stream-sync), so serialization costs nothing. */
template<typename T>
agama::device_array<T>& gpu_scratch_xyz() {
    static agama::device_array<T>* s = new agama::device_array<T>();  return *s;
}
template<typename T>
agama::device_array<T>& gpu_scratch_phi() {
    static agama::device_array<T>* s = new agama::device_array<T>();  return *s;
}

/** Single mutex serializing all use of the persistent scratch buffers above
    (shared across both precisions T -- simpler to reason about, and the GPU
    would serialize the work anyway). Function-local static: constructed on
    first use, thread-safe init per C++11. */
std::mutex& gpu_scratch_mutex() {
    static std::mutex m;  return m;
}
#endif

/** The single list of GPU-capable concrete potential types, X-macro style.
    Consumed twice -- by can_dispatch() and try_dispatch() below -- so the two
    can never drift apart. It grows as more potentials get migrated. Order
    doesn't matter for correctness (dynamic_cast is exact); listing the
    most-common first is a tiny perf hint. */
#define AGAMA_GPU_POT_LIST(X) \
    X(NFW)           \
    X(Plummer)       \
    X(Isochrone)     \
    X(MiyamotoNagai) \
    X(Logarithmic)   \
    X(Harmonic)

/** Capability predicate: true iff try_dispatch() would succeed for this
    potential. A Composite is dispatchable iff ALL of its members are
    (recursively, so nested composites work). Non-templated: capability
    depends only on the concrete type, not on precision or policy. */
bool can_dispatch(const BasePotential& pot)
{
    #define AGAMA_GPU_CAN(PotClass) \
        if(dynamic_cast<const PotClass*>(&pot) != NULL) return true;
    AGAMA_GPU_POT_LIST(AGAMA_GPU_CAN)
    #undef AGAMA_GPU_CAN
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!can_dispatch(*comp->component(c)))
                return false;
        return true;
    }
    return false;
}

/** Try-dispatch on concrete potential type. Returns true if `pot` matched one
    of the known GPU-capable types and the batch eval was invoked. The lambda
    captures `pol`, `xyz_p`, `phi_p` by value; pol determines whether the
    forall() inside each evalmanyCarT becomes a Serial/OpenMP loop or a CUDA
    kernel launch.

    `add` selects accumulate-into-phi instead of overwrite (default false,
    preserving all pre-composite call sites). A Composite is evaluated by
    looping over its members ON THE SAME BUFFERS: the first member stores,
    each subsequent member accumulates (add=true) -- all the summation happens
    wherever phi_p lives, so for the Cuda policy there are ZERO extra
    host/device transfers. can_dispatch() is checked up front so a composite
    containing any unsupported member fails cleanly without partially writing
    phi_p.

    Why C++14 generic lambdas: lets the same dispatch body service three
    different Policy types without three copies. The compiler instantiates a
    distinct dispatch function per (T, Policy) pair. */
template<typename T, class Policy>
bool try_dispatch(const BasePotential& pot, Policy pol,
                  std::size_t N, const T* xyz_p, T* phi_p, bool add = false)
{
    #define AGAMA_GPU_TRY(PotClass) \
        if(const PotClass* p = dynamic_cast<const PotClass*>(&pot)) { \
            p->template evalmanyCarT<T>(pol, N, xyz_p, phi_p, add);   \
            return true;                                              \
        }
    AGAMA_GPU_POT_LIST(AGAMA_GPU_TRY)
    #undef AGAMA_GPU_TRY
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        if(!can_dispatch(pot))   // all-or-nothing: don't partially write phi_p
            return false;
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!try_dispatch<T>(*comp->component(c), pol, N, xyz_p, phi_p,
                                /*add*/ add || c > 0))
                return false;    // unreachable after the can_dispatch check
        return true;
    }
    return false;
}

}  // anonymous namespace

std::string unsupportedGPUPotentialName(const BasePotential& pot)
{
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!can_dispatch(*comp->component(c)))
                return unsupportedGPUPotentialName(*comp->component(c)) +
                    " (component #" + std::to_string(c) + " of the composite)";
    }
    return pot.name();
}

template<typename T>
int evalPotentialGPU(const BasePotential& pot,
                     std::size_t N, const T* xyz_h, T* phi_h,
                     const char* device)
{
    // Device-string semantics:
    //   "cpu"    -> OpenMP.  Users read 'cpu' as "what AGAMA normally does",
    //               and the legacy CPU path has always been OpenMP-parallel
    //               across all cores.  This is the honest CPU baseline.
    //   "openmp" -> OpenMP (explicit alias of "cpu").
    //   "serial" -> Serial single-thread loop (debugging / baseline timing).
    //   "cuda"   -> GPU batch path (HAVE_CUDA=1 builds only).
    if(std::strcmp(device, "cpu") == 0 || std::strcmp(device, "openmp") == 0) {
        return try_dispatch<T>(pot, agama::OpenMP{}, N, xyz_h, phi_h)
            ? POT_GPU_OK : POT_GPU_EUNSUPP;
    }
    if(std::strcmp(device, "serial") == 0) {
        return try_dispatch<T>(pot, agama::Serial{}, N, xyz_h, phi_h)
            ? POT_GPU_OK : POT_GPU_EUNSUPP;
    }
    if(std::strcmp(device, "cuda") == 0) {
#ifdef HAVE_CUDA
        // Nothing to do for an empty batch -- and skipping it avoids a
        // cudaMemcpy on the (still-null) scratch pointer.
        if(N == 0)
            return POT_GPU_OK;
        // Serialize the whole branch: the Python binding drops the GIL around
        // this call, so concurrent Python threads can land here simultaneously.
        // The shared scratch buffers (reserve can cudaFree a buffer another
        // thread's kernel is reading; from_host would interleave inputs) make
        // the entire reserve + upload + dispatch + download sequence a critical
        // section. The path is fully synchronous, so this costs nothing.
        std::lock_guard<std::mutex> lock(gpu_scratch_mutex());
        // Reuse process-wide scratch buffers instead of fresh cudaMalloc / cudaFree
        // every call. reserve() is a no-op when the current allocation is large
        // enough, so steady-state-N workloads pay zero allocation cost after warm-up.
        agama::device_array<T>& d_xyz = gpu_scratch_xyz<T>();
        agama::device_array<T>& d_phi = gpu_scratch_phi<T>();
        d_xyz.reserve(N * 3);
        d_phi.reserve(N);
        d_xyz.from_host(xyz_h, N * 3);
        bool ok = try_dispatch<T>(pot, agama::Cuda{}, N, d_xyz.data(), d_phi.data());
        if(!ok) return POT_GPU_EUNSUPP;
        d_phi.to_host(phi_h, N);
        return POT_GPU_OK;
#else
        return POT_GPU_ENOTBUILT;
#endif
    }
    return POT_GPU_EBADDEV;
}

template<typename T>
int evalPotentialGPUDevice(const BasePotential& pot,
                           std::size_t N, const T* d_xyz, T* d_phi,
                           unsigned long long input_stream)
{
#ifdef HAVE_CUDA
    // No scratch, no mutex: this path touches no shared state -- the kernel
    // reads/writes the caller's own device buffers, so concurrent calls from
    // multiple Python threads are safe without serialization (the driver
    // serializes work on the default stream anyway).
    //
    // CAI v3 'stream' semantics: 0 = key absent (producer gave no stream --
    // assume data is ready, the common CuPy default-stream case); 1 = legacy
    // default stream (same stream our kernel launches on, ordering automatic);
    // 2 = per-thread default stream sentinel. Any other value is a real
    // producer stream handle: synchronize it before launching so pending
    // writes to d_xyz are visible to our default-stream kernel.
    if(input_stream != 0 && input_stream != 1 && input_stream != 2) {
        // A real stream handle is a pointer to a driver-internal struct which
        // cudaStreamSynchronize dereferences WITHOUT validation -- a garbage
        // value from a malformed CAI dict segfaults inside the driver (no
        // catchable error is ever returned). Cheap plausibility check: any
        // genuine pointer is at least word-aligned, so reject misaligned
        // values with a catchable exception (the Python binding converts it
        // to RuntimeError). Aligned garbage remains UB, as in the CAI spec.
        if(input_stream % alignof(void*) != 0)
            throw std::runtime_error(
                "__cuda_array_interface__ 'stream' value " +
                std::to_string(input_stream) +
                " is not a plausible CUDA stream handle");
        AGAMA_CUDA_CHECK(cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(input_stream)));
    }
    if(!try_dispatch<T>(pot, agama::Cuda{}, N, d_xyz, d_phi))
        return POT_GPU_EUNSUPP;
    // Simple synchronous v1 semantics: result is fully materialized in d_phi
    // when we return, so the Python caller may consume it on any stream.
    AGAMA_CUDA_CHECK(cudaStreamSynchronize(0));
    return POT_GPU_OK;
#else
    (void)pot; (void)N; (void)d_xyz; (void)d_phi; (void)input_stream;
    return POT_GPU_ENOTBUILT;
#endif
}

// Explicit instantiations for the two precisions the Python boundary exposes.
template int evalPotentialGPU<float >(const BasePotential&, std::size_t,
                                      const float*,  float*,  const char*);
template int evalPotentialGPU<double>(const BasePotential&, std::size_t,
                                      const double*, double*, const char*);
template int evalPotentialGPUDevice<float >(const BasePotential&, std::size_t,
                                            const float*,  float*,  unsigned long long);
template int evalPotentialGPUDevice<double>(const BasePotential&, std::size_t,
                                            const double*, double*, unsigned long long);

}  // namespace potential
