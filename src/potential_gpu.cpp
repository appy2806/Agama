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
#include "potential_dehnen.h"
#include "potential_disk.h"
#include "potential_descriptor.h"
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
// N*3 output scratch for the acceleration path (the phi scratch above is reused
// as-is for the length-N density output). Same lifetime/locking rules.
template<typename T>
agama::device_array<T>& gpu_scratch_acc() {
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
    X(Harmonic)      \
    X(UniformAcceleration)

/** True iff `pot` is one of the four modifier wrappers. A modifier is handled
    not by its own batch method but by collapsing the whole wrapper chain into
    the GpuPotXform of a descriptor term (see potential_descriptor.h) and running
    the descriptor kernel -- so it needs a separate predicate from the
    concrete-type list above. */
bool is_modifier(const BasePotential& pot)
{
    return dynamic_cast<const Shifted <BasePotential>*>(&pot) != NULL
        || dynamic_cast<const Tilted  <BasePotential>*>(&pot) != NULL
        || dynamic_cast<const Rotating<BasePotential>*>(&pot) != NULL
        || dynamic_cast<const Scaled  <BasePotential>*>(&pot) != NULL;
}

/** Capability predicate: true iff try_dispatch() would succeed for this
    potential. A Composite is dispatchable iff ALL of its members are
    (recursively, so nested composites work). Non-templated: capability
    depends only on the concrete type, not on precision or policy. */
bool can_dispatch(const BasePotential& pot, double time = 0)
{
    #define AGAMA_GPU_CAN(PotClass) \
        if(dynamic_cast<const PotClass*>(&pot) != NULL) return true;
    AGAMA_GPU_POT_LIST(AGAMA_GPU_CAN)
    #undef AGAMA_GPU_CAN
    // A modifier chain is dispatchable iff buildGpuPotDesc can represent whatever
    // it wraps AND the transform it collapses to at this `time` is finite. That
    // function is the single authority on both questions, so ask it rather than
    // duplicating the rules -- and ask it at the SAME time try_dispatch will use,
    // so the composite all-or-nothing pre-check below cannot pass here and then
    // fail there after partially writing the outputs.
    if(is_modifier(pot)) {
        GpuPotDesc<double> desc;
        return buildGpuPotDesc(pot, desc, time, /*requireTimeIndependent*/ false);
    }
    // Dehnen is deliberately NOT in AGAMA_GPU_POT_LIST above: it is only
    // GPU-dispatchable in the spherical case (axisRatioY==axisRatioZ==1) --
    // the triaxial potential needs math::integrate, which has no device path.
    // Special-cased here (and in try_dispatch below) instead of folded into
    // the blanket macro, which grants unconditional capability per type.
    if(const Dehnen* p = dynamic_cast<const Dehnen*>(&pot))
        return isSpherical(p->symmetry());
    // DiskAnsatz is likewise not in AGAMA_GPU_POT_LIST above: it is only
    // GPU-dispatchable when its stored radial/vertical functors are one of
    // the 5 recognized closed-form types (see recognizeDiskAnsatz /
    // DiskAnsatz::gpuDesc in potential_disk.h); an instance built from
    // arbitrary user functions (the second constructor) has no device path.
    if(const DiskAnsatz* p = dynamic_cast<const DiskAnsatz*>(&pot)) {
        DiskAnsatzDesc<double> d;
        return p->gpuDesc(d);
    }
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!can_dispatch(*comp->component(c), time))
                return false;
        return true;
    }
    return false;
}

/** Which batch operation try_dispatch routes to. One dispatch body serves all
    three so the AGAMA_GPU_POT_LIST stays single-sourced. */
enum EvalMode {
    MODE_PHI,   ///< evalmanyCarT:       out1 = Phi (length N),     out3 unused
    MODE_ACC,   ///< evalmanyPhiAccCarT: out1 = Phi (nullable),     out3 = acc (length 3*N)
    MODE_DENS   ///< evalmanyDensCarT:   out1 = rho (length N),     out3 unused
};

/** Batch evaluator driven by a GpuPotDesc instead of by a concrete class: the
    path taken by any potential wrapped in one or more modifiers.

    Why the descriptor and not the class's own evalmany*T: a modifier's transform
    has to be applied to the input position and to the output vector, and the
    per-class batch methods have no way to accept one. Folding it into a
    descriptor term instead keeps this to a SINGLE kernel launch with no scratch
    buffers -- the alternative (a pre-transform kernel, the class kernel, then a
    post-transform kernel, plus an N*3 scratch buffer for the transformed
    positions) costs three launches and a policy-dependent allocation. It also
    means arbitrarily nested chains cost the same as one modifier.

    Unmodified potentials deliberately do NOT come through here: they keep their
    existing class-dispatch path, so their results stay bit-for-bit unchanged.

    `mode` is resolved on the host, before the launch, so the kernels contain no
    branch on it.

    MODE_PHI here goes through the fused Phi+acceleration leaf and throws the
    acceleration away, where the class path would have used a Phi-only leaf.
    Everything inlines, so the dead derivative work is a plain DCE candidate;
    adding a third Phi-only switch over the tags to guarantee it was judged not
    worth the duplicated per-tag glue, since Phi-alone on a modified potential is
    not a hot path (force and orbits are, and both need the derivative anyway).
    Revisit if a profile ever says otherwise. */
template<typename T, class Policy>
void evalmanyDescT(Policy pol, std::size_t N, const T* xyz, EvalMode mode,
                   T* out1, T* out3, const GpuPotDesc<T>& desc, bool add)
{
    switch(mode) {
    case MODE_PHI:
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            T phi, acc[3];
            gpu_desc_phi_acc(desc, xyz[i*3+0], xyz[i*3+1], xyz[i*3+2], &phi, acc);
            out1[i] = add ? out1[i] + phi : phi;
        });
        break;
    case MODE_ACC:
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            T phi, acc[3];
            gpu_desc_phi_acc(desc, xyz[i*3+0], xyz[i*3+1], xyz[i*3+2],
                out1 ? &phi : (T*)NULL, acc);
            if(out1)
                out1[i] = add ? out1[i] + phi : phi;
            out3[i*3+0] = add ? out3[i*3+0] + acc[0] : acc[0];
            out3[i*3+1] = add ? out3[i*3+1] + acc[1] : acc[1];
            out3[i*3+2] = add ? out3[i*3+2] + acc[2] : acc[2];
        });
        break;
    case MODE_DENS:
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T v = gpu_desc_dens(desc, xyz[i*3+0], xyz[i*3+1], xyz[i*3+2]);
            out1[i] = add ? out1[i] + v : v;
        });
        break;
    }
}

/** Try-dispatch on concrete potential type. Returns true if `pot` matched one
    of the known GPU-capable types and the batch eval was invoked. The lambda
    captures `pol` and the buffer pointers by value; pol determines whether the
    forall() inside each evalmany* becomes a Serial/OpenMP loop or a CUDA
    kernel launch. `mode` selects Phi / fused Phi+acc / density; the switch is
    uniform across the batch so it costs nothing inside the kernels (it is
    resolved on the host before launch).

    `add` selects accumulate-into-output instead of overwrite (default false,
    preserving all pre-composite call sites). A Composite is evaluated by
    looping over its members ON THE SAME BUFFERS: the first member stores,
    each subsequent member accumulates (add=true) -- all the summation happens
    wherever the output lives, so for the Cuda policy there are ZERO extra
    host/device transfers. This works uniformly for Phi, acceleration and
    density (composite density = sum of member densities). can_dispatch() is
    checked up front so a composite containing any unsupported member fails
    cleanly without partially writing the outputs.

    Why C++14 generic lambdas: lets the same dispatch body service three
    different Policy types without three copies. The compiler instantiates a
    distinct dispatch function per (T, Policy) pair. */
template<typename T, class Policy>
bool try_dispatch(const BasePotential& pot, Policy pol,
                  std::size_t N, const T* xyz_p, EvalMode mode,
                  T* out1, T* out3, double time = 0, bool add = false)
{
    #define AGAMA_GPU_TRY(PotClass) \
        if(const PotClass* p = dynamic_cast<const PotClass*>(&pot)) {              \
            switch(mode) {                                                         \
            case MODE_PHI:  p->template evalmanyCarT<T>(pol, N, xyz_p, out1,       \
                                time, add);                                        \
                            break;                                                 \
            case MODE_ACC:  p->template evalmanyPhiAccCarT<T>(pol, N, xyz_p,       \
                                out1, out3, time, add);                            \
                            break;                                                 \
            case MODE_DENS: p->template evalmanyDensCarT<T>(pol, N, xyz_p, out1,   \
                                time, add);                                        \
                            break;                                                 \
            }                                                                      \
            return true;                                                           \
        }
    AGAMA_GPU_POT_LIST(AGAMA_GPU_TRY)
    #undef AGAMA_GPU_TRY
    if(const Dehnen* p = dynamic_cast<const Dehnen*>(&pot)) {
        if(!isSpherical(p->symmetry()))
            return false;   // triaxial: not GPU-dispatchable (see can_dispatch above)
        switch(mode) {
        case MODE_PHI:  p->template evalmanyCarT<T>(pol, N, xyz_p, out1, time, add);
                        break;
        case MODE_ACC:  p->template evalmanyPhiAccCarT<T>(pol, N, xyz_p, out1, out3, time, add);
                        break;
        case MODE_DENS: p->template evalmanyDensCarT<T>(pol, N, xyz_p, out1, time, add);
                        break;
        }
        return true;
    }
    if(const DiskAnsatz* p = dynamic_cast<const DiskAnsatz*>(&pot)) {
        DiskAnsatzDesc<double> d;
        if(!p->gpuDesc(d))
            return false;   // arbitrary user functions: not GPU-dispatchable (see can_dispatch above)
        switch(mode) {
        case MODE_PHI:  p->template evalmanyCarT<T>(pol, N, xyz_p, out1, time, add);
                        break;
        case MODE_ACC:  p->template evalmanyPhiAccCarT<T>(pol, N, xyz_p, out1, out3, time, add);
                        break;
        case MODE_DENS: p->template evalmanyDensCarT<T>(pol, N, xyz_p, out1, time, add);
                        break;
        }
        return true;
    }
    // Modifier chain (Shifted / Tilted / Rotating / Scaled, possibly nested):
    // collapse the whole chain plus the object it wraps into a descriptor and
    // run the descriptor kernel. Baking the transform in at `time` is EXACT
    // here, not an approximation, because the whole batch shares one `time`.
    if(is_modifier(pot)) {
        GpuPotDesc<double> desc0;
        if(!buildGpuPotDesc(pot, desc0, time, /*requireTimeIndependent*/ false))
            return false;
        evalmanyDescT<T>(pol, N, xyz_p, mode, out1, out3,
            castGpuPotDesc<T>(desc0), add);
        return true;
    }
    if(const Composite* comp = dynamic_cast<const Composite*>(&pot)) {
        if(!can_dispatch(pot, time))   // all-or-nothing: don't partially write outputs
            return false;
        for(unsigned int c = 0; c < comp->size(); c++)
            if(!try_dispatch<T>(*comp->component(c), pol, N, xyz_p, mode,
                                out1, out3, time, /*add*/ add || c > 0))
                return false;    // unreachable after the can_dispatch check
        return true;
    }
    return false;
}

/** Shared host-pointer entry: routes on the device string and runs `mode`.
    out1 (length N, may be NULL for the acc-only force path) and out3 (length
    3*N, NULL except MODE_ACC) are host buffers; the cuda branch stages them
    through the persistent mutex-guarded scratch buffers. */
template<typename T>
int evalGPUHost(const BasePotential& pot,
                std::size_t N, const T* xyz_h, EvalMode mode,
                T* out1_h, T* out3_h, const char* device, double time)
{
    // Device-string semantics:
    //   "cpu"    -> OpenMP.  Users read 'cpu' as "what AGAMA normally does",
    //               and the legacy CPU path has always been OpenMP-parallel
    //               across all cores.  This is the honest CPU baseline.
    //   "openmp" -> OpenMP (explicit alias of "cpu").
    //   "serial" -> Serial single-thread loop (debugging / baseline timing).
    //   "cuda"   -> GPU batch path (HAVE_CUDA=1 builds only).
    if(std::strcmp(device, "cpu") == 0 || std::strcmp(device, "openmp") == 0) {
        return try_dispatch<T>(pot, agama::OpenMP{}, N, xyz_h, mode, out1_h, out3_h, time)
            ? POT_GPU_OK : POT_GPU_EUNSUPP;
    }
    if(std::strcmp(device, "serial") == 0) {
        return try_dispatch<T>(pot, agama::Serial{}, N, xyz_h, mode, out1_h, out3_h, time)
            ? POT_GPU_OK : POT_GPU_EUNSUPP;
    }
    if(std::strcmp(device, "cuda") == 0) {
#ifdef HAVE_CUDA
        // Nothing to do for an empty batch -- and skipping it avoids a
        // cudaMemcpy on the (still-null) scratch pointers.
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
        d_xyz.reserve(N * 3);
        d_xyz.from_host(xyz_h, N * 3);
        T* d_out1 = NULL;
        T* d_out3 = NULL;
        if(out1_h) {
            agama::device_array<T>& s = gpu_scratch_phi<T>();  // also serves rho (length N)
            s.reserve(N);
            d_out1 = s.data();
        }
        if(out3_h) {
            agama::device_array<T>& s = gpu_scratch_acc<T>();
            s.reserve(N * 3);
            d_out3 = s.data();
        }
        if(!try_dispatch<T>(pot, agama::Cuda{}, N, d_xyz.data(), mode, d_out1, d_out3, time))
            return POT_GPU_EUNSUPP;
        if(out1_h)
            gpu_scratch_phi<T>().to_host(out1_h, N);
        if(out3_h)
            gpu_scratch_acc<T>().to_host(out3_h, N * 3);
        return POT_GPU_OK;
#else
        (void)out1_h; (void)out3_h; (void)time;
        return POT_GPU_ENOTBUILT;
#endif
    }
    return POT_GPU_EBADDEV;
}

/** Shared device-pointer entry (CAI passthrough): runs `mode` directly on the
    caller's GPU buffers. Same synchronization contract as evalPotentialGPUDevice
    (documented in potential_gpu.h). */
template<typename T>
int evalGPUDeviceCommon(const BasePotential& pot,
                        std::size_t N, const T* d_xyz, EvalMode mode,
                        T* d_out1, T* d_out3, unsigned long long input_stream,
                        double time)
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
    if(!try_dispatch<T>(pot, agama::Cuda{}, N, d_xyz, mode, d_out1, d_out3, time))
        return POT_GPU_EUNSUPP;
    // Simple synchronous v1 semantics: result is fully materialized in the
    // output buffer(s) when we return, so the Python caller may consume it on
    // any stream.
    AGAMA_CUDA_CHECK(cudaStreamSynchronize(0));
    return POT_GPU_OK;
#else
    (void)pot; (void)N; (void)d_xyz; (void)mode; (void)d_out1; (void)d_out3;
    (void)input_stream; (void)time;
    return POT_GPU_ENOTBUILT;
#endif
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
                     const char* device, double time)
{
    return evalGPUHost<T>(pot, N, xyz_h, MODE_PHI, phi_h, /*out3*/ NULL, device, time);
}

template<typename T>
int evalForceGPU(const BasePotential& pot,
                 std::size_t N, const T* xyz_h, T* acc_h,
                 const char* device, double time)
{
    // Phi output disabled (NULL out1): the fused kernel computes acceleration only.
    return evalGPUHost<T>(pot, N, xyz_h, MODE_ACC, /*out1*/ NULL, acc_h, device, time);
}

template<typename T>
int evalDensityGPU(const BasePotential& pot,
                   std::size_t N, const T* xyz_h, T* rho_h,
                   const char* device, double time)
{
    return evalGPUHost<T>(pot, N, xyz_h, MODE_DENS, rho_h, /*out3*/ NULL, device, time);
}

template<typename T>
int evalPotentialGPUDevice(const BasePotential& pot,
                           std::size_t N, const T* d_xyz, T* d_phi,
                           unsigned long long input_stream, double time)
{
    return evalGPUDeviceCommon<T>(pot, N, d_xyz, MODE_PHI, d_phi, NULL, input_stream, time);
}

template<typename T>
int evalForceGPUDevice(const BasePotential& pot,
                       std::size_t N, const T* d_xyz, T* d_acc,
                       unsigned long long input_stream, double time)
{
    return evalGPUDeviceCommon<T>(pot, N, d_xyz, MODE_ACC, /*d_out1*/ NULL, d_acc,
        input_stream, time);
}

template<typename T>
int evalDensityGPUDevice(const BasePotential& pot,
                         std::size_t N, const T* d_xyz, T* d_rho,
                         unsigned long long input_stream, double time)
{
    return evalGPUDeviceCommon<T>(pot, N, d_xyz, MODE_DENS, d_rho, NULL, input_stream, time);
}

// Explicit instantiations for the two precisions the Python boundary exposes.
template int evalPotentialGPU<float >(const BasePotential&, std::size_t,
                                      const float*,  float*,  const char*, double);
template int evalPotentialGPU<double>(const BasePotential&, std::size_t,
                                      const double*, double*, const char*, double);
template int evalForceGPU<float >(const BasePotential&, std::size_t,
                                  const float*,  float*,  const char*, double);
template int evalForceGPU<double>(const BasePotential&, std::size_t,
                                  const double*, double*, const char*, double);
template int evalDensityGPU<float >(const BasePotential&, std::size_t,
                                    const float*,  float*,  const char*, double);
template int evalDensityGPU<double>(const BasePotential&, std::size_t,
                                    const double*, double*, const char*, double);
template int evalPotentialGPUDevice<float >(const BasePotential&, std::size_t,
                                            const float*,  float*,  unsigned long long, double);
template int evalPotentialGPUDevice<double>(const BasePotential&, std::size_t,
                                            const double*, double*, unsigned long long, double);
template int evalForceGPUDevice<float >(const BasePotential&, std::size_t,
                                        const float*,  float*,  unsigned long long, double);
template int evalForceGPUDevice<double>(const BasePotential&, std::size_t,
                                        const double*, double*, unsigned long long, double);
template int evalDensityGPUDevice<float >(const BasePotential&, std::size_t,
                                          const float*,  float*,  unsigned long long, double);
template int evalDensityGPUDevice<double>(const BasePotential&, std::size_t,
                                          const double*, double*, unsigned long long, double);

}  // namespace potential
