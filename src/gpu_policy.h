// gpu_policy.h — single-source CPU/GPU execution policy for AGAMA.
//
// Provides:
//   - Policy tags: Serial, OpenMP, Cuda
//   - AGAMA_DEVICE / AGAMA_DEVICE_INLINE: no-ops on host-only builds,
//     __host__ __device__ when compiled by nvcc.
//   - forall(Policy, N, lambda): the workhorse parallel-for primitive.
//   - parallel_reduce_sum(Policy, N, init, op): sum reduction over [0, N).
//   - device_array<T>: RAII move-only wrapper around cudaMalloc/cudaMemcpy.
//     Falls back to a thin std::vector wrapper with the same API on CPU builds.
//
// The lambda passed to forall/parallel_reduce_sum must be AGAMA_DEVICE-callable
// when the Cuda policy is used. Use the AGAMA_DEVICE macro on the lambda:
//
//     forall(Cuda{}, N, [=] AGAMA_DEVICE (std::size_t i) { ... });
//
// nvcc must be invoked with --expt-extended-lambda to allow this on Cuda.
//
// Language standard: header itself is C++11-compatible. C++14 (or later) is
// required for the surrounding code that uses generic lambdas.

#pragma once

#include "gpu_device.h"   // AGAMA_DEVICE / AGAMA_DEVICE_INLINE macros
#include <cstddef>
#include <vector>
#include <algorithm>

#ifdef HAVE_CUDA
  #include <cuda_runtime.h>
  #include <stdexcept>
  #include <string>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace agama {

// ---------------------------------------------------------------------------
// Policy tags. forall(Policy, ...) dispatches on these.
// ---------------------------------------------------------------------------
struct Serial {};
struct OpenMP {};

#ifdef HAVE_CUDA
struct Cuda {
    int block_size = 256;
    cudaStream_t stream = 0;  // 0 = default stream
};
#endif

// ---------------------------------------------------------------------------
// CUDA error check. Throws std::runtime_error on failure with the CUDA
// error string attached. AGAMA's existing error path is exception-based.
// ---------------------------------------------------------------------------
#ifdef HAVE_CUDA
inline void agama_cuda_check(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        std::string msg = "CUDA error at ";
        msg += file; msg += ":"; msg += std::to_string(line);
        msg += " ("; msg += expr; msg += "): ";
        msg += cudaGetErrorString(e);
        throw std::runtime_error(msg);
    }
}
#define AGAMA_CUDA_CHECK(expr) ::agama::agama_cuda_check((expr), #expr, __FILE__, __LINE__)
#endif

// ===========================================================================
// forall — parallel-for over [0, N).
// ===========================================================================

template<class F>
inline void forall(Serial, std::size_t N, F f) {
    for (std::size_t i = 0; i < N; ++i) f(i);
}

template<class F>
inline void forall(OpenMP, std::size_t N, F f) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i)
        f(static_cast<std::size_t>(i));
}

// CUDA-policy overloads of forall live below. They are visible only inside an nvcc
// translation unit (gated on __CUDACC__, not just HAVE_CUDA) because the body uses
// kernel-launch syntax `<<<...>>>` which is an nvcc-only extension — g++ cannot parse
// it, even inside an uninstantiated template. A .cpp file that calls forall(Cuda{},...)
// must therefore be compiled through nvcc (see CUDA_TUS in Makefile.list).
#if defined(HAVE_CUDA) && defined(__CUDACC__)
namespace detail {
template<class F>
__global__ void agama_forall_k(std::size_t N, F f) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) f(i);
}
}  // namespace detail

template<class F>
inline void forall(Cuda p, std::size_t N, F f) {
    if (N == 0) return;
    std::size_t blocks = (N + p.block_size - 1) / p.block_size;
    detail::agama_forall_k<<<static_cast<unsigned>(blocks), p.block_size, 0, p.stream>>>(N, f);
    AGAMA_CUDA_CHECK(cudaPeekAtLastError());
}
#endif

// ===========================================================================
// parallel_reduce_sum — sum reduction of op(i) for i in [0, N).
// op must return a value implicitly convertible to T.
// ===========================================================================

template<class T, class Op>
inline T parallel_reduce_sum(Serial, std::size_t N, T init, Op op) {
    T acc = init;
    for (std::size_t i = 0; i < N; ++i) acc += op(i);
    return acc;
}

template<class T, class Op>
inline T parallel_reduce_sum(OpenMP, std::size_t N, T init, Op op) {
    T acc = init;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:acc) schedule(static)
    #endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i)
        acc += op(static_cast<std::size_t>(i));
    return acc;
}

// CUDA-policy overload of parallel_reduce_sum — nvcc-only for the same reason as forall(Cuda).
#if defined(HAVE_CUDA) && defined(__CUDACC__)
namespace detail {
// Per-thread, per-T persistent device scratch slot of size sizeof(T).
// Allocated lazily on first call, never freed (CUDA driver releases on context
// destruction). Eliminates the ~10 µs cudaMallocAsync + cudaFreeAsync fixed
// cost that parallel_reduce_sum<Cuda> would otherwise pay every call.
// Why thread_local: lets parallel_reduce_sum<Cuda> be called concurrently from
// different CPU threads (e.g. OpenMP host code dispatching to GPU) without the
// scratch becoming a race target. Within a single CPU thread, sequential
// reduce calls reuse the same slot — safe because each call ends with
// cudaStreamSynchronize before returning.
// Static thread_local in an inline function template has external linkage,
// so the slot is shared across TUs that share the template instantiation.
template<class T>
inline T* get_reduce_scratch() {
    static thread_local T* d_scratch = nullptr;
    if (d_scratch == nullptr) {
        AGAMA_CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(T)));
    }
    return d_scratch;
}

// Block-wide tree reduction in shared memory, then atomicAdd to global.
// Assumes block_size is a power of two and <= 1024.
template<class T, class Op>
__global__ void agama_reduce_sum_k(std::size_t N, Op op, T* __restrict__ out) {
    extern __shared__ unsigned char smem_raw[];
    T* sdata = reinterpret_cast<T*>(smem_raw);
    unsigned tid = threadIdx.x;
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    sdata[tid] = (i < N) ? static_cast<T>(op(i)) : T(0);
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}
}  // namespace detail

template<class T, class Op>
inline T parallel_reduce_sum(Cuda p, std::size_t N, T init, Op op) {
    if (N == 0) return init;
    // Clamp block size to a power of two within [32, 1024].
    int bs = p.block_size;
    if (bs < 32)  bs = 32;
    if (bs > 1024) bs = 1024;
    // round down to nearest power of two
    int pow2 = 1; while (pow2 * 2 <= bs) pow2 *= 2; bs = pow2;
    // Use the cached per-thread scratch slot; no per-call cudaMalloc/cudaFree.
    T* d_result = detail::get_reduce_scratch<T>();
    AGAMA_CUDA_CHECK(cudaMemcpyAsync(d_result, &init, sizeof(T),
                                     cudaMemcpyHostToDevice, p.stream));
    std::size_t blocks = (N + bs - 1) / bs;
    std::size_t smem_bytes = static_cast<std::size_t>(bs) * sizeof(T);
    detail::agama_reduce_sum_k<<<static_cast<unsigned>(blocks), bs, smem_bytes, p.stream>>>(
        N, op, d_result);
    AGAMA_CUDA_CHECK(cudaPeekAtLastError());
    T result;
    AGAMA_CUDA_CHECK(cudaMemcpyAsync(&result, d_result, sizeof(T),
                                     cudaMemcpyDeviceToHost, p.stream));
    AGAMA_CUDA_CHECK(cudaStreamSynchronize(p.stream));
    return result;
}
#endif

// ===========================================================================
// device_array<T> — RAII wrapper around device memory (CUDA) or std::vector
// (CPU). Same surface on both: data(), size(), from_host(), to_host(), resize().
// Move-only on CUDA; copyable on CPU (since it's a std::vector underneath).
// ===========================================================================

#ifdef HAVE_CUDA

template<class T>
class device_array {
    T* d_ = nullptr;
    std::size_t n_ = 0;
public:
    device_array() = default;
    explicit device_array(std::size_t n) : n_(n) {
        if (n > 0) AGAMA_CUDA_CHECK(cudaMalloc(&d_, n * sizeof(T)));
    }
    ~device_array() {
        if (d_) cudaFree(d_);  // dtor: do not throw
    }
    device_array(device_array&& o) noexcept : d_(o.d_), n_(o.n_) {
        o.d_ = nullptr; o.n_ = 0;
    }
    device_array& operator=(device_array&& o) noexcept {
        if (this != &o) {
            if (d_) cudaFree(d_);
            d_ = o.d_; n_ = o.n_;
            o.d_ = nullptr; o.n_ = 0;
        }
        return *this;
    }
    device_array(const device_array&) = delete;
    device_array& operator=(const device_array&) = delete;

    void resize(std::size_t n) {
        if (n == n_) return;
        if (d_) { cudaFree(d_); d_ = nullptr; n_ = 0; }
        if (n > 0) {
            AGAMA_CUDA_CHECK(cudaMalloc(&d_, n * sizeof(T)));
            n_ = n;
        }
    }
    /** Ensure the device buffer holds at least `n` elements, never shrink.
        Idempotent when n <= current size: pointer + capacity unchanged.
        NOTE: growth is free-then-realloc, NOT in-place -- existing contents
        are DISCARDED whenever the buffer actually grows. That is fine for the
        intended use (persistent scratch buffers that are fully overwritten by
        from_host / a kernel before every read), but do not use reserve() on a
        buffer whose contents must survive the growth. */
    void reserve(std::size_t n) {
        if (n <= n_) return;
        // Exception safety: null out d_/n_ BEFORE the malloc (which may throw,
        // e.g. GPU OOM), so a failed growth leaves the array validly empty
        // instead of dangling at the freed old buffer (mirrors resize()).
        if (d_) { cudaFree(d_); d_ = nullptr; n_ = 0; }
        AGAMA_CUDA_CHECK(cudaMalloc(&d_, n * sizeof(T)));
        n_ = n;
    }
    void from_host(const T* h, std::size_t n) {
        AGAMA_CUDA_CHECK(cudaMemcpy(d_, h, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void to_host(T* h, std::size_t n) const {
        AGAMA_CUDA_CHECK(cudaMemcpy(h, d_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    T*       data()       { return d_; }
    const T* data() const { return d_; }
    std::size_t size() const { return n_; }
    bool   empty() const { return n_ == 0; }
};

#else  // !HAVE_CUDA — CPU fallback with matching surface.

template<class T>
class device_array {
    std::vector<T> v_;
public:
    device_array() = default;
    explicit device_array(std::size_t n) : v_(n) {}
    void resize(std::size_t n) { v_.resize(n); }
    void reserve(std::size_t n) { if (n > v_.size()) v_.resize(n); }  // CPU fallback: grow-only
    void from_host(const T* h, std::size_t n) {
        if (n > v_.size()) v_.resize(n);
        std::copy(h, h + n, v_.begin());
    }
    void to_host(T* h, std::size_t n) const {
        std::copy(v_.begin(), v_.begin() + n, h);
    }
    T*       data()       { return v_.data(); }
    const T* data() const { return v_.data(); }
    std::size_t size() const { return v_.size(); }
    bool   empty() const { return v_.empty(); }
};

#endif  // HAVE_CUDA

}  // namespace agama