// gpu_device.h — host/device annotation macros only.
//
// This is the minimum surface needed by headers that just want to tag a leaf
// function as device-callable without pulling in <cuda_runtime.h>, the Cuda
// policy struct, device_array, or the forall/reduce kernel launches (all of
// which live in gpu_policy.h). Public so it can be included from low-level
// headers like math_base.h without inflating their compile time.
//
// Under nvcc (__CUDACC__): AGAMA_DEVICE expands to __host__ __device__ and
// AGAMA_DEVICE_INLINE to __host__ __device__ __forceinline__.
// Under any other compiler: AGAMA_DEVICE is empty and AGAMA_DEVICE_INLINE
// is plain `inline`. So a function tagged AGAMA_DEVICE_INLINE compiles
// unchanged on the CPU path.

#pragma once

#ifdef __CUDACC__
  #define AGAMA_DEVICE        __host__ __device__
  #define AGAMA_DEVICE_INLINE __host__ __device__ __forceinline__
#else
  #define AGAMA_DEVICE
  #define AGAMA_DEVICE_INLINE inline
#endif
