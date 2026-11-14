#include "common.h"
#include <vx_spawn2.h>

__kernel void relu_k(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < arg->M * arg->N) {
    auto src = reinterpret_cast<const float*>(arg->C_addr);
    auto dst = reinterpret_cast<float*>(arg->Y_addr);
    dst[i] = __builtin_fmaxf(src[i], 0.0f);
  }
}

__kernel void relu_grid_stride(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  auto src = reinterpret_cast<const float*>(arg->C_addr);
  auto dst = reinterpret_cast<float*>(arg->Y_addr);
  for (uint32_t i = first; i < arg->M * arg->N; i += stride) {
    dst[i] = __builtin_fmaxf(src[i], 0.0f);
  }
}
