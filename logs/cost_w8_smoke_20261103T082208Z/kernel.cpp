#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;
constexpr uint32_t kComputeWarps = VX_CFG_ISSUE_WIDTH;
constexpr uint32_t kRows = kComputeWarps * ctx::xtileM;
constexpr uint32_t kCols = ctx::xtileN;
constexpr uint32_t kAElements = kRows * ctx::tileK;
constexpr uint32_t kInputElements = kAElements + ctx::tileK * kCols;
constexpr uint32_t kOutputElements = kRows * kCols;

inline __attribute__((always_inline)) void load_tile(ctx::input_t* a, uint32_t bar, uint32_t row,
                      uint32_t col, uint32_t k) {
  vx_barrier_expect_tx(bar, 2);
  vx_dxa_issue_2d_wg(0, bar, a, k, row);
  vx_dxa_issue_2d_wg(1, bar, a + kAElements, col, k);
}

inline __attribute__((always_inline)) void compute_tile(ctx::fragment_acc& frag, ctx::input_t* a, uint32_t rank) {
  auto desc_b = vt::vx_make_smem_desc(a + kAElements, 0);
  auto a_warp = a + rank * ctx::xtileM * ctx::tileK;
#if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
  ctx::fragment_a frag_a;
  ctx::load_matrix_sync(frag_a, a_warp, ctx::tileK);
  ctx::wgmma_sync(frag, frag_a, desc_b, frag);
#else
  auto desc_a = vt::vx_make_smem_desc(a_warp, ctx::tileK * sizeof(ctx::input_t));
  ctx::wgmma_sync(frag, desc_a, desc_b, frag);
#endif
}

inline __attribute__((always_inline)) void store_fused(ctx::fragment_acc& frag, float* output, uint32_t row,
                        uint32_t col, uint32_t rank, uint32_t stride) {
#pragma unroll
  for (uint32_t i = 0; i < ctx::fragment_acc::NR; ++i) {
    frag.data[i] = __builtin_fmaxf(frag.data[i], 0.0f);
  }
  ctx::store_matrix_sync(output + (row + rank * ctx::xtileM) * stride + col, frag, stride);
}

__kernel void gemm_base(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t rank = get_sub_group_id();
  const uint32_t cta = get_local_group_id();
  const uint32_t columns = arg->N / kCols;
  const uint32_t outputs = (arg->M / kRows) * columns;
  const uint32_t k_tiles = arg->K / ctx::tileK;
  const uint32_t first = blockIdx.x;
  const uint32_t stride = gridDim.x;
  const uint32_t input_tiles = (1 + (outputs - 1 - first) / stride) * k_tiles;
  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto output = reinterpret_cast<float*>(arg->Y_addr);
  if (rank == 0) {
    load_tile(smem, cta, (first / columns) * kRows, (first % columns) * kCols, 0);
  }
  uint32_t step = 0;
  for (uint32_t q = first; q < outputs; q += stride) {
    const uint32_t row = (q / columns) * kRows;
    const uint32_t col = (q % columns) * kCols;
    ctx::fragment_acc frag;
    ctx::fill_fragment(frag, 0);
    for (uint32_t i = 0; i < k_tiles; ++i, ++step) {
      const uint32_t slot = step & 1;
      if (rank == 0 && step + 1 < input_tiles) {
        const uint32_t next_slot = slot ^ 1;
        const bool same_output = i + 1 < k_tiles;
        const uint32_t next_q = same_output ? q : q + stride;
        load_tile(smem + next_slot * kInputElements, (next_slot << 8) | cta,
                  (next_q / columns) * kRows, (next_q % columns) * kCols,
                  same_output ? (i + 1) * ctx::tileK : 0);
      }
      const uint32_t bar = (slot << 8) | cta;
      vx_barrier(bar, kComputeWarps);
      compute_tile(frag, smem + slot * kInputElements, rank);
      if (step + 2 < input_tiles) {
        vx_barrier(bar, kComputeWarps);
      }
    }
    store_fused(frag, output, row, col, rank, arg->N);
  }
}

template <bool SeparateActivation, uint32_t ActivationWarps = 1, uint32_t Unroll = 1, bool ProducerRelu = false, bool SerialOutput = false, bool Roundtrip = false>
inline __attribute__((always_inline)) void pipeline(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t rank = get_sub_group_id();
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t cta = get_local_group_id();
  const uint32_t first = blockIdx.x;
  const uint32_t stride = gridDim.x;
  const uint32_t columns = arg->N / kCols;
  const uint32_t outputs = (arg->M / kRows) * columns;
  const uint32_t count = 1 + (outputs - 1 - first) / stride;
  const uint32_t k_tiles = arg->K / ctx::tileK;
  const uint32_t input_tiles = count * k_tiles;
  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto c_queue = reinterpret_cast<float*>(smem + 2 * kInputElements);
  auto output = reinterpret_cast<float*>(arg->Y_addr);

  if (rank == kComputeWarps) {
    uint32_t step = 0;
    for (uint32_t q = first; q < outputs; q += stride) {
      const uint32_t row = (q / columns) * kRows;
      const uint32_t col = (q % columns) * kCols;
      for (uint32_t i = 0; i < k_tiles; ++i, ++step) {
        const uint32_t slot = step & 1;
        const uint32_t ready = (slot << 8) | cta;
        if (step >= 2) {
          vx_barrier(((slot + 2) << 8) | cta, kComputeWarps + 1);
        }
        load_tile(smem + slot * kInputElements, ready, row, col, i * ctx::tileK);
        vx_barrier_arrive(ready, kComputeWarps + 1);
      }
    }
    const uint32_t tail = input_tiles > 2 ? input_tiles - 2 : 0;
    for (uint32_t i = tail; i < input_tiles; ++i) {
      vx_barrier((((i & 1) + 2) << 8) | cta, kComputeWarps + 1);
    }
  } else if (SeparateActivation && rank > kComputeWarps) {
    uint32_t ordinal = 0;
    for (uint32_t q = first; q < outputs; q += stride, ++ordinal) {
      const uint32_t row = (q / columns) * kRows;
      const uint32_t col = (q % columns) * kCols;
      const uint32_t slot = ordinal & (arg->c_depth - 1);
      vx_barrier(((slot + 4) << 8) | cta, kComputeWarps + ActivationWarps);
      const float* tile = c_queue + slot * kOutputElements;
      constexpr uint32_t step = ActivationWarps * VX_CFG_NUM_THREADS;
      const uint32_t first_element = (rank - kComputeWarps - 1) * VX_CFG_NUM_THREADS + lane;
      for (uint32_t base = first_element; base < kOutputElements; base += step * Unroll) {
#pragma unroll
        for (uint32_t j = 0; j < Unroll; ++j) {
          const uint32_t i = base + j * step;
          float value = tile[i];
          if constexpr (!ProducerRelu) {
            value = __builtin_fmaxf(value, 0.0f);
          }
          output[(row + i / kCols) * arg->N + col + i % kCols] = value;
        }
      }
      vx_barrier_arrive(((slot + 6) << 8) | cta, kComputeWarps + ActivationWarps);
    }
  } else {
    uint32_t step = 0;
    uint32_t ordinal = 0;
    for (uint32_t q = first; q < outputs; q += stride, ++ordinal) {
      const uint32_t row = (q / columns) * kRows;
      const uint32_t col = (q % columns) * kCols;
      ctx::fragment_acc frag;
      ctx::fill_fragment(frag, 0);
      if constexpr (SeparateActivation && SerialOutput) {
        if (ordinal >= arg->c_depth) {
          vx_barrier((6 << 8) | cta, kComputeWarps + ActivationWarps);
        }
      }
      for (uint32_t i = 0; i < k_tiles; ++i, ++step) {
        const uint32_t slot = step & 1;
        vx_barrier((slot << 8) | cta, kComputeWarps + 1);
        compute_tile(frag, smem + slot * kInputElements, rank);
        vx_barrier_arrive(((slot + 2) << 8) | cta, kComputeWarps + 1);
      }
      if constexpr (SeparateActivation) {
        const uint32_t slot = ordinal & (arg->c_depth - 1);
        if (!SerialOutput && ordinal >= arg->c_depth) {
          vx_barrier(((slot + 6) << 8) | cta, kComputeWarps + ActivationWarps);
        }
        if constexpr (ProducerRelu) {
#pragma unroll
          for (uint32_t i = 0; i < ctx::fragment_acc::NR; ++i) {
            frag.data[i] = __builtin_fmaxf(frag.data[i], 0.0f);
          }
        }
        ctx::store_matrix_sync(c_queue + slot * kOutputElements + rank * ctx::xtileM * kCols,
                               frag, kCols);
        // BAR arrive drains LSU, publishing all shared-memory stores.
        vx_barrier_arrive(((slot + 4) << 8) | cta, kComputeWarps + ActivationWarps);
      } else {
        if constexpr (Roundtrip) {
          auto ptr = c_queue + rank * ctx::xtileM * kCols;
          ctx::store_matrix_sync(ptr, frag, kCols);
          vx_barrier((4 << 8) | cta, kComputeWarps);
          using load_ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;
          load_ctx::load_matrix_sync(frag, ptr, kCols);
        }
        store_fused(frag, output, row, col, rank, arg->N);
      }
    }
    if constexpr (SeparateActivation) {
      const uint32_t tail = count > arg->c_depth ? count - arg->c_depth : 0;
      for (uint32_t i = tail; i < count; ++i) {
        const uint32_t slot = i & (arg->c_depth - 1);
        vx_barrier(((slot + 6) << 8) | cta, kComputeWarps + ActivationWarps);
      }
    }
  }
}

__kernel void gemm_pc(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<false>(arg);
}

__kernel void gemm_three(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true>(arg);
}

__kernel void gemm_three_a1_u1_p0(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 1, 1, false>(arg);
}

__kernel void gemm_three_a1_u1_p1(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 1, 1, true>(arg);
}

__kernel void gemm_three_a1_u4_p0(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 1, 4, false>(arg);
}

__kernel void gemm_three_a1_u4_p1(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 1, 4, true>(arg);
}

__kernel void gemm_three_a2_u1_p0(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 2, 1, false>(arg);
}

__kernel void gemm_three_a2_u1_p1(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 2, 1, true>(arg);
}

__kernel void gemm_three_a2_u4_p0(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 2, 4, false>(arg);
}

__kernel void gemm_three_a2_u4_p1(kernel_arg_t* __UNIFORM__ arg) {
  pipeline<true, 2, 4, true>(arg);
}

template <uint32_t Warps>
inline __attribute__((always_inline)) void joined_two(kernel_arg_t* __UNIFORM__ arg) {
  if (get_sub_group_id() < kComputeWarps + 1) {
    pipeline<false>(arg);
  }
  // Keep idle warp contexts allocated until all active roles finish.
  vx_barrier((7 << 8) | get_local_group_id(), Warps);
}
__kernel void gemm_join5(kernel_arg_t* __UNIFORM__ arg) { joined_two<5>(arg); }
__kernel void gemm_join6(kernel_arg_t* __UNIFORM__ arg) { joined_two<6>(arg); }
__kernel void gemm_join7(kernel_arg_t* __UNIFORM__ arg) { joined_two<7>(arg); }
__kernel void gemm_serial1(kernel_arg_t* __UNIFORM__ arg) { pipeline<true,1,4,false,true>(arg); }
__kernel void gemm_serial2(kernel_arg_t* __UNIFORM__ arg) { pipeline<true,2,4,false,true>(arg); }
__kernel void gemm_roundtrip(kernel_arg_t* __UNIFORM__ arg) { pipeline<false,1,1,false,false,true>(arg); }
