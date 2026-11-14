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
  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto output = reinterpret_cast<float*>(arg->Y_addr);
  for (uint32_t q = blockIdx.x; q < outputs; q += gridDim.x) {
    const uint32_t row = (q / columns) * kRows;
    const uint32_t col = (q % columns) * kCols;
    ctx::fragment_acc frag;
    ctx::fill_fragment(frag, 0);
    if (rank == 0) {
      load_tile(smem, cta, row, col, 0);
    }
    for (uint32_t i = 0; i < k_tiles; ++i) {
      const uint32_t slot = i & 1;
      if (rank == 0 && i + 1 < k_tiles) {
        const uint32_t next_slot = slot ^ 1;
        load_tile(smem + next_slot * kInputElements, (next_slot << 8) | cta,
                  row, col, (i + 1) * ctx::tileK);
      }
      const uint32_t bar = (slot << 8) | cta;
      vx_barrier(bar, kComputeWarps);
      compute_tile(frag, smem + slot * kInputElements, rank);
      if (i + 2 < k_tiles) {
        vx_barrier(bar, kComputeWarps);
      }
    }
    store_fused(frag, output, row, col, rank, arg->N);
    if (q + gridDim.x < outputs) {
      // Every compute warp must finish reading the old tile before reuse.
      vx_barrier((7u << 8) | cta, kComputeWarps);
    }
  }
}

template <bool SeparateActivation>
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
  } else if (SeparateActivation && rank == kComputeWarps + 1) {
    uint32_t ordinal = 0;
    for (uint32_t q = first; q < outputs; q += stride, ++ordinal) {
      const uint32_t row = (q / columns) * kRows;
      const uint32_t col = (q % columns) * kCols;
      const uint32_t slot = ordinal & (arg->c_depth - 1);
      vx_barrier(((slot + 4) << 8) | cta, kComputeWarps + 1);
      const float* tile = c_queue + slot * kOutputElements;
      for (uint32_t i = lane; i < kOutputElements; i += VX_CFG_NUM_THREADS) {
        output[(row + i / kCols) * arg->N + col + i % kCols] = __builtin_fmaxf(tile[i], 0.0f);
      }
      vx_barrier_arrive(((slot + 6) << 8) | cta, kComputeWarps + 1);
    }
  } else {
    uint32_t step = 0;
    uint32_t ordinal = 0;
    for (uint32_t q = first; q < outputs; q += stride, ++ordinal) {
      const uint32_t row = (q / columns) * kRows;
      const uint32_t col = (q % columns) * kCols;
      ctx::fragment_acc frag;
      ctx::fill_fragment(frag, 0);
      for (uint32_t i = 0; i < k_tiles; ++i, ++step) {
        const uint32_t slot = step & 1;
        vx_barrier((slot << 8) | cta, kComputeWarps + 1);
        compute_tile(frag, smem + slot * kInputElements, rank);
        vx_barrier_arrive(((slot + 2) << 8) | cta, kComputeWarps + 1);
      }
      if constexpr (SeparateActivation) {
        const uint32_t slot = ordinal & (arg->c_depth - 1);
        if (ordinal >= arg->c_depth) {
          vx_barrier(((slot + 6) << 8) | cta, kComputeWarps + 1);
        }
        ctx::store_matrix_sync(c_queue + slot * kOutputElements + rank * ctx::xtileM * kCols,
                               frag, kCols);
        // BAR arrive drains LSU, publishing all shared-memory stores.
        vx_barrier_arrive(((slot + 4) << 8) | cta, kComputeWarps + 1);
      } else {
        store_fused(frag, output, row, col, rank, arg->N);
      }
    }
    if constexpr (SeparateActivation) {
      const uint32_t tail = count > arg->c_depth ? count - arg->c_depth : 0;
      for (uint32_t i = tail; i < count; ++i) {
        const uint32_t slot = i & (arg->c_depth - 1);
        vx_barrier(((slot + 6) << 8) | cta, kComputeWarps + 1);
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
