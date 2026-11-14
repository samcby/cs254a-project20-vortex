#include "common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  auto trace = reinterpret_cast<uint32_t*>(arg->trace_addr);
  const uint32_t rank = get_sub_group_id();
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t cta = get_local_group_id();
  const uint32_t depth = arg->depth;

  if (arg->pipeline && rank == 0) {
    for (uint32_t i = 0; i < arg->tiles; ++i) {
      const uint32_t slot = i % depth;
      const uint32_t ready = (slot << 8) | cta;
      const uint32_t free = ((slot + depth) << 8) | cta;
      if (i >= depth) {
        vx_barrier(free, 2);
      }
      vx_barrier_expect_tx(ready, 1);
      vx_dxa_issue_1d_wg(0, ready, smem + slot * kTileElements, i * kTileElements);
      if (trace && lane == 0) {
        trace[3 * i] = static_cast<uint32_t>(vx_rdcycle());
      }
      vx_barrier_arrive(ready, 2);
    }
    // Finish every release generation before the CTA's barrier slots are reused.
    const uint32_t first = arg->tiles > depth ? arg->tiles - depth : 0;
    for (uint32_t i = first; i < arg->tiles; ++i) {
      vx_barrier((((i % depth) + depth) << 8) | cta, 2);
    }
  } else {
    for (uint32_t i = 0; i < arg->tiles; ++i) {
      const uint32_t slot = i % depth;
      const uint32_t ready = (slot << 8) | cta;
      const uint32_t free = ((slot + depth) << 8) | cta;
      if (!arg->pipeline && rank == 0) {
        vx_barrier_expect_tx(ready, 1);
        vx_dxa_issue_1d_wg(0, ready, smem + slot * kTileElements, i * kTileElements);
        if (trace && lane == 0) {
          trace[3 * i] = static_cast<uint32_t>(vx_rdcycle());
        }
      }
      vx_barrier(ready, 2);
      if (rank == 1) {
        if (trace && lane == 0) {
          trace[3 * i + 1] = static_cast<uint32_t>(vx_rdcycle());
        }
        for (uint32_t j = lane; j < kTileElements; j += VX_CFG_NUM_THREADS) {
          uint32_t value = smem[slot * kTileElements + j];
          for (uint32_t k = 0; k < arg->work; ++k) {
            value = value * 1664525u + 1013904223u;
          }
          dst[i * kTileElements + j] = value;
        }
        if (trace && lane == 0) {
          trace[3 * i + 2] = static_cast<uint32_t>(vx_rdcycle());
        }
      }
      if (arg->pipeline) {
        vx_barrier_arrive(free, 2);
      } else {
        vx_barrier(free, 2);
      }
    }
  }
}
