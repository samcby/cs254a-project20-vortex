#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t rank = get_sub_group_id();
  const uint32_t compute_warps = VX_CFG_ISSUE_WIDTH;
  const uint32_t participants = compute_warps + 1;
  const uint32_t cta_M = compute_warps * ctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_M;
  const uint32_t tile_col = blockIdx.x * ctx::xtileN;
  const uint32_t a_elements = cta_M * ctx::tileK;
  const uint32_t buffer_elements = a_elements + ctx::tileK * ctx::xtileN;
  const uint32_t cta = get_local_group_id();
  const uint32_t depth = arg->depth;
  const uint32_t tiles = arg->K / ctx::tileK;
  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());

  if (rank == compute_warps) {
    for (uint32_t i = 0; i < tiles; ++i) {
      const uint32_t slot = i % depth;
      const uint32_t ready = (slot << 8) | cta;
      const uint32_t free = ((slot + depth) << 8) | cta;
      auto a = smem + slot * buffer_elements;
      auto b = a + a_elements;
      if (i >= depth) {
        vx_barrier(free, participants);
      }
      vx_barrier_expect_tx(ready, 2);
      vx_dxa_issue_2d_wg(0, ready, a, i * ctx::tileK, tile_row);
      vx_dxa_issue_2d_wg(1, ready, b, tile_col, i * ctx::tileK);
      vx_barrier_arrive(ready, participants);
    }
    // Leave no pending release generation when the CTA retires.
    const uint32_t first = tiles > depth ? tiles - depth : 0;
    for (uint32_t i = first; i < tiles; ++i) {
      vx_barrier((((i % depth) + depth) << 8) | cta, participants);
    }
  } else {
    ctx::fragment_acc frag;
    ctx::fill_fragment(frag, 0);
    for (uint32_t i = 0; i < tiles; ++i) {
      const uint32_t slot = i % depth;
      const uint32_t ready = (slot << 8) | cta;
      const uint32_t free = ((slot + depth) << 8) | cta;
      vx_barrier(ready, participants);
      auto a = smem + slot * buffer_elements + rank * ctx::xtileM * ctx::tileK;
      auto b = smem + slot * buffer_elements + a_elements;
      auto desc_b = vt::vx_make_smem_desc(b, 0);
#if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
      ctx::fragment_a frag_a;
      ctx::load_matrix_sync(frag_a, a, ctx::tileK);
      ctx::wgmma_sync(frag, frag_a, desc_b, frag);
#else
      auto desc_a = vt::vx_make_smem_desc(a, ctx::tileK * sizeof(ctx::input_t));
      ctx::wgmma_sync(frag, desc_a, desc_b, frag);
#endif
      vx_barrier_arrive(free, participants);
    }
    auto output = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    ctx::store_matrix_sync(output + (tile_row + rank * ctx::xtileM) * arg->N + tile_col,
                           frag, arg->N);
  }
}
