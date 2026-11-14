#include "common.h"
#include <vortex2.h>
#include <VX_types.h>
#include <dxa.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <unistd.h>

namespace vt = vortex::tensor;
using cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, WGMMA_NRC>;

vx_device_h device = nullptr;
vx_queue_h queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h gemm = nullptr;
vx_kernel_h relu = nullptr;
vx_buffer_h buffers[4] = {};

void cleanup() {
  for (auto buffer : buffers) {
    if (buffer) { vx_buffer_release(buffer); }
  }
  if (gemm) { vx_kernel_release(gemm); }
  if (relu) { vx_kernel_release(relu); }
  if (module_) { vx_module_release(module_); }
  if (queue) { vx_queue_release(queue); }
  if (device) { vx_device_release(device); }
}

#define CHECK(expr) do { \
  int status = (expr); \
  if (status) { \
    std::cerr << #expr << " failed: " << status << std::endl; \
    cleanup(); \
    std::exit(1); \
  } \
} while (false)

struct Counters {
  uint64_t cycles = 0, instructions = 0, reads = 0, writes = 0;
};

Counters snapshot(const char* name, const Counters* previous = nullptr) {
  Counters value;
  CHECK(vx_device_mpm_query(device, VX_DCR_MPM_CLASS_MEM, VX_CSR_MCYCLE, 0, &value.cycles));
  CHECK(vx_device_mpm_query(device, VX_DCR_MPM_CLASS_MEM, VX_CSR_MINSTRET, 0, &value.instructions));
  CHECK(vx_device_mpm_query(device, VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_MEM_READS, 0, &value.reads));
  CHECK(vx_device_mpm_query(device, VX_DCR_MPM_CLASS_MEM, VX_CSR_MPM_MEM_WRITES, 0, &value.writes));
  std::cout << "SNAPSHOT " << name << " cumulative_cycles=" << value.cycles
            << " cumulative_instructions=" << value.instructions << std::endl;
  if (previous) {
    if (value.cycles < previous->cycles || value.instructions < previous->instructions) {
      std::cerr << "Unexpected core counter reset" << std::endl;
      cleanup(); std::exit(1);
    }
    // Core MPM counters survive launch reset; uncore memory counters do not.
    value.cycles -= previous->cycles;
    value.instructions -= previous->instructions;
  }
  std::cout << "STAGE " << name << " cycles=" << value.cycles
            << " instructions=" << value.instructions << " reads=" << value.reads
            << " writes=" << value.writes << std::endl;
  return value;
}

uint16_t encode_half(int value) {
  const float input = value / 16.0f;
  uint32_t bits;
  std::memcpy(&bits, &input, sizeof(bits));
  return rv_ftoh_s(bits, 0, nullptr);
}

int main(int argc, char** argv) {
  kernel_arg_t arg = {};
  arg.M = 64; arg.N = 32; arg.K = 64; arg.depth = 2; arg.c_depth = 1;
  uint32_t mode = 2, seed = 1, activation_stride = 0, persistent_ctas = 1;
  int opt;
  while ((opt = getopt(argc, argv, "m:n:k:v:s:a:q:g:")) != -1) {
    switch (opt) {
    case 'm': arg.M = std::strtoul(optarg, nullptr, 0); break;
    case 'n': arg.N = std::strtoul(optarg, nullptr, 0); break;
    case 'k': arg.K = std::strtoul(optarg, nullptr, 0); break;
    case 'v': mode = std::strtoul(optarg, nullptr, 0); break;
    case 's': seed = std::strtoul(optarg, nullptr, 0); break;
    case 'q': arg.c_depth = std::strtoul(optarg, nullptr, 0); break;
    case 'g': persistent_ctas = std::strtoul(optarg, nullptr, 0); break;
    case 'a': activation_stride = std::strtoul(optarg, nullptr, 0); break;
    default: return 1;
    }
  }
  if (!arg.M || !arg.N || !arg.K || (mode < 2 || mode > 4 || !persistent_ctas || (arg.c_depth != 1 && arg.c_depth != 2)) || activation_stride > 1
      || arg.M > 512 || arg.N > 512 || arg.K > 256) {
    return 1;
  }
  CHECK(vx_device_open(0, &device));
  uint64_t threads, warps, width;
  CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &threads));
  CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &warps));
  CHECK(vx_device_query(device, VX_CAPS_ISSUE_WIDTH, &width));
  const uint32_t cta_m = width * cfg::xtileM;
  const uint32_t active_warps = width + mode - 2;
  if (threads != VX_CFG_NUM_THREADS || width != VX_CFG_ISSUE_WIDTH || warps < active_warps
      || arg.M % cta_m || arg.N % cfg::xtileN || arg.K % cfg::tileK) {
    cleanup(); return 1;
  }
  if (persistent_ctas > (arg.M / cta_m) * (arg.N / cfg::xtileN)) { cleanup(); return 1; }
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  CHECK(vx_queue_create(device, &qi, &queue));
  arg.fuse = mode >= 2;
  const uint32_t size_a = arg.M * arg.K, size_b = arg.K * arg.N, size_c = arg.M * arg.N;
  std::vector<int16_t> a(size_a), b(size_b);
  std::vector<uint16_t> a_half(size_a), b_half(size_b);
  std::vector<float> raw(size_c), expected(size_c), output(size_c), intermediate(size_c);
  uint32_t rng = seed;
  for (uint32_t i = 0; i < size_a; ++i) {
    rng = rng * 1664525u + 1013904223u;
    a[i] = i < arg.K ? 0 : static_cast<int>(rng % 33) - 16;
    a_half[i] = encode_half(a[i]);
  }
  for (uint32_t i = 0; i < size_b; ++i) {
    rng = rng * 1664525u + 1013904223u;
    b[i] = i % arg.N == 0 ? 0 : static_cast<int>(rng % 33) - 16;
    b_half[i] = encode_half(b[i]);
  }
  uint32_t positive = 0, negative = 0, zero = 0;
  for (uint32_t r = 0; r < arg.M; ++r) {
    for (uint32_t c = 0; c < arg.N; ++c) {
      int32_t sum = 0;
      for (uint32_t k = 0; k < arg.K; ++k) {
        sum += a[r * arg.K + k] * b[k * arg.N + c];
      }
      const uint32_t i = r * arg.N + c;
      raw[i] = sum / 256.0f;
      expected[i] = mode == 0 ? raw[i] : std::fmax(raw[i], 0.0f);
      positive += sum > 0; negative += sum < 0; zero += sum == 0;
    }
  }
  if (!positive || !negative || !zero) {
    cleanup(); return 1;
  }
  std::cout << "CONFIG mode=" << mode << " M=" << arg.M << " N=" << arg.N
            << " K=" << arg.K << " seed=" << seed << " hardware_warps=" << warps
            << " compute_warps=" << width << " active_CTA_warps=" << active_warps
            << " activation_stride=" << activation_stride << std::endl;
  std::cout << "COVERAGE positive=" << positive << " negative=" << negative << " zero=" << zero << std::endl;
  const uint32_t bytes[4] = {size_a * 2, size_b * 2, size_c * 4, size_c * 4};
  uint64_t addresses[4];
  for (uint32_t i = 0; i < 4; ++i) {
    CHECK(vx_buffer_create(device, bytes[i], VX_MEM_READ | VX_MEM_WRITE, &buffers[i]));
    CHECK(vx_buffer_address(buffers[i], &addresses[i]));
  }
  arg.A_addr = addresses[0]; arg.B_addr = addresses[1];
  arg.C_addr = mode == 1 ? addresses[2] : addresses[3]; arg.Y_addr = addresses[3];
  CHECK(vx_enqueue_write(queue, buffers[0], 0, a_half.data(), bytes[0], 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(queue, buffers[1], 0, b_half.data(), bytes[1], 0, nullptr, nullptr));
  std::vector<uint32_t> sentinel(size_c, 0x7fc00000u);
  CHECK(vx_enqueue_write(queue, buffers[2], 0, sentinel.data(), bytes[2], 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(queue, buffers[3], 0, sentinel.data(), bytes[3], 0, nullptr, nullptr));
  CHECK(vortex::dxa::program_2d(device, 0, arg.A_addr, arg.K, arg.M,
                               arg.K * 2, cfg::tileK, cta_m, 2));
  CHECK(vortex::dxa::program_2d(device, 1, arg.B_addr, arg.N, arg.K,
                               arg.N * 2, cfg::xtileN, cfg::tileK, 2));
  CHECK(vortex::dxa::set_layout(device, 1, vortex::dxa::Layout::BlockMajor, 2, 2));
  CHECK(vortex::dxa::set_tile_geometry(device, 1, cfg::tcN, cfg::b_blk_k));
  CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  CHECK(vx_module_get_kernel(module_, mode == 4 ? "gemm_three" : (mode == 3 ? "gemm_pc" : "gemm_base"), &gemm));
  CHECK(vx_module_get_kernel(module_, activation_stride ? "relu_grid_stride" : "relu_k", &relu));
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li); li.kernel = gemm;
  li.args_host = &arg; li.args_size = sizeof(arg); li.ndim = 1;
  li.grid_dim[0] = persistent_ctas; li.grid_dim[1] = 1;
  li.block_dim[0] = active_warps * threads; li.block_dim[1] = 1;
  li.lmem_size = 2 * (cta_m * cfg::tileK + cfg::tileK * cfg::xtileN) * 2
      + (mode == 4 ? arg.c_depth * cta_m * cfg::xtileN * 4 : 0);
  std::cout << "PERSISTENT output_tiles=" << (arg.M / cta_m) * (arg.N / cfg::xtileN)
            << " ctas=" << persistent_ctas << " c_depth=" << arg.c_depth
            << " smem=" << li.lmem_size << std::endl;
  CHECK(vx_check_occupancy(device, li.block_dim[0], li.lmem_size));
  vx_event_h first = nullptr, last = nullptr;
  CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &first));
  CHECK(vx_event_wait_value(first, 1, VX_TIMEOUT_INFINITE));
  const Counters gemm_counts = snapshot("gemm");
  Counters activation_counts;
  last = first;
  if (mode == 1) {
    li.kernel = relu; li.ndim = 1; li.lmem_size = 0;
    li.block_dim[0] = 128; li.block_dim[1] = 1;
    li.grid_dim[0] = (size_c + 127) / 128; li.grid_dim[1] = 1;
    if (activation_stride) {
      li.grid_dim[0] = std::min<uint32_t>(li.grid_dim[0], warps / (128 / threads));
    }
    CHECK(vx_enqueue_launch(queue, &li, 1, &first, &last));
    CHECK(vx_event_wait_value(last, 1, VX_TIMEOUT_INFINITE));
    activation_counts = snapshot("relu", &gemm_counts);
  }
  std::cout << "CHAIN cycles=" << gemm_counts.cycles + activation_counts.cycles
            << " instructions=" << gemm_counts.instructions + activation_counts.instructions
            << " reads=" << gemm_counts.reads + activation_counts.reads
            << " writes=" << gemm_counts.writes + activation_counts.writes
            << " launches=" << (mode == 1 ? 2 : 1) << std::endl;
  vx_event_h read = nullptr;
  CHECK(vx_enqueue_read(queue, output.data(), buffers[3], 0, bytes[3], 1, &last, &read));
  CHECK(vx_event_wait_value(read, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read);
  if (mode == 1) {
    CHECK(vx_enqueue_read(queue, intermediate.data(), buffers[2], 0, bytes[2], 1, &last, &read));
    CHECK(vx_event_wait_value(read, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read);
  }
  uint32_t errors = 0;
  uint64_t hash = 14695981039346656037ull;
  for (uint32_t i = 0; i < size_c; ++i) {
    if (output[i] != expected[i] || (mode == 1 && intermediate[i] != raw[i])) {
      if (errors < 8) {
        std::cout << "Mismatch " << i << " actual=" << output[i] << " expected=" << expected[i] << std::endl;
      }
      ++errors;
    }
    uint32_t bits;
    std::memcpy(&bits, &output[i], sizeof(bits));
    hash = (hash ^ bits) * 1099511628211ull;
  }
  std::cout << "VERIFY errors=" << errors << " output_hash=" << hash << std::endl;
  if (last != first) { vx_event_release(last); }
  vx_event_release(first);
  cleanup();
  std::cout << (errors ? "FAILED!" : "PASSED!") << std::endl;
  return errors ? 1 : 0;
}
