#include "common.h"
#include <vortex.h>
#include <dxa.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unistd.h>

vx_device_h device = nullptr;
vx_queue_h queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h trace_buffer = nullptr;

void cleanup() {
  if (src_buffer) { vx_buffer_release(src_buffer); }
  if (dst_buffer) { vx_buffer_release(dst_buffer); }
  if (trace_buffer) { vx_buffer_release(trace_buffer); }
  if (kernel) { vx_kernel_release(kernel); }
  if (module_) { vx_module_release(module_); }
  if (queue) { vx_queue_release(queue); }
  if (device) {
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

#define RT_CHECK(expr) do { \
  int status = (expr); \
  if (status) { \
    std::cerr << #expr << " failed: " << status << std::endl; \
    cleanup(); \
    return 1; \
  } \
} while (false)

int main(int argc, char** argv) {
  kernel_arg_t arg = {};
  arg.tiles = 9;
  arg.depth = 2;
  arg.pipeline = 1;
  bool tracing = false;
  int opt;
  while ((opt = getopt(argc, argv, "n:d:w:p:t")) != -1) {
    switch (opt) {
    case 'n': arg.tiles = std::strtoul(optarg, nullptr, 0); break;
    case 'd': arg.depth = std::strtoul(optarg, nullptr, 0); break;
    case 'w': arg.work = std::strtoul(optarg, nullptr, 0); break;
    case 'p': arg.pipeline = std::strtoul(optarg, nullptr, 0); break;
    case 't': tracing = true; break;
    default: return 1;
    }
  }
  if (!arg.tiles || arg.tiles > 4096 || (arg.depth != 1 && arg.depth != 2)
      || arg.pipeline > 1 || arg.work > 4096) {
    return 1;
  }
  const uint32_t elements = arg.tiles * kTileElements;
  const uint32_t bytes = elements * sizeof(uint32_t);
  std::vector<uint32_t> input(elements);
  std::vector<uint32_t> output(elements, 0);
  std::vector<uint32_t> timestamps(3 * arg.tiles, 0);
  for (uint32_t i = 0; i < elements; ++i) {
    input[i] = i * 2654435761u + 17u;
  }
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  RT_CHECK(vx_check_occupancy(device, 2 * VX_CFG_NUM_THREADS,
                            arg.depth * kTileElements * sizeof(uint32_t)));
  RT_CHECK(vx_buffer_create(device, bytes, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_create(device, bytes, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &arg.src_addr));
  RT_CHECK(vx_buffer_address(dst_buffer, &arg.dst_addr));
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, input.data(), bytes, 0, nullptr, nullptr));
  if (tracing) {
    RT_CHECK(vx_buffer_create(device, timestamps.size() * sizeof(uint32_t), VX_MEM_WRITE, &trace_buffer));
    RT_CHECK(vx_buffer_address(trace_buffer, &arg.trace_addr));
  }
  RT_CHECK(vortex::dxa::program_1d(device, 0, arg.src_addr, elements, kTileElements, sizeof(uint32_t)));
  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = 1;
  li.block_dim[0] = 2 * VX_CFG_NUM_THREADS;
  li.lmem_size = arg.depth * kTileElements * sizeof(uint32_t);
  vx_event_h launch = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch));
  vx_event_h read = nullptr;
  RT_CHECK(vx_enqueue_read(queue, output.data(), dst_buffer, 0, bytes, 1, &launch, &read));
  RT_CHECK(vx_event_wait_value(read, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read);
  if (tracing) {
    RT_CHECK(vx_enqueue_read(queue, timestamps.data(), trace_buffer, 0,
                            timestamps.size() * sizeof(uint32_t), 1, &launch, &read));
    RT_CHECK(vx_event_wait_value(read, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read);
    for (uint32_t i = 0; i < arg.tiles; ++i) {
      std::cout << "TILE " << i << " issue=" << timestamps[3 * i]
                << " begin=" << timestamps[3 * i + 1]
                << " end=" << timestamps[3 * i + 2] << std::endl;
    }
  }
  vx_event_release(launch);
  uint32_t errors = 0;
  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t expected = input[i];
    for (uint32_t k = 0; k < arg.work; ++k) {
      expected = expected * 1664525u + 1013904223u;
    }
    if (output[i] != expected) {
      if (errors < 8) {
        std::cout << "Mismatch " << i << ": " << output[i] << " != " << expected << std::endl;
      }
      ++errors;
    }
  }
  std::cout << "tiles=" << arg.tiles << " depth=" << arg.depth
            << " work=" << arg.work << " pipeline=" << arg.pipeline
            << " errors=" << errors << std::endl;
  cleanup();
  std::cout << (errors ? "FAILED!" : "PASSED!") << std::endl;
  return errors ? 1 : 0;
}
