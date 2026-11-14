#pragma once
#include <stdint.h>

struct kernel_arg_t {
  uint32_t tiles;
  uint32_t depth;
  uint32_t work;
  uint32_t pipeline;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint64_t trace_addr;
};

constexpr uint32_t kTileElements = 64;
