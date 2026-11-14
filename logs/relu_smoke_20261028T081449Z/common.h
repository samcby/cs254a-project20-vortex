#pragma once
#include <stdint.h>

struct kernel_arg_t {
  uint32_t M, N, K;
  uint32_t depth;
  uint32_t fuse;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t Y_addr;
};
