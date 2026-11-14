#!/usr/bin/env python3
"""Create an independent activation-role experiment from the frozen persistent app."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'persistent_pipeline/persistent_pipeline'
APP = HERE / 'activation_balance'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError('Frozen source mismatch: ' + before)
    return text.replace(before, after)


def prepare():
    reference = json.loads((HERE.parents[1] / 'persistent_w16_matrix_results.json').read_text())['source_hashes']
    if any(hashlib.sha256((OLD / n).read_bytes()).hexdigest() != h for n, h in reference.items()):
        raise RuntimeError('The previous validated app changed')
    APP.mkdir(exist_ok=True)
    kernel = (OLD / 'kernel.cpp').read_text()
    kernel = replace_once(kernel, 'template <bool SeparateActivation>',
        'template <bool SeparateActivation, uint32_t ActivationWarps = 1, uint32_t Unroll = 1, bool ProducerRelu = false>')
    kernel = kernel.replace('SeparateActivation && rank == kComputeWarps + 1',
                            'SeparateActivation && rank > kComputeWarps')
    kernel = replace_once(kernel, '''      for (uint32_t i = lane; i < kOutputElements; i += VX_CFG_NUM_THREADS) {
        output[(row + i / kCols) * arg->N + col + i % kCols] = __builtin_fmaxf(tile[i], 0.0f);
      }''', '''      constexpr uint32_t step = ActivationWarps * VX_CFG_NUM_THREADS;
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
      }''')
    # The input handshake always includes four compute warps plus one loader.
    kernel = kernel.replace('((slot + 4) << 8) | cta, kComputeWarps + 1',
                            '((slot + 4) << 8) | cta, kComputeWarps + ActivationWarps')
    kernel = kernel.replace('((slot + 6) << 8) | cta, kComputeWarps + 1',
                            '((slot + 6) << 8) | cta, kComputeWarps + ActivationWarps')
    kernel = replace_once(kernel, '''        ctx::store_matrix_sync(c_queue''', '''        if constexpr (ProducerRelu) {
#pragma unroll
          for (uint32_t i = 0; i < ctx::fragment_acc::NR; ++i) {
            frag.data[i] = __builtin_fmaxf(frag.data[i], 0.0f);
          }
        }
        ctx::store_matrix_sync(c_queue''')
    for aw in (1, 2):
        for unroll in (1, 4):
            for producer in (0, 1):
                kernel += f'''\n__kernel void gemm_three_a{aw}_u{unroll}_p{producer}(kernel_arg_t* __UNIFORM__ arg) {{
  pipeline<true, {aw}, {unroll}, {'true' if producer else 'false'}>(arg);
}}
'''
    (APP / 'kernel.cpp').write_text(kernel)
    host = (OLD / 'main.cpp').read_text()
    host = replace_once(host, '  int opt;',
        '  uint32_t activation_warps = 1, activation_unroll = 1, producer_relu = 0;\n  int opt;')
    host = replace_once(host, '"m:n:k:v:s:a:q:g:"', '"m:n:k:v:s:a:q:g:w:u:e:"')
    host = replace_once(host, "    case 'a':", "    case 'w': activation_warps = std::strtoul(optarg, nullptr, 0); break;\n"
        "    case 'u': activation_unroll = std::strtoul(optarg, nullptr, 0); break;\n"
        "    case 'e': producer_relu = std::strtoul(optarg, nullptr, 0); break;\n    case 'a':")
    host = replace_once(host, '  CHECK(vx_device_open(0, &device));',
        '''  if ((activation_warps != 1 && activation_warps != 2)
      || (activation_unroll != 1 && activation_unroll != 4) || producer_relu > 1) { return 1; }
  CHECK(vx_device_open(0, &device));''')
    host = replace_once(host, 'const uint32_t active_warps = width + mode - 2;',
        'const uint32_t active_warps = width + (mode == 4 ? 1 + activation_warps : mode - 2);')
    host = replace_once(host,
        '  CHECK(vx_module_get_kernel(module_, mode == 4 ? "gemm_three" : (mode == 3 ? "gemm_pc" : "gemm_base"), &gemm));',
        '''  const std::string entry = mode == 4
      ? "gemm_three_a" + std::to_string(activation_warps) + "_u" + std::to_string(activation_unroll)
          + "_p" + std::to_string(producer_relu)
      : (mode == 3 ? "gemm_pc" : "gemm_base");
  std::cout << "ACTIVATION warps=" << (mode == 4 ? activation_warps : 0)
            << " unroll=" << activation_unroll << " producer_relu=" << producer_relu
            << " entry=" << entry << std::endl;
  CHECK(vx_module_get_kernel(module_, entry.c_str(), &gemm));''')
    host = host.replace('#include <iostream>', '#include <iostream>\n#include <string>')
    (APP / 'main.cpp').write_text(host)
    (APP / 'Makefile').write_text((OLD / 'Makefile').read_text().replace('PROJECT := persistent_pipeline', 'PROJECT := activation_balance'))
    for name in ('common.h', 'relu.cpp'):
        (APP / name).write_bytes((OLD / name).read_bytes())
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in APP.iterdir() if p.is_file()}


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
