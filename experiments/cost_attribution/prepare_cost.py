#!/usr/bin/env python3
"""Prepare isolated kernel controls and timing-neutral SimX observation sources."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import SOURCE, WORK, result_path
import shutil

HERE=Path(__file__).resolve().parent
OVERLAY = WORK / 'cost-attribution-observer' / 'simx'
OLD=HERE.parent/'activation_balance/activation_balance'
APP=HERE/'cost_pipeline'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Unexpected source: '+old[:90])
    return text.replace(old,new)


def prepare_app():
    ref=json.loads(result_path('activation_w16_matrix_results.json').read_text())
    assert all(sha(OLD/n)==h for n,h in ref['source_hashes'].items())
    APP.mkdir(exist_ok=True)
    kernel=(OLD/'kernel.cpp').read_text()
    kernel=once(kernel,'bool ProducerRelu = false>','bool ProducerRelu = false, bool SerialOutput = false, bool Roundtrip = false>')
    kernel=once(kernel,'      ctx::fill_fragment(frag, 0);','''      ctx::fill_fragment(frag, 0);
      if constexpr (SeparateActivation && SerialOutput) {
        if (ordinal >= arg->c_depth) {
          vx_barrier((6 << 8) | cta, kComputeWarps + ActivationWarps);
        }
      }''')
    kernel=once(kernel,'        if (ordinal >= arg->c_depth) {\n          vx_barrier(((slot + 6)',
        '        if (!SerialOutput && ordinal >= arg->c_depth) {\n          vx_barrier(((slot + 6)')
    kernel=once(kernel,'''      } else {
        store_fused(frag, output, row, col, rank, arg->N);''','''      } else {
        if constexpr (Roundtrip) {
          auto ptr = c_queue + rank * ctx::xtileM * kCols;
          ctx::store_matrix_sync(ptr, frag, kCols);
          vx_barrier((4 << 8) | cta, kComputeWarps);
          // WGMMA stores n-major fragments; WMMA's accumulator loader is m-major.
          // Volatile reads keep the intended memory roundtrip observable.
          const volatile float* reload = ptr;
          const uint32_t base_row = lane / ctx::tcN;
          const uint32_t base_col = lane % ctx::tcN;
          vt::detail::unroll_for<ctx::fragment_acc::NR>([&](auto r) {
            const uint32_t row_off = (r % ctx::m_steps) * ctx::tcM;
            const uint32_t col_off = (r / ctx::m_steps) * ctx::tcN;
            frag.data[r] = reload[(base_row + row_off) * kCols + base_col + col_off];
          });
        }
        store_fused(frag, output, row, col, rank, arg->N);''')
    kernel+='''
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
'''
    (APP/'kernel.cpp').write_text(kernel)
    host=(OLD/'main.cpp').read_text()
    host=once(host,'  int opt;','  uint32_t reserve_output = 0;\n  int opt;')
    host=once(host,'"m:n:k:v:s:a:q:g:w:u:e:"','"m:n:k:v:s:a:q:g:w:u:e:r:"')
    host=once(host,"    case 'a':","    case 'r': reserve_output = std::strtoul(optarg, nullptr, 0); break;\n    case 'a':")
    host=once(host,'mode > 4','mode > 9')
    host=once(host,'  CHECK(vx_device_open(0, &device));','''  if ((mode == 8 && arg.c_depth != 1) || reserve_output > 1) { return 1; }
  CHECK(vx_device_open(0, &device));''')
    host=once(host,'const uint32_t active_warps = width + (mode == 4 ? 1 + activation_warps : mode - 2);',
        '''const uint32_t active_warps = (mode == 4 || mode == 8) ? width + 1 + activation_warps
      : (mode >= 5 && mode <= 7) ? mode : mode == 9 ? width + 1 : width + mode - 2;''')
    host=once(host,'  const std::string entry = mode == 4','  const std::string entry = mode == 8 ? "gemm_serial" + std::to_string(activation_warps)\n      : mode == 9 ? "gemm_roundtrip"\n      : (mode >= 5 && mode <= 7) ? "gemm_join" + std::to_string(mode)\n      : mode == 4')
    host=once(host,'+ (mode == 4 ? arg.c_depth * cta_m * cfg::xtileN * 4 : 0);',
        '+ ((mode == 4 || mode == 8 || mode == 9 || reserve_output) ? arg.c_depth * cta_m * cfg::xtileN * 4 : 0);')
    host=once(host,'  CHECK(vx_check_occupancy(device, li.block_dim[0], li.lmem_size));','''  std::cout << "COST_MAP A=" << arg.A_addr << " B=" << arg.B_addr << " Y=" << arg.Y_addr
            << " A_bytes=" << bytes[0] << " B_bytes=" << bytes[1] << " Y_bytes=" << bytes[3] << std::endl;
  CHECK(vx_check_occupancy(device, li.block_dim[0], li.lmem_size));''')
    host=once(host,'  Counters activation_counts;','''  const uint32_t counters[] = {VX_CSR_MPM_SCHED_IDLE, VX_CSR_MPM_ACTIVE_WARPS, VX_CSR_MPM_STALLED_WARPS,
    VX_CSR_MPM_ISSUED_WARPS, VX_CSR_MPM_STALL_SCRB, VX_CSR_MPM_STALL_OPDS,
    VX_CSR_MPM_STALL_ALU, VX_CSR_MPM_STALL_FPU, VX_CSR_MPM_STALL_LSU, VX_CSR_MPM_STALL_SFU,
    VX_CSR_MPM_STALL_TCU, VX_CSR_MPM_INSTR_ALU, VX_CSR_MPM_INSTR_FPU, VX_CSR_MPM_INSTR_LSU,
    VX_CSR_MPM_INSTR_SFU, VX_CSR_MPM_INSTR_TCU, VX_CSR_MPM_LOADS, VX_CSR_MPM_STORES};
  for (auto csr : counters) {
    uint64_t value;
    CHECK(vx_device_mpm_query(device, VX_DCR_MPM_CLASS_CORE, csr, 0, &value));
    std::cout << "COST_COUNTER csr=" << csr << " value=" << value << std::endl;
  }
  Counters activation_counts;''')
    (APP/'main.cpp').write_text(host)
    (APP/'Makefile').write_text((OLD/'Makefile').read_text().replace('PROJECT := activation_balance','PROJECT := cost_pipeline'))
    for n in ('common.h','relu.cpp'):shutil.copyfile(OLD/n,APP/n)
    return {p.name:sha(p) for p in APP.iterdir() if p.is_file()}


def prepare_observer():
    target=OVERLAY.parent
    if target.exists() and not (target/'.project20-owned').exists():raise RuntimeError('Unowned overlay directory')
    target.mkdir(parents=True,exist_ok=True);(target/'.project20-owned').write_text('timing-neutral-observer\n')
    shutil.copytree(SOURCE/'sim/simx',OVERLAY,dirs_exist_ok=True)
    changed={}
    def patch(n,fn):
        p=OVERLAY/n;original=sha(p);p.write_text(fn(p.read_text()));changed[n]=dict(original=original,observed=sha(p))
    def sfu(s):
        return once(s,'''if (!core_->lsu_drained())
						continue; // drain LSU before the barrier''','''if (!core_->lsu_drained()) {
            DTH(3, "COST_GATE block wid=" << trace->wid << ", uuid=" << trace->uuid << std::endl);
						continue; // drain LSU before the barrier
          }
          DTH(3, "COST_GATE pass wid=" << trace->wid << ", uuid=" << trace->uuid << std::endl);''')
    patch('sfu_unit.cpp',sfu)
    def wctl(s):
        return once(s,'    if (trace->eop) {\n      if (is_tx_expect) {','''    if (trace->eop) {
      DTH(3, "COST_OP wid=" << trace->wid << ", uuid=" << trace->uuid << ", bar=" << bar_id
          << ", kind=" << (is_tx_expect ? 0 : is_sync_bar ? 1 : wctlArgs.is_bar_arrive ? 2 : 3)
          << ", arg=" << arg2 << std::endl);
      if (is_tx_expect) {''')
    patch('wctl_unit.cpp',wctl)
    def barrier(s):
        s=once(s,'  if (is_global) {\n    barrier.count = count;','''  DTH(3, "COST_ARR bar=" << bar_id << ", wid=" << wid << ", sync=" << is_sync_bar
      << ", target=" << count << ", count=" << barrier.count << ", events=" << barrier.events << ", phase=" << barrier.phase << std::endl);
  if (is_global) {
    barrier.count = count;''')
        s=s.replace('scheduler_->resume(i);','DTH(3, "COST_RESUME bar=" << bar_id << ", wid=" << i << std::endl);\n          scheduler_->resume(i);')
        s=once(s,'  return wait;','''  DTH(3, "COST_WAIT bar=" << bar_id << ", wid=" << wid << ", wait=" << wait << std::endl);
  return wait;''')
        s=once(s,'    barrier.count = barrier_count_p1 % count;','''    barrier.count = barrier_count_p1 % count;
    DTH(3, "COST_STATE bar=" << bar_id << ", count=" << barrier.count << ", events=" << barrier.events << ", phase=" << barrier.phase << std::endl);''')
        s=once(s,'  barrier.events += count;','''  barrier.events += count;
  DTH(3, "COST_STATE bar=" << bar_id << ", count=" << barrier.count << ", events=" << barrier.events << ", phase=" << barrier.phase << std::endl);''')
        s=once(s,'  --barrier.events;','''  --barrier.events;
  DTH(3, "COST_STATE bar=" << bar_id << ", count=" << barrier.count << ", events=" << barrier.events << ", phase=" << barrier.phase << std::endl);''')
        return s
    patch('barrier_unit.cpp',barrier)
    def sched(s):
        return once(s,'  active_warps_.set(wid);','''  active_warps_.set(wid);
  DTH(3, "COST_DISPATCH wid=" << wid << ", slot=" << warp.cta_csrs.cta_id
      << ", rank=" << warp.cta_csrs.cta_rank << ", size=" << warp.cta_csrs.cta_size
      << ", block=" << warp.cta_csrs.block_idx[0] << ", lmem=" << warp.cta_csrs.lmem_addr << std::endl);''')
    patch('scheduler.cpp',sched)
    def dispatcher(s):
        return once(s,'  wid_to_slot_[wid] = num_warps_;  // clear assignment','''  DTH(3, "COST_EXIT wid=" << wid << ", slot=" << slot << std::endl);
  wid_to_slot_[wid] = num_warps_;  // clear assignment''')
    patch('cta_dispatcher.cpp',dispatcher)
    (HERE/'observer_sources.json').write_text(json.dumps(dict(overlay=str(OVERLAY),changed=changed),indent=2)+'\n')
    return changed


def prepare_capacity():
    """Separate binary: hold one extra idle context around unchanged ordinary work."""
    prepare_app()
    dest=HERE/'cost_capacity';dest.mkdir(exist_ok=True)
    kernel=(APP/'kernel.cpp').read_text()
    body=kernel[kernel.index('__kernel void gemm_base('):kernel.index('\ntemplate <bool SeparateActivation')]
    body=once(body,'__kernel void gemm_base(', 'inline __attribute__((always_inline)) void capacity_base(')
    kernel+='\n'+body+'''
template <uint32_t Warps>
inline __attribute__((always_inline)) void capacity_join(kernel_arg_t* __UNIFORM__ arg) {
  if (get_sub_group_id() < kComputeWarps) { capacity_base(arg); }
  vx_barrier((7 << 8) | get_local_group_id(), Warps);
}
__kernel void gemm_capacity4(kernel_arg_t* __UNIFORM__ arg) { capacity_join<4>(arg); }
__kernel void gemm_capacity5(kernel_arg_t* __UNIFORM__ arg) { capacity_join<5>(arg); }
'''
    (dest/'kernel.cpp').write_text(kernel)
    host=(APP/'main.cpp').read_text()
    host=once(host,'mode > 9','mode > 11')
    host=once(host,'const uint32_t active_warps = (mode == 4',
        'const uint32_t active_warps = mode >= 10 ? mode - 6 : (mode == 4')
    host=once(host,'const std::string entry = mode == 8',
        'const std::string entry = mode >= 10 ? "gemm_capacity" + std::to_string(mode - 6) : mode == 8')
    (dest/'main.cpp').write_text(host)
    for n in ('common.h','relu.cpp'):shutil.copyfile(APP/n,dest/n)
    (dest/'Makefile').write_text((APP/'Makefile').read_text().replace('PROJECT := cost_pipeline','PROJECT := cost_capacity'))
    return {p.name:sha(p) for p in dest.iterdir() if p.is_file()},dest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--observer',action='store_true');a=p.parse_args()
    print(json.dumps(prepare_observer() if a.observer else prepare_app(),indent=2))
