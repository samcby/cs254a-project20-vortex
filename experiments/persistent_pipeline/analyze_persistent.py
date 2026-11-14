#!/usr/bin/env python3
"""Validate persistent stage results and reconstruct per-output compute/activation activity."""

import csv
import gzip
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import log_path
import re
from statistics import median

HERE = Path(__file__).resolve().parent
COURSE = HERE.parents[1]


def local(path):
    return log_path(path)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def parse_activity(case):
    ranks = {}
    compute = {rank: [] for rank in range(4)}
    activation = {rank: [] for rank in (range(4) if case['mode'] == 3 else [5])}
    transfers = 0
    dispatch = re.compile(r'dispatch CTA warp: cid=0, wid=(\d+),.*rank=(\d+)/(\d+)')
    tcu = re.compile(r'TRACE\s+(\d+): .*tcu execute: op=WGMMA,.*wid=(\d+),.*rd=f')
    fpu = re.compile(r'TRACE\s+(\d+): .*fpu execute: op=FMIN_MAX,.*wid=(\d+),')
    with gzip.open(local(case['trace']), 'rt', errors='replace') as src:
        for line in src:
            m = dispatch.search(line)
            if m:
                ranks[int(m[1])] = int(m[2])
            m = tcu.search(line)
            if m:
                compute[ranks[int(m[2])]].append(int(m[1]))
            m = fpu.search(line)
            if m:
                rank = ranks[int(m[2])]
                if rank not in activation:
                    raise RuntimeError('Unexpected activation warp')
                activation[rank].append(int(m[1]))
            if re.search(r'dxa-core\[\d+\] start:', line):
                transfers += 1
    outputs = case['tiles']
    per_warp = (case['k'] // 16) * 32
    per_activation = 16 if case['mode'] == 3 else 64
    if any(len(events) != outputs * per_warp for events in compute.values()):
        raise RuntimeError('Unexpected WGMMA event count')
    if any(len(events) != outputs * per_activation for events in activation.values()):
        raise RuntimeError('Unexpected activation event count')
    if transfers != outputs * (case['k'] // 16) * 2:
        raise RuntimeError('Unexpected DXA work count')
    groups = []
    for q in range(outputs):
        c_events = [t for events in compute.values() for t in events[q * per_warp:(q + 1) * per_warp]]
        a_events = [t for events in activation.values() for t in events[q * per_activation:(q + 1) * per_activation]]
        if case['mode'] == 4:
            if min(a_events) <= max(c_events):
                raise RuntimeError('Separate activation began before the complete output tile was computed')
        else:
            # Fused epilogues consume each warp's own completed accumulator;
            # they need not wait for the other warps' independent output rows.
            for rank in range(4):
                c_rank = compute[rank][q * per_warp:(q + 1) * per_warp]
                a_rank = activation[rank][q * per_activation:(q + 1) * per_activation]
                if min(a_rank) <= max(c_rank):
                    raise RuntimeError('Fused activation began before its warp completed K')
        groups.append(dict(output=q, compute_start=min(c_events), compute_end=max(c_events),
                           activation_start=min(a_events), activation_end=max(a_events),
                           compute_events=c_events, activation_events=a_events))
    for q, group in enumerate(groups):
        later = groups[q + 1] if q + 1 < outputs else None
        group['activation_events_during_next_compute_window'] = (
            sum(later['compute_start'] <= t <= later['compute_end'] for t in group['activation_events']) if later else 0)
        group['same_cycle_activation_and_next_compute'] = (
            len(set(group['activation_events']) & set(later['compute_events'])) if later else 0)
    rows = [{k: v for k, v in group.items() if not k.endswith('_events')} for group in groups]
    summary = dict(mode=case['mode'], output_tiles=outputs, dxa_transfers=transfers,
                   wgmma_compute_events=sum(map(len, compute.values())),
                   activation_events=sum(map(len, activation.values())),
                   activation_events_during_next_compute_window=sum(g['activation_events_during_next_compute_window'] for g in groups),
                   same_cycle_activation_and_next_compute=sum(g['same_cycle_activation_and_next_compute'] for g in groups),
                   rows=rows, trace=case['trace'])
    return summary


def main():
    specs = [(16, 'matrix'), (16, 'depth'), (8, 'depth'), (16, 'trace'), (16, 'repeat')]
    runs = {(w, p): json.loads((COURSE / f'persistent_w{w}_{p}_results.json').read_text()) for w, p in specs}
    reference = runs[(16, 'matrix')]
    cases = {}
    for key, run in runs.items():
        if run['status'] != 'complete' or run.get('ordinary_prefetch') != 'cross-output-tile':
            raise RuntimeError('Incomplete or obsolete baseline')
        if run['source_hashes'] != reference['source_hashes'] or not all(c['passed'] for c in run['cases']):
            raise RuntimeError('Source or correctness mismatch')
        expected_config = dict(reference['effective_config'], VX_CFG_NUM_WARPS=str(key[0]))
        if run['effective_config'] != expected_config:
            raise RuntimeError('Unexpected hardware configuration change')
        for c in run['cases']:
            if c['driver'] != 'simx' or min(c['coverage'].values()) <= 0 or c['cta_warps'] != c['mode'] + 2:
                raise RuntimeError('Unexpected test configuration')
            identity = tuple(c[x] for x in ('tiles', 'k', 'mode', 'c_depth', 'seed'))
            if key[1] not in ('trace', 'repeat'):
                cases[(key[0],) + identity] = c
    by_input = {}
    for (w, tiles, k, mode, depth, seed), c in cases.items():
        by_input.setdefault((tiles, k, seed), set()).add(c['output_hash'])
    if any(len(hashes) != 1 for hashes in by_input.values()):
        raise RuntimeError('Output mismatch across execution methods')
    for c in runs[(16, 'trace')]['cases']:
        key = (16,) + tuple(c[x] for x in ('tiles', 'k', 'mode', 'c_depth', 'seed'))
        if c['chain'] != cases[key]['chain']:
            raise RuntimeError('Debug execution changed counters')
    activities = [parse_activity(c) for c in runs[(16, 'trace')]['cases']]
    for activity in activities:
        with (HERE / f"activity_v{activity['mode']}.csv").open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=activity['rows'][0].keys())
            writer.writeheader()
            writer.writerows(activity['rows'])
    matrix_rows = []
    for tiles in (1, 2, 5):
        for k in (16, 64):
            values = [cases[(16, tiles, k, mode, 1, 1)]['chain']['cycles'] for mode in (2, 3, 4)]
            matrix_rows.append([tiles, k] + values + [f'{100 * (values[2] / values[1] - 1):+.2f}%'])
    depth_rows = []
    for k in (16, 64):
        d1 = cases[(16, 5, k, 4, 1, 1)]
        d2 = cases[(16, 5, k, 4, 2, 1)]
        depth_rows.append([k, d1['chain']['cycles'], d1['smem'], d2['chain']['cycles'], d2['smem'],
                           f"{100 * (d2['chain']['cycles'] / d1['chain']['cycles'] - 1):+.2f}%"])
    repeats = []
    for mode, depth in ((2, 1), (3, 1), (4, 1), (4, 2)):
        values = [c['chain']['cycles'] for c in runs[(16, 'repeat')]['cases']
                  if c['mode'] == mode and c['c_depth'] == depth]
        repeats.append([mode, depth, min(values), median(values), max(values)])
    rtl_path = COURSE / 'persistent_w16_rtl_results.json'
    rtl = json.loads(rtl_path.read_text()) if rtl_path.exists() else None
    if rtl and rtl.get('source_hashes') != reference['source_hashes']:
        raise RuntimeError('RTL source differs from the final SimX source')
    rtl_cases = rtl.get('cases', []) if rtl else []
    if rtl and rtl['status'] == 'complete' and (len(rtl_cases) != 3 or not all(c['passed'] for c in rtl_cases)):
        raise RuntimeError('Incomplete RTL correctness evidence')
    rtl_note = ''
    if rtl and rtl.get('error') and 'exit=124' in rtl['error'] and not rtl_cases:
        rtl_note = ('首次 rtlsim runtime 模型构建超过本批每例 600 秒的上限，进程组已结束；'
                    '超时前仍在 Verilator 生成的 C++ 模型编译阶段，没有进入 kernel 数值检查。'
                    '这是构建时间限制，不能解释为三阶段内核死锁，也不能报告为 RTL PASS。'
                    '日志还出现 jobserver 不可用并退回 -j1 的警告，实际并行未达到请求的 THREADS=4。'
                    '脚本现已把首次模型构建独立成 build_rtlsim 步骤，默认允许 1800 秒，'
                    '可用 --rtl-build-timeout 调整；每个数值 smoke 仍限 600 秒。'
                    '分离构建后的 RTL 流程本批尚未重跑。')
    summary = dict(simx_successful_executions=sum(len(run['cases']) for run in runs.values()),
        rtl_status=rtl['status'] if rtl else 'not-run', rtl_cases=rtl_cases,
        rtl_error=rtl.get('error') if rtl else None, rtl_note=rtl_note,
        matrix=matrix_rows, output_queue_depth=depth_rows, repeats=repeats,
        activities=activities, source_hashes=reference['source_hashes'],
        provenance={f'w{w}_{p}': run['logs'] for (w, p), run in runs.items()},
        limitations=['single persistent CTA only', 'activation uses one SIMT warp',
                     'ordinary baseline prefetches across output tiles',
                     'new kernel layouts: do not directly compare absolute cycles with earlier apps'])
    (HERE / 'analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (HERE / 'measurements.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['hardware_warps', 'phase', 'output_tiles', 'K', 'mode', 'C_depth', 'seed',
                         'cycles', 'instructions', 'reads', 'writes', 'smem_bytes', 'output_hash'])
        for (w, p), run in runs.items():
            for c in run['cases']:
                writer.writerow([w, p] + [c[x] for x in ('tiles', 'k', 'mode', 'c_depth', 'seed')]
                                + [c['chain'][x] for x in ('cycles', 'instructions', 'reads', 'writes')]
                                + [c['smem'], c['output_hash']])

    report = f'''# Project 20：连续输出 tile 的核内三阶段原型

## 结果概览

已实现同一 CTA 连续处理输出 tile 的普通融合、两阶段和三阶段 kernel。最终 {summary['simx_successful_executions']} 次 SimX 运行通过正负混合输入的精确数值检查。三阶段的共享内存输出队列支持 1/2 个槽位及尾部排空，并有计算 tile j+1 与 activation tile j 实际重叠的 trace 证据。

当前三阶段明显慢于两阶段和普通融合。主要观察是单个 activation warp 的处理跨度很长，增加输出队列深度只带来很小变化。现阶段完成的是功能与数据流验证，未获得三阶段加速结论。

## 1. 平台和公平对照

- 固定 Vortex v3.0 commit `{reference['source_commit']}`，工具链 v3.0.1。
- 主实验为 1 核、16 hardware warp、32 lanes/warp、issue width=4、4 TCU blocks，64 KiB LMEM；ALU/LSU/FPU 各 1 block，LSU pending=16，AMO RS=8，L2/L3 关闭。
- 新应用为 `tests/regression/persistent_pipeline`，旧应用和旧基线保留。没有修改 RTL/SimX 源码。
- 三个版本都连续处理多个完整输出 tile，每 CTA 只启动一次。普通融合 4 warp，独立 loader 两阶段 5 warp，独立 loader/compute/activation 三阶段 6 warp。
- 普通融合对照也支持跨输出 tile 的 A/B 预取。首版只在每个输出 tile 内预取，后续加强并重跑所有正式对照，避免将预取策略差异误当成专用 warp 的全部收益；首版日志留存但不混入下表。
- 每个输出 tile 为 64×32，K 必须完整归约后才能进入 ReLU。主矩阵为 M=64、N=32×输出 tile 数。
- A/B 双缓冲占 6 KiB。三阶段 C 队列分别占 8/16 KiB，因此每 CTA 共享内存为 14/22 KiB。普通融合和两阶段为 6 KiB。

## 2. 队列协议

输入队列使用 barrier 槽 0/1 表示 ready、2/3 表示 free，由 1 loader + 4 compute warp 参与。输入序号在输出 tile 边界也连续，支持跨输出预取。

输出队列使用槽 4/5 表示 ready、6/7 表示 free，由 4 compute + 1 activation warp 参与。compute 将完整 accumulator 存入共享内存后 arrive ready，activation 等待后读数据、执行 ReLU、写最终输出，再 arrive free。compute 可以在 activation 处理旧结果时先计算下一输出 tile，但覆盖 C 槽位前必须等待 free。

BAR/arrive 的现有实现等待 LSU 排空；SimX 位于 `sfu_unit.cpp`，RTL 位于 `VX_core.sv`/`VX_wctl_unit.sv`。原型沿用此语义，没有增加理想化零开销通信。输入和输出队列在 CTA 退出前分别完成尾部 free 握手。

## 3. 正确性与范围

主实验覆盖输出 tile 数 1/2/5、K=16/64、三种方法，共 18 例。随后追加 C depth=2、第二个 seed、8-warp 对照、debug trace 和关键用例重复运行。

CPU reference 沿用精确 dyadic 输入：A/B 为 `[-16,16]/16`，包含零行与零列。K 范围内 FP16 输入、FP32 点积均可精确表示。所有方法与队列深度的输出 hash 在相同输入下相同；未写回的 NaN sentinel 会触发失败。

全部正式用例只有一个 persistent CTA。尚未验证多个 persistent CTA 之间的交错，也未实现跨核通信。

## 4. 周期结果

主实验为 16 hardware warp、C depth=1、seed=1。周期越低越好。

{table(['输出 tile 数', 'K', '普通融合', '两阶段', '三阶段', '三阶段 vs 两阶段'], matrix_rows)}

不能将绝对周期直接与前几轮不同 binary/调度方式的应用比较。这里主要比较同一新应用中的三种组织，包含填充、排空、同步和存储成本。

## 5. 输出队列深度

5 个输出 tile，16 hardware warp：

{table(['K', 'depth1 cycles', 'depth1 SMEM bytes', 'depth2 cycles', 'depth2 SMEM bytes', 'depth2 周期变化'], depth_rows)}

depth2 多占 8 KiB 共享内存，但本组周期变化很小，不能把加深输出队列视作解决当前瓶颈的有效方法。

关键的 5-tile、K=64 用例各重复三次，范围如下。不同 seed/批次可有小幅周期差别，未将这些小差别作强结论。

{table(['mode：2普通/3两阶段/4三阶段', 'C depth', 'min cycles', 'median cycles', 'max cycles'], repeats)}

## 6. 实际阶段重叠

使用未插入 kernel 时间戳的既有 SimX debug 事件。当前形状下，每输出 tile、每计算 warp 有 `(K/16)×32` 个 WGMMA 计算 uop；两阶段每计算 warp 有 16 个 FMIN_MAX，三阶段的独立 activation warp 每输出 tile 有 64 个 FMIN_MAX。按 warp 内顺序分组，并核对全部事件数量。

{table(['mode', 'DXA transfers', 'WGMMA compute events', 'activation FMIN_MAX events', 'activation(j) 位于 compute(j+1) 时间窗口的事件', '与下一 tile TCU 事件同周期的不同 tick 数之和'], [[a['mode'], a['dxa_transfers'], a['wgmma_compute_events'], a['activation_events'], a['activation_events_during_next_compute_window'], a['same_cycle_activation_and_next_compute']] for a in activities])}

三阶段各输出 tile 的 activation 首个执行事件均晚于该 tile 的全部 WGMMA 计算事件；两阶段融合 epilogue 则检查每个 warp 完成自己的 K 归约后再处理其输出行，无须等待其他 warp 的独立输出行。三阶段观察到跨输出 tile 的计算/激活重叠；窗口交叠计数不等于硬件利用率，窗口也包含访存和调度等待。逐输出时间窗口见 `activity_v3.csv` 和 `activity_v4.csv`。

当前 activation 是一个 warp 逐步遍历 2048 个 FP32 元素，执行共享内存读、fmax 和最终存储。它的活动跨度明显长于下一 tile 的 WGMMA 活动跨度，因此即使有重叠，吞吐仍可能被 activation 限制。BAR 的核心级 LSU drain、队列交接和寄存器/栈管理也可能贡献成本，尚未完成逐项因果分解。

## 7. RTL 小规模检查

RTL 检查状态：`{summary['rtl_status']}`。已记录的逐例结果如下：

{table(['输出 tile 数', 'K', 'mode', '数值检查', '设备计数周期'], [[c['tiles'], c['k'], c['mode'], c['passed'], c.get('chain', {}).get('cycles')] for c in rtl_cases]) if rtl_cases else '尚无已完成 RTL 用例；详细状态和日志见对应 JSON。'}

{rtl_note}

这些小用例用于检查数值与进展，不能替代完整 model-parity 或 XRT 集成验证。若 RTL 尚在构建或失败，本报告不会将 SimX PASS 当作 RTL PASS。

## 8. 下一步

优先分解 activation 成本，例如比较 1/2 个 activation warp（保持 4 个计算 warp），并用 stage-role 对照区分共享内存交接与普通融合的差异；或采用连续元素分块以增加单个 activation warp 的并行访存。扩大资源时仍需记录总 warp/SMEM 代价。

应继续保留支持跨输出预取的普通融合强基线。完成多个 persistent CTA 的验证后，再扩展 L2 原子队列与 KMU placement。完整 WASP 编译器/RFQ、PPA 和跨核结果仍不在当前完成范围。

## 9. 复现

```bash
python3 'experiments/persistent_pipeline/run_persistent.py' --phase matrix --warps 16
python3 'experiments/persistent_pipeline/run_persistent.py' --phase depth --warps 16
python3 'experiments/persistent_pipeline/run_persistent.py' --phase depth --warps 8
python3 'experiments/persistent_pipeline/run_persistent.py' --phase trace --warps 16
python3 'experiments/persistent_pipeline/run_persistent.py' --phase repeat --warps 16
python3 'experiments/persistent_pipeline/run_persistent.py' --phase rtl --warps 16 --rtl-build-timeout 3600
python3 'experiments/persistent_pipeline/analyze_persistent.py'
```

源码镜像位于本目录 `persistent_pipeline/`。逐例数据为 `measurements.csv`，完整汇总与日志路径在 `analysis.json`，原始 debug trace 压缩保留。RTL 命令为后续重试入口，本批没有完成 RTL 数值检查；构建请求 `THREADS=4`，但本次日志报告 jobserver 失效并退回串行构建。
'''
    (HERE / 'RESULTS.md').write_text(report, encoding='utf-8')
    print(json.dumps(dict(simx=summary['simx_successful_executions'], rtl=summary['rtl_status'],
                          matrix=matrix_rows, activity=[{k: a[k] for k in ('mode', 'activation_events_during_next_compute_window',
                                       'same_cycle_activation_and_next_compute')} for a in activities]), ensure_ascii=False))


if __name__ == '__main__':
    main()
