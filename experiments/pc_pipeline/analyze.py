#!/usr/bin/env python3
"""Summarize saved measurements and existing SimX DXA/TCU trace events."""

import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import log_path, result_path
import re

HERE = Path(__file__).resolve().parent


def local_path(path):
    return log_path(path)


def main():
    old = json.loads(result_path('proposal_probe_results.json').read_text())
    runs = {phase: json.loads(result_path(f'pc_{phase}_results.json').read_text())
            for phase in ('micro', 'gemm', 'diagnostic', 'trace', 'indexing')}
    for phase, run in runs.items():
        if phase != 'diagnostic' and run['status'] != 'complete':
            raise RuntimeError('Incomplete run: ' + phase)
        if not all(case['passed'] for case in run['cases']):
            raise RuntimeError('Failed case: ' + phase)
    if len(runs['diagnostic']['cases']) != 6 or len(runs['trace']['debug_traces']) != 2:
        raise RuntimeError('Expected six controls and two recovered debug traces')
    if any(step['exit_code'] and not step['name'].startswith('trace_depth')
           for step in runs['diagnostic']['steps']):
        raise RuntimeError('Diagnostic failure outside the recovered trace step')
    cases = {case['name']: case for run in runs.values() for case in run['cases']}

    def cycles(name):
        return int(cases[name]['summary'][1])

    def change(after, before):
        return 100 * (after / before - 1)

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
            ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])

    trace_summary = []
    for trace in runs['trace']['debug_traces']:
        text = local_path(trace['log']).read_text(errors='replace')
        starts = [int(x) for x in re.findall(r'TRACE\s+(\d+): .*dxa-core\[0\] start:', text)]
        ends = [int(x) for x in re.findall(r'TRACE\s+(\d+): .*dxa-core\[0\] complete:', text)]
        compute = [int(x) for x in re.findall(r'TRACE\s+(\d+): .*?-tcu execute: op=WGMMA,.*rd=f', text)]
        if len(starts) != 8 or len(ends) != 8 or len(compute) != 512:
            raise RuntimeError('Unexpected trace event counts')
        intervals = list(zip(starts, ends))
        if not all(a < b for a, b in intervals):
            raise RuntimeError('Invalid DXA interval pairing')
        overlap = [tick for tick in compute if any(a <= tick < b for a, b in intervals)]
        summary = re.findall(r'PERF:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)', text)[-1]
        expected = cases[f"pc_d{trace['depth']}_small_k64"]['summary']
        if list(summary) != expected:
            raise RuntimeError('Debug execution differs from measured execution')
        trace_summary.append(dict(depth=trace['depth'], dxa_intervals=intervals,
            wgmma_compute_events=len(compute), events_inside_dxa_interval=len(overlap),
            distinct_overlap_cycles=len(set(overlap)), first_overlap_cycle=min(overlap) if overlap else None,
            summary=summary, source=trace['log']))
        with (HERE / f"trace_depth{trace['depth']}_events.csv").open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['event', 'start_cycle', 'end_cycle', 'inside_dxa_interval'])
            for a, b in intervals:
                writer.writerow(['DXA_active', a, b, ''])
            for tick in compute:
                writer.writerow(['WGMMA_compute_uop', tick, tick,
                                 int(any(a <= tick < b for a, b in intervals))])

    old_single = {case['k']: case for case in old['cases'] if case['variant'] == 'single'}
    old_rows = []
    for case in sorted((c for c in old['cases'] if c['variant'] == 'double'), key=lambda c: c['k']):
        baseline = old_single[case['k']]
        old_rows.append([case['k'], int(baseline['summary'][1]), int(case['summary'][1]),
            f"{change(int(case['summary'][1]), int(baseline['summary'][1])):+.2f}%",
            f"{change(int(case['summary'][0]), int(baseline['summary'][0])):+.2f}%"])

    gemm_rows = []
    for k in (16, 32, 48, 64, 256):
        gemm_rows.append([f'64×32×{k}', cycles(f'single_small_k{k}'), cycles(f'double_small_k{k}'),
                          cycles(f'pc_d1_small_k{k}'), cycles(f'pc_d2_small_k{k}')])
    gemm_rows.append(['128×128×64', cycles('single_large_k64'), cycles('double_large_k64'),
                      cycles('pc_d1_large_k64'), cycles('pc_d2_large_k64')])
    micro_rows = [[work, cycles(f'perf_p0_w{work}'), cycles(f'control_pc_d1_w{work}'),
                   cycles(f'perf_p1_w{work}'),
                   f"{change(cycles(f'perf_p1_w{work}'), cycles(f'control_pc_d1_w{work}')):+.2f}%"]
                  for work in (0, 32)]
    control_rows = [[f'{m}×{n}×{k}', cycles(f'control5_m{m}_n{n}_k{k}'),
                     cycles(f'pc_d2_{size}_k{k}'),
                     f"{change(cycles(f'pc_d2_{size}_k{k}'), cycles(f'control5_m{m}_n{n}_k{k}')):+.2f}%"]
                    for m, n, size in ((64, 32, 'small'), (128, 128, 'large')) for k in (64, 256)]
    mask_rows = []
    for m, n, size, k in ((64, 32, 'small', 64), (64, 32, 'small', 256),
                           (128, 128, 'large', 64), (128, 128, 'large', 256)):
        before = cycles(f'pc_d2_{size}_k{k}')
        after = cycles(f'mask_d2_m{m}_n{n}_k{k}')
        mask_rows.append([f'{m}×{n}×{k}', before, after, f'{change(after, before):+.2f}%'])
    count = sum(len(run['cases']) for run in runs.values()) + len(trace_summary)
    summary = dict(source_commit=runs['gemm']['source_commit'], successful_executions=count,
        effective_config=runs['indexing']['effective_config'],
        old_buffer_comparison=old_rows, gemm_comparison=gemm_rows,
        micro_comparison=micro_rows, five_warp_controls=control_rows,
        indexing_comparison=mask_rows, trace_summary=trace_summary,
        provenance={phase: run['logs'] for phase, run in runs.items()},
        diagnostic_note='Control cases passed; first trace redirection failed on a space in path. Trace phase recovered both runs.')
    (HERE / 'analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    report = f'''# Project 20：核内 producer–consumer 第一轮实验

本记录对应从 proposal 准备进入实现后的第一批实验。结论只针对固定 v3.0 和当前配置；本批完成了软件阶段原型，没有实现完整 WASP 编译器、RFQ、activation 阶段或跨核队列。

## 1. 配置、代码和验证范围

- Vortex commit：`{runs['gemm']['source_commit']}`；RV32、工具链 v3.0.1。
- 1 核、8 hardware warps、32 lanes/warp、issue width=4、4 TCU blocks；FP16 输入，FP32 输出，RS WGMMA。
- 有效配置：`{json.dumps(runs['indexing']['effective_config'], sort_keys=True)}`。L2/L3 关闭；缓存状态口径与原始基线一致。
- 新构建目录：`/root/cs254a/project20/vortex/build32-pipeline`，旧 baseline 构建保留。
- 新应用：`dxa_pc`、`sgemm_tcu_wg_pc`、`sgemm_tcu_wg_control5`、`sgemm_tcu_wg_pc_mask`。均在 `tests/regression/` 下另建目录；没有修改受版本控制的上游 RTL、SimX 或现有 GEMM 应用。
- 最终协议版本累计 **{count} 次运行通过**：包含常规正确性/计数器测试和 2 次 debug trace。GEMM 沿用上游 CPU reference 和按 K 规定的 ULP 门限；微测试采用 uint32 逐元素精确匹配。
- 首版也通过了测试，但其 release barrier 尾部尚未排空。最终版补齐所有槽位的 release 握手后重新测量；首版日志留存，不混入以下比较。

## 2. 对已有小测试的分析

以下为 M=N=128 的旧记录；变化为“双缓冲 / 单缓冲 − 1”。

{table(['K', '单缓冲 cycles', '双缓冲 cycles', '周期变化', '指令数变化'], old_rows)}

在本组输入中，双缓冲额外动态指令数随 K tile 数线性增加：`728 × (K/16) − 432`。这表明管理开销持续存在，但不能将每条额外指令直接折算成固定周期代价。

双缓冲的 shared-memory bank stalls 在 K=32/128/256 时下降，在 K=64 时上升；K=256 的 stalls 下降而总周期上升。IPC 的提高也包含更多管理指令。因此，单项 stall 或 IPC 都不能代替端到端周期。

原配置每 CTA 使用 4 warp、共享内存 3/6 KiB，而 LMEM 为 64 KiB。两种缓冲容量均容得下两个 CTA，不能把旧结果中的退化直接解释为“共享内存翻倍导致 CTA 驻留减少”。改变 K 也会改变 footprint 和访问跨度，旧实验没有隔离这些因素。

## 3. 新原型与同步协议

`dxa_pc` 使用同一 CTA 内的两个 warp：一个发 DXA，另一个消费共享内存中的 tile，并写回结果。消费操作包含可配置次数的 uint32 递推，用于制造可控计算量，不代表 AI kernel。

每个槽位分别使用 ready 和 free barrier。生产者先登记预期 DXA completion，再发请求并非阻塞地 arrive；消费者等待 ready 后读 tile，最后 arrive free。生产者重用槽位前等待 free。最后一轮也完整排空 free generation，避免 CTA 退休时遗留未完成的释放计数。

GEMM 原型保留 4 个计算 warp，增加第 5 个加载 warp；计算 tile 的 M 维仍由 4 个计算 warp 决定，barrier 则由 5 个 warp 参与。修正 host launch、tile 索引和 barrier 数量后，此组织在本批 SimX 中可以正确运行。原 host 示例的 4-warp 设定不等于本批已证明硬件不能容纳额外 loader。

## 4. 微测试：区别流水深度和协议开销

64 tiles，每 tile 64 个 uint32。所有版本使用两个 CTA warp；“串行”版本每个 tile 同步等待，“PC”版本采用上述独立阶段协议。

{table(['每元素递推次数', '串行 depth2 cycles', 'PC depth1 cycles', 'PC depth2 cycles', 'PC depth2 相对 depth1'], micro_rows)}

PC 相对串行的改善同时包含同步结构和指令开销差异；PC depth1/depth2 是更直接的缓冲深度对照。递推次数增大后，两者差距变小，不能把微测试的收益外推到 GEMM。

## 5. GEMM：能重叠，但尚未超过原有双缓冲

64×32 的输出恰好对应一个 CTA；128×128 对应 8 个 CTA。下表为最新、同一批构建中的设备周期，包含 kernel 内填充、排空和同步，不是 host 仿真墙钟时间。

{table(['M×N×K', '上游单缓冲 4 warp', '上游双缓冲 4 warp', 'PC depth1 5 warp', 'PC depth2 5 warp'], gemm_rows)}

同为 K=64，单 CTA 的上游双缓冲优于单缓冲，而 8-CTA 用例仍复现了旧记录中双缓冲较慢的结果。这说明 workload/调度条件会改变 buffering 的效果。

PC depth2 优于 PC depth1，但当前原型还没有超过上游 4-warp 双缓冲。需要同时考虑协议管理、指令选择和 CTA 并发，不能只根据“发生了重叠”推断最终会更快。

### 5-warp 资源对照

在普通双缓冲中增加一个只参与 barrier、不执行 GEMM 的 warp。计算量和输出 tile 不变，CTA 同样占用 5 个 warp。

{table(['M×N×K', '5-warp 普通双缓冲 cycles', '5-warp PC depth2 cycles', 'PC 相对对照'], control_rows)}

8 个 hardware warp 下，4-warp CTA 的 warp 容量上限是 2 个 CTA，5-warp CTA 的上限是 1 个。128×128×64 中，4-warp 双缓冲为 41,970 cycles，5-warp 普通双缓冲为 52,810，PC 为 47,940。对照支持额外 warp/同步参与的成本相当明显；它同时改变了 barrier 开销，不能视作只改变 occupancy 的纯对照。

## 6. 时序证据

从既有 SimX debug 事件配对 DXA worker 的 start/complete，得到半开区间 `[start,end)`；检查 WGMMA 浮点目的寄存器计算 uop 的执行时间是否落在其中。没有修改模拟器增加理想化重叠。

{table(['PC 缓冲深度', 'DXA transfers', 'WGMMA 计算事件', '落在 DXA 活动区间的事件', '这些事件的不同周期数'], [[x['depth'], len(x['dxa_intervals']), x['wgmma_compute_events'], x['events_inside_dxa_interval'], x['distinct_overlap_cycles']] for x in trace_summary])}

depth2 中，下一块 A tile 的 DXA 活动区间为 1725–2217，第一个与之重叠的 WGMMA 计算事件出现在 cycle 2086。depth1 没有此类交叠事件。debug 与不启用 debug 的对应执行指令数、周期一致。

这是加载与计算实际重叠的证据。上述事件计数不能直接当作 TCU 或内存带宽利用率。可复查 `trace_depth1_events.csv`、`trace_depth2_events.csv` 和 `analysis.json` 中的原始日志路径。

## 7. 缓冲索引的单变量实验

原型反汇编存在 `remu`。因为 host 只允许 depth=1/2，将三处 `i % depth` 替换为等价的 `i & (depth-1)`，其他协议和硬件配置保持一致，另建 mask 应用。覆盖 depth1/2、单 tile、奇数 tile、多次槽位复用与多 CTA。

{table(['M×N×K', '原 PC depth2 cycles', 'mask 索引 cycles', 'mask 相对原 PC'], mask_rows)}

mask 版本 kernel text 中是否存在 `remu`：`{runs['indexing']['kernel_text_has_remu']}`。这组实验只解释当前手工原型的索引代价，不等于复现了 WASP 的编译器优化。

## 8. 结论与下一批工作

当前已经从“上游单/双缓冲比较”推进到“正确的独立 loader/compute 原型，并有时序重叠证据”。目前的性能瓶颈更值得围绕 CTA 资源分配、缓冲管理和既有双缓冲的差异继续定位，而不是仅增加队列深度。

下一批建议：以最佳已验证索引版本为起点，先将 ReLU 融合入 GEMM，并做独立 ReLU kernel 对照，再评估是否值得增加独立 activation 阶段。上游测试输入为非负数；验证 ReLU 时应增加正负混合输入和独立 CPU reference，避免 ReLU 退化成恒等运算。跨核/L2 队列在此之后开展。

本批结果仅验证正常时序 SimX。未进行 RTL parity、XRT 集成、PPA 或 FPGA 验证；也未形成完整 load–GEMM–activation 算子链的性能结论。GEMM 本身的输入输出工作量不变，本批不主张已经减少整链中间结果的片外往返。

## 9. 复现与记录

在 Ubuntu-22.04 的 root 环境执行；源码和构建保持在无空格 Linux 路径：

```bash
python3 'experiments/pc_pipeline/run.py' --phase micro
python3 'experiments/pc_pipeline/run.py' --phase gemm
python3 'experiments/pc_pipeline/run.py' --phase diagnostic
python3 'experiments/pc_pipeline/run.py' --phase indexing
python3 'experiments/pc_pipeline/analyze.py'
```

`diagnostic` 包含资源对照和两次 trace；`trace` 可只重跑 trace。脚本记录命令、退出码、配置、源文件快照或哈希和逐例结果。完整溯源见 `analysis.json`。

首次 diagnostic 的 6 个控制用例通过，但 trace 的 shell 日志重定向因路径中的空格失败。已改用 Linux 无空格临时日志再复制回共享目录，单独 trace 阶段两例成功，失败日志保留。配置元数据读取补充了 build 中使用的 RV32 XLEN 参数；这不改变已运行用例的硬件配置。
'''
    (HERE / 'RESULTS.md').write_text(report, encoding='utf-8')
    print(json.dumps(dict(successful_executions=count, indexing=mask_rows,
                          traces=[{k: x[k] for k in ('depth', 'events_inside_dxa_interval')} for x in trace_summary]),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
