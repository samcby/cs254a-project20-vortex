#!/usr/bin/env python3
"""Check the controlled resource sweep and reconstruct CTA residency."""

import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import log_path

from config_probe import HERE, differences
from trace_residency import parse

COURSE = HERE.parents[1]


def local(path):
    return log_path(path)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def change(after, before):
    return 100 * (after / before - 1)


def main():
    runs = {(w, phase): json.loads((COURSE / f'warp{w}_{phase}_results.json').read_text())
            for w in (8, 16) for phase in ('matrix', 'trace')}
    previous = json.loads((COURSE / 'relu_matrix_results.json').read_text())
    previous_extra = json.loads((COURSE / 'relu_validation_results.json').read_text())
    old = {(c['m'], c['n'], c['k'], c['seed'], c['mode']): c
           for c in previous['cases'] + previous_extra['cases'] if c['mode'] in (2, 3)}
    cases = {}
    references = runs[(8, 'matrix')]
    for (warps, phase), run in runs.items():
        if run['status'] != 'complete' or not all(c['passed'] for c in run['cases']):
            raise RuntimeError('Unfinished or failed sweep')
        if run['source_hashes'] != references['source_hashes']:
            raise RuntimeError('Application source mismatch')
        if run['binary_sha256'] != references['binary_sha256']:
            raise RuntimeError('Device binary changed between resource configurations')
        diff = differences(references['effective_config'], run['effective_config'])
        if set(diff) != ({'VX_CFG_NUM_WARPS'} if warps == 16 else set()):
            raise RuntimeError('Unexpected configuration difference')
        for c in run['cases']:
            if c['hardware_warps'] != warps or c['cta_warps'] != (4 if c['mode'] == 2 else 5):
                raise RuntimeError('Unexpected launch shape')
            key = (c['m'], c['n'], c['k'], c['seed'], c['mode'])
            if phase == 'matrix':
                cases[(warps,) + key] = c
                if warps == 8 and key in old and (c['chain'] != old[key]['chain']
                                                  or c['output_hash'] != old[key]['output_hash']):
                    raise RuntimeError('Eight-warp baseline no longer reproduces prior results')
            elif c['chain'] != next(x['chain'] for x in runs[(warps, 'matrix')]['cases']
                                   if all(x[k] == c[k] for k in ('m', 'n', 'k', 'seed', 'mode'))):
                raise RuntimeError('Debug instrumentation changed execution counts')

    groups = {key[1:5] for key in cases}
    for group in groups:
        if len({cases[(w,) + group + (mode,)]['output_hash'] for w in (8, 16) for mode in (2, 3)}) != 1:
            raise RuntimeError('Output mismatch across hardware/algorithm configurations')

    residency = []
    for w in (8, 16):
        for c in runs[(w, 'trace')]['cases']:
            summary, rows = parse(local(c['trace']), w, c['cta_warps'], 8)
            summary.update(mode=c['mode'], trace=c['trace'])
            residency.append(summary)
            with (HERE / f"residency_w{w}_v{c['mode']}.csv").open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    shapes = [(64, 32, k) for k in (64, 256)] + [(m, 128, k) for m in (128, 256) for k in (64, 256)]
    performance, scaling, traffic = [], [], []
    for m, n, k in shapes:
        b8, p8, b16, p16 = [cases[(w, m, n, k, 1, mode)]['chain']
                            for w, mode in ((8, 2), (8, 3), (16, 2), (16, 3))]
        performance.append([f'{m}×{n}×{k}', b8['cycles'], p8['cycles'],
            f"{change(p8['cycles'], b8['cycles']):+.2f}%", b16['cycles'], p16['cycles'],
            f"{change(p16['cycles'], b16['cycles']):+.2f}%"])
        scaling.append([f'{m}×{n}×{k}', f"{change(b16['cycles'], b8['cycles']):+.2f}%",
                        f"{change(p16['cycles'], p8['cycles']):+.2f}%"])
        traffic.append([f'{m}×{n}×{k}'] + [x['reads'] + x['writes'] for x in (b8, p8, b16, p16)])
    peak_rows = [[x['hardware_warps'], '普通融合' if x['mode'] == 2 else 'PC 融合',
                  x['peak']['live_warps'], x['peak']['started_live_ctas'],
                  x['peak']['fully_admitted_live_ctas'], x['peak']['all_warps_live_ctas']]
                 for x in residency]
    count = sum(len(run['cases']) for run in runs.values())
    summary = dict(successful_runs=count, matrix_runs=len(cases), trace_runs=len(residency),
        controlled_differences=differences(references['effective_config'], runs[(16, 'matrix')]['effective_config']),
        source_hashes=references['source_hashes'], identical_binary_sha256=references['binary_sha256'],
        performance=performance, scaling_16_vs_8=scaling, memory_request_totals=traffic,
        residency=residency, provenance={f'w{w}_{p}': run['logs'] for (w, p), run in runs.items()},
        limitations=['not an equal-area or equal-register-capacity comparison',
                     'CTA residency peaks are event-order peaks, not average occupancy',
                     'only finite exact-dyadic inputs and timing SimX are verified'])
    (HERE / 'analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (HERE / 'measurements.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['hardware_warps', 'M', 'N', 'K', 'seed', 'mode', 'cycles',
                         'instructions', 'dram_read_requests', 'dram_write_requests', 'output_hash'])
        for key in sorted(cases):
            c = cases[key]
            writer.writerow(list(key) + [c['chain'][name] for name in ('cycles', 'instructions', 'reads', 'writes')]
                            + [c['output_hash']])

    report = f'''# Project 20：固定计算与访存资源的 8/16-warp 对照

## 结论

本轮用相同 kernel 软件验证额外 warp 状态容量的影响。**32 个性能/正确性用例、4 个 debug 用例全部通过**。多 CTA 场景下，PC 融合在 16-warp 配置中比自身的 8-warp 配置减少约 17%–20% 的周期；相对普通融合的差距由约 19%–23% 缩小到 −2.6% 至 +1.0%。单 CTA 场景基本不变。

调度 trace 同时验证：普通融合中“一个 CTA 的所有 warp 同时存活”的峰值由 2 增至 4，PC 由 1 增至 3。结果支持先前多 CTA 退化与 warp 驻留资源、调度交错密切相关的判断，但不等于证明全部差距都由某一个 occupancy 数字解释。

## 1. 实验控制

- 固定源码：Vortex v3.0，commit `{references['source_commit']}`，工具链 v3.0.1。
- 独立构建：`build32-warp-capacity-8`、`build32-warp-capacity-16`，原构建与原始基线保留。
- 固定 1 核、32 lanes/warp、issue width=4、4 TCU blocks、ALU/LSU/FPU 各 1 block、64 KiB LMEM，以及全部缓存配置；L2/L3 关闭。
- NUM_WARPS 改为 16 时，默认 LSU pending size 会从 16 变为 32，AMO RS size 从 8 变为 16。因此本轮对两个配置均显式固定 `VX_CFG_LSU_PENDING_SIZE=16`、`VX_CFG_AMO_RS_SIZE=8`、`VX_CFG_LMEM_LOG_SIZE=16`。
- 完整解析后的配置差异仅为 `VX_CFG_NUM_WARPS: 8→16`。8-warp 控制没有改变此前的任何解析配置项，且重现了先前相同用例的周期、请求数和输出 hash。
- 采用上一轮已经验证的 `gemm_relu` 原样源码，普通融合为 4 warp/CTA，PC 为 4 compute + 1 loader。二者都融合 ReLU，使用双缓冲；未增加独立 activation warp。
- 两个配置构建出的 **device kernel binary 哈希完全一致**：`{references['binary_sha256']}`。SimX driver 分别按对应配置构建，不混用。

增加 hardware warp 会自然增加寄存器/上下文和调度状态容量。本实验固定的是计算单元、共享内存和显式访存队列，不是等面积、等寄存器总容量比较；没有测量 PPA。

## 2. 工作负载和正确性

主 seed=1 覆盖单 CTA 的 64×32×64/256、8 CTA 的 128×128×64/256，以及 16 CTA 的 256×128×64/256。64×32×256 和 128×128×64 追加 seed=2026。

输入、CPU reference、零行/零列、NaN sentinel 和精确数值检查均沿用上一轮。所有 shape/seed 下，两个执行方法和两个硬件配置的最终输出 hash 一致。4 个 debug 用例的周期、指令数和内存请求数也分别与对应普通运行一致。

## 3. 周期结果

下表为 seed=1 的模拟设备周期，越低越好。“PC 变化”指 PC/普通融合−1。

{table(['M×N×K', '8w 普通融合', '8w PC', '8w PC 变化', '16w 普通融合', '16w PC', '16w PC 变化'], performance)}

将各方法自身的 16-warp 结果与其 8-warp 结果比较：

{table(['M×N×K', '普通融合周期变化', 'PC 周期变化'], scaling)}

单 CTA 没有额外的 CTA 并发机会，两种方法都仅有几个周期的差异。多 CTA 中 PC 获益明显更大；普通融合的变化较小，且不总是更快。这使“额外可驻留 warp 帮助 PC 隐藏等待”的解释比“增加了计算吞吐”更符合当前对照结果。

## 4. Trace：完整与部分入驻的 CTA

从既有 scheduler 的 `dispatch CTA warp` 和 `tmask=0` 退出事件重建 warp→CTA 归属。每个 trace 包含 8 个 CTA；普通融合各有 32 次 warp 分配/退出，PC 各有 40 次，全部配对且最终无残留。

{table(['hardware warps', '方法', '峰值 live warp', '峰值已开始且未结束 CTA', '峰值全部 rank 曾入驻且未结束 CTA', '峰值全部 warp 同时存活 CTA'], peak_rows)}

dispatcher 是逐 warp 分配的，不会先原子预留一个 CTA 所需的全部 warp。一个 CTA 的部分 warp 退出后，其他 CTA 也可能提前入驻。因此“8 个 hardware warp 只能装一个 5-warp CTA”只适合作为完整 CTA 的容量直觉，不能解释为调度器绝不允许第二个 CTA 出现。

上述不同峰值不一定发生于同一时刻，也不是时间平均 occupancy。dispatch 日志没有精确 cycle 字段，因此 CSV 的 `preceding_trace_cycle` 仅用于定位相邻 trace，峰值统计依据事件顺序，不据此计算平均驻留时间。

## 5. 内存请求与动态执行

下表为全局内存 read+write 请求总数，包含数据、指令和栈访问；不直接当作 tensor 字节或能耗。

{table(['M×N×K', '8w 普通融合', '8w PC', '16w 普通融合', '16w PC'], traffic)}

相同 kernel binary 仍可能因更多物理 warp 槽位首次初始化、调度交错和栈地址访问而执行不同数量的动态指令、产生不同请求。故这次干预是 warp 状态容量及其调度效果的整体变化，尚未将每项成本独立分解。

测量沿用已修正口径：核心周期/指令跨 launch 累计，内存计数每次 launch 重置。本批所有比较都是单次融合 kernel，不含主机调度间隔；未将先前分离 kernel 的缓存重置问题混入本轮对照。

## 6. 对项目方向的影响

现在已有证据说明，手工 warp specialization 的有效性与可驻留 warp 资源相关。应继续保留 8-warp 课程基线和 16-warp 敏感性对照，不能仅挑选更有利配置报告加速，也不能把增加 warp 描述成没有硬件成本的优化。

后续完整三阶段队列设计，应同时考虑 loader、compute、activation 的 warp 开销和 output tile 调度；普通融合仍是强对照。当前尚未实现独立 activation warp、跨核/L2 队列、KMU placement 或完整 WASP 编译器/RFQ。RTL parity、XRT 和 PPA 也未进行。

## 7. 复现和文件

```bash
python3 'experiments/warp_capacity/config_probe.py'
python3 'experiments/warp_capacity/run_sweep.py' --phase matrix --warps 8
python3 'experiments/warp_capacity/run_sweep.py' --phase matrix --warps 16
python3 'experiments/warp_capacity/run_sweep.py' --phase trace --warps 8
python3 'experiments/warp_capacity/run_sweep.py' --phase trace --warps 16
python3 'experiments/warp_capacity/analyze_sweep.py'
```

- `configurations.json`：自然配置变化、控制后的完整配置。
- `analysis.json`、`measurements.csv`：汇总与逐例数据。
- `residency_w*_v*.csv`：逐事件 warp/CTA 归属变化。
- 原始 trace 已 gzip 压缩保留；日志、配置、源码和 device binary 快照路径见 `analysis.json`。

本批仅新增实验脚本与独立构建，没有修改受版本控制的 RTL/SimX/原始 kernel 源码。
'''
    (HERE / 'RESULTS.md').write_text(report, encoding='utf-8')
    print(json.dumps(dict(runs=count, performance=performance, residency=peak_rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
