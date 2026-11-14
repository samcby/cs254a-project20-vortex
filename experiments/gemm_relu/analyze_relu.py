#!/usr/bin/env python3
"""Validate chain counter accounting and summarize the signed-input experiments."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURSE = HERE.parents[1]


def main():
    runs = {phase: json.loads((COURSE / f'relu_{phase}_results.json').read_text())
            for phase in ('matrix', 'activation', 'validation')}
    cases = {}
    launches = 0
    for phase, run in runs.items():
        if run['status'] != 'complete' or run.get('counter_semantics') != 'core-delta_uncore-per-launch_v2':
            raise RuntimeError('Incomplete or obsolete counter accounting: ' + phase)
        if run['source_hashes'] != runs['matrix']['source_hashes']:
            raise RuntimeError('Comparisons require the same device application sources')
        for case in run['cases']:
            if not case['passed'] or min(case['coverage'].values()) <= 0:
                raise RuntimeError('Missing correctness or activation coverage')
            for metric in ('cycles', 'instructions', 'reads', 'writes'):
                if sum(stage[metric] for stage in case['stages']) != case['chain'][metric]:
                    raise RuntimeError('Incorrect stage aggregation')
            for metric in ('cycles', 'instructions'):
                if case['chain'][metric] != case['snapshots'][-1]['cumulative_' + metric]:
                    raise RuntimeError('Core counter was double counted')
            key = tuple(case[x] for x in ('m', 'n', 'k', 'seed', 'mode', 'activation_stride'))
            if key in cases:
                raise RuntimeError('Duplicate case')
            cases[key] = case
            launches += case['chain']['launches']

    groups = sorted({key[:4] for key in cases})
    for group in groups:
        relu_cases = [cases[group + tail] for tail in ((1, 0), (1, 1), (2, 0), (3, 0))]
        if len({case['output_hash'] for case in relu_cases}) != 1:
            raise RuntimeError('ReLU outputs differ across execution methods')

    def case(m, n, k, mode, stride=0, seed=1):
        return cases[(m, n, k, seed, mode, stride)]

    def delta(a, b):
        return 100 * (a / b - 1)

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
            ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])

    shapes = [(64, 32, k) for k in (16, 48, 64, 256)] + [(128, 128, k) for k in (64, 256)]
    performance, traffic, activation = [], [], []
    for m, n, k in shapes:
        raw = case(m, n, k, 0)
        separate = case(m, n, k, 1)
        tuned = case(m, n, k, 1, 1)
        fused = case(m, n, k, 2)
        pc = case(m, n, k, 3)
        performance.append([f'{m}×{n}×{k}', separate['chain']['cycles'], tuned['chain']['cycles'],
            fused['chain']['cycles'], pc['chain']['cycles'],
            f"{delta(fused['chain']['cycles'], tuned['chain']['cycles']):+.2f}%",
            f"{delta(pc['chain']['cycles'], fused['chain']['cycles']):+.2f}%"])
        traffic.append([f'{m}×{n}×{k}', tuned['chain']['reads'], tuned['chain']['writes'],
            fused['chain']['reads'], fused['chain']['writes'],
            pc['chain']['reads'], pc['chain']['writes'], 2 * m * n * 4])
        activation.append([f'{m}×{n}×{k}', raw['chain']['cycles'],
            fused['chain']['cycles'] - raw['chain']['cycles'],
            fused['chain']['instructions'] - raw['chain']['instructions'],
            separate['stages'][1]['cycles'], tuned['stages'][1]['cycles']])

    summary = dict(successful_host_runs=len(cases), successful_kernel_launches=launches,
        input_groups=len(groups), performance=performance, memory_requests=traffic,
        activation_costs=activation, provenance={p: r['logs'] for p, r in runs.items()},
        effective_config=runs['matrix']['effective_config'], source_hashes=runs['matrix']['source_hashes'],
        metric='cumulative core cycle/instruction deltas, plus per-launch uncore memory counts',
        limitations=['cache state resets on each SimX launch', 'host scheduling gaps excluded',
                     'exact dyadic finite inputs, not arbitrary real-valued numerical stress',
                     'activation fused in compute warps; no separate activation warp yet'])
    (HERE / 'analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (HERE / 'measurements.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['M', 'N', 'K', 'seed', 'mode', 'activation_stride', 'launches',
                         'cycles', 'instructions', 'dram_read_requests', 'dram_write_requests',
                         'positive', 'negative', 'zero', 'output_hash'])
        for key in sorted(cases):
            c = cases[key]
            writer.writerow(list(key) + [c['chain'][x] for x in ('launches', 'cycles', 'instructions', 'reads', 'writes')]
                            + [c['coverage'][x] for x in ('positive', 'negative', 'zero')] + [c['output_hash']])

    report = f'''# Project 20：GEMM→ReLU 与融合对照

## 范围与结论

本轮新增正负混合输入下的 GEMM→ReLU 实验，比较普通双缓冲 GEMM 后接独立 ReLU、普通融合，以及独立 loader/compute 的融合版本。另加入 grid-stride ReLU，降低独立 activation 的 CTA 管理开销。

最终 **{len(cases)} 次主机应用运行、{launches} 次 kernel launch 全部通过**。普通融合在全部 6 个主配置中优于经过 grid-stride 优化的分离执行。PC 融合在单 CTA 的 64×32×256 中优于普通融合约 8.77%，第二个输入 seed 复现同样周期；其余主配置仍较慢。结果说明收益依赖 workload 与 CTA 资源分配。

## 1. 固定平台与实现

- Vortex v3.0 commit `{runs['matrix']['source_commit']}`，工具链 v3.0.1，正常时序 SimX。
- 1 核、8 hardware warps、32 lanes/warp、issue width=4、4 TCU blocks、64 KiB LMEM；L2/L3 关闭。硬件配置沿用上一批。
- 新应用为 `tests/regression/gemm_relu`，使用已有 `build32-pipeline`；旧应用和原始基线保留，RTL/SimX 实现未修改。
- 同一个 device module 包含 `gemm_base`、`gemm_pc`、`relu_k`、`relu_grid_stride`。各执行方法使用同一份 binary、相同输入和相同 buffer 分配顺序。
- 普通 GEMM 每 CTA 4 warp，PC 为 4 compute + 1 loader。两者都使用两组 A/B 共享内存缓冲；PC 沿用上一批通过验证的位运算索引及 ready/free 排空协议。
- `mode=0` 为无激活 GEMM 控制；`mode=1` 为分离执行；`mode=2` 为普通融合；`mode=3` 为 PC 加融合激活。普通融合与 PC 融合均在完成整个 K 归约后，对 accumulator 执行 fmax，再存最终结果。本批没有独立 activation warp。
- 独立 ReLU 为 out-of-place C→Y，128 threads/CTA。默认版本每线程一个元素；grid-stride 版本限制为两个 CTA，通过循环覆盖所有元素。

## 2. 正确性：输入真正触发 ReLU

A/B 元素从 `[-16,16]/16` 取值，同时固定零行、零列。输入可由 FP16 精确表示，乘积和 K≤256 的累加在 FP32 中也可精确表示。CPU 使用整数点积除以 256 得到 reference，因此所有输出采用精确数值比较，无须放宽浮点容限；正负零按数值相等处理。

每个用例都包含正输出、负输出及零输出。分离版本额外下载并校验中间 C，确认 GEMM 本身正确；融合版本检查最终 ReLU 输出。所有 ReLU 版本的输出 hash 在相同 shape/seed 下完全一致。输出预置 NaN sentinel，未写回的元素会触发检查失败。

覆盖 M×N×K 为 64×32×16/48/64/256、128×128×64/256；在 64×32×48、64×32×256、128×128×64 上追加 seed=2026，主 seed=1。

## 3. 测量口径：核心与 uncore 计数器不同

源码确认 `sim/simx/core.cpp` 中核心 MPM counters 跨 launch 累计保留；`sim/simx/processor.cpp` 的全局内存统计在每次 run/reset 时清零。cache tags 也在 launch reset 时清空。event profiling 的 start/end_ns 则来自主机时钟。

因此每个 kernel 完成后，先读取核心周期/指令和全局内存读写请求，再启动下一 kernel：

- 第一阶段的核心计数从 0 起算，第二阶段核心计数用后一次快照减去前一次快照。
- 内存请求数按各次 launch 的计数相加。
- 整链核心周期必须等于最后一次核心累计快照；分析脚本逐例核对该不变量。

这里报告的是 **模拟设备执行周期之和**，包含每次 kernel 的启动与排空，但不含主机提交间隔。分离 kernel 的缓存状态不跨 launch 保留，因此这些结果不能直接当作真实 GPU 的 warm-cache 端到端延迟。

初版统计曾将累计核心快照直接相加，重复计入 GEMM。发现后按上述源码语义修正，并重新完成全部正式比较；`counter_semantics=core-delta_uncore-per-launch_v2` 是最终记录标识。旧 smoke 和首次 matrix 日志仅用于排错，未混入下表。

## 4. 设备周期对照

下表主 seed=1，周期越低越好。最后两列均为周期变化，负值表示改善。

{table(['M×N×K', '分离 ReLU', '分离 grid-stride', '普通融合', 'PC 融合', '普通融合 vs 优化分离', 'PC vs 普通融合'], performance)}

单 CTA 的 64×32×256，普通融合为 20,825 cycles，PC 为 18,999；第二个 seed 同样为 20,825/18,999。该用例中有较长的 K 循环，可供加载和计算重叠，而且没有第二个 CTA 的并发机会可被额外 loader 挤占。这是与观察一致的解释，尚未完成独立因果分解。

128×128 对应 8 个 CTA：PC 的 5-warp CTA 在当前 8-warp 核心上的 warp 容量上限为 1 CTA，而普通 4-warp CTA 为 2 CTA。多 CTA 下 PC 仍落后，这与上一轮资源对照的方向一致。

## 5. 激活自身开销与普通融合

{table(['M×N×K', '纯 GEMM cycles', '融合新增 cycles', '融合新增指令', '独立 ReLU cycles', 'grid-stride ReLU cycles'], activation)}

普通融合在单 CTA 中额外执行 64 条 warp 指令，在 8 CTA 中增加 512 条，对应每个计算 warp 的 16 个 accumulator 元素各执行一次 fmax。独立 ReLU 需要另外一轮调度、load/store 和 kernel/CTA 管理。grid-stride 明显降低了这些管理成本，但仍需读取中间矩阵并写最终输出。

这里的纯 GEMM 控制和普通融合写入相同最终输出地址。分离执行先写 C，再写 Y，额外缓冲地址可能影响访存时序；所有方法均分配 C/Y，未将少分配一个 buffer 的布局变化混入融合比较。

## 6. 内存流量

数字为 SimX 全局内存计数器的请求数，包含指令、栈和数据请求。不能将全部差值直接解释为中间 tensor 字节，也不据此推算能耗。

{table(['M×N×K', '优化分离 read', '优化分离 write', '普通融合 read', '普通融合 write', 'PC read', 'PC write', '消除的逻辑 C write+read 字节'], traffic)}

融合消除了完整中间 C 的一次写与一次读：64×32 时为 16 KiB，128×128 时为 128 KiB 的逻辑 tensor 传输。实际物理请求还受到缓存、代码和栈访问影响。

128×128×64 中，PC 的读写请求少于普通融合，但周期更长。数据传输、计算重叠和 CTA 并发需要一起评估。

## 7. 下一批值得验证的方向

优先研究 warp 资源分配：保持 issue width、TCU/FPU/LSU 数量与 LMEM 不变，比较两种执行方法在 8/16 hardware warp 配置下的变化，检查多 CTA 差距是否收敛。此实验需要单独构建并重新确认全部有效配置，尚未执行。

独立 activation warp 暂不直接视作收益来源：当前 ReLU 很轻，普通融合已经便宜。后续若加入真正的 activation 阶段，应明确测量交接成本；更重的算子链或 GEMM→activation→GEMM 才可能提供更多可隐藏的工作。跨核/L2 和 RTL parity 仍待后续。

## 8. 文件与复现

```bash
python3 'experiments/gemm_relu/run_relu.py' --phase matrix
python3 'experiments/gemm_relu/run_relu.py' --phase activation
python3 'experiments/gemm_relu/run_relu.py' --phase validation
python3 'experiments/gemm_relu/analyze_relu.py'
```

源码镜像在本目录 `gemm_relu/`；`analysis.json` 包含日志路径和汇总，`measurements.csv` 保存逐例数据。正式批次日志目录分别为：

{chr(10).join('- `' + run['logs'] + '`' for run in runs.values())}

本轮是 kernel 软件/运行时 API 层实验；验证范围为 SimX，未声称已通过 RTL、XRT、FPGA 或 PPA 验证。
'''
    (HERE / 'RESULTS.md').write_text(report, encoding='utf-8')
    print(json.dumps(dict(host_runs=len(cases), kernel_launches=launches, performance=performance), ensure_ascii=False))


if __name__ == '__main__':
    main()
