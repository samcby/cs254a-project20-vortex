#!/usr/bin/env python3
"""Validate CTA-count sweeps and reconstruct logical CTA work and residency."""

import csv
import gzip
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import log_path, result_path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'warp_capacity'))
from trace_residency import parse as residency

FIELDS = ('m', 'n', 'k', 'ctas', 'mode', 'activation_warps', 'unroll', 'c_depth', 'seed')
VARIANTS = [(2, 0), (3, 0), (4, 1), (4, 2)]


def local(path):
    return log_path(path)


def label(mode, aw):
    return {2: '普通融合', 3: '两阶段'}.get(mode, f'三阶段 A{aw}/U4')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
        + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def work(case):
    dispatch = re.compile(r'dispatch CTA warp: cid=0, wid=(\w+), cta_id=(\w+), rank=(\w+)/(\w+).*blockIdx=\((\w+),(\w+)\)')
    exit_ = re.compile(r'core0.*warp-state: wid=(\w+), tmask=([01]+)\s*$')
    tcu = re.compile(r'TRACE\s+(\d+): .*tcu execute: op=WGMMA,.*wid=(\d+),.*rd=f')
    fpu = re.compile(r'TRACE\s+(\d+): .*fpu execute: op=FMIN_MAX,.*wid=(\d+),')
    active, compute, activation = {}, {}, {}
    transfers = 0
    with gzip.open(local(case['trace']), 'rt', errors='replace') as src:
        for line in src:
            m = dispatch.search(line)
            if m:
                wid, slot, rank, size, x, y = map(int, m.groups())
                assert y == 0 and wid not in active
                active[wid] = (x, rank)
            m = tcu.search(line)
            if m:
                identity = active[int(m[2])]
                compute.setdefault(identity, []).append(int(m[1]))
            m = fpu.search(line)
            if m:
                identity = active[int(m[2])]
                activation.setdefault(identity, []).append(int(m[1]))
            if re.search(r'dxa-core\[\d+\] start:', line):
                transfers += 1
            m = exit_.search(line)
            if m and int(m[2], 2) == 0:
                active.pop(int(m[1]))
    assert not active
    per_compute = (case['k']//16)*32
    act_ranks = list(range(5, 5+case['activation_warps'])) if case['mode'] == 4 else list(range(4))
    per_activation = 64//len(act_ranks)
    assert set(compute) == {(cta, r) for cta in range(case['ctas']) for r in range(4)}
    assert set(activation) == {(cta, r) for cta in range(case['ctas']) for r in act_ranks}
    overlap = 0
    same_tick = 0
    rows = []
    for cta, count in enumerate(case['tiles_per_cta']):
        assert all(len(compute[(cta,r)]) == count*per_compute for r in range(4))
        assert all(len(activation[(cta,r)]) == count*per_activation for r in act_ranks)
        groups = []
        for ordinal in range(count):
            c = [t for r in range(4) for t in compute[(cta,r)][ordinal*per_compute:(ordinal+1)*per_compute]]
            a = [t for r in act_ranks for t in activation[(cta,r)][ordinal*per_activation:(ordinal+1)*per_activation]]
            if case['mode'] == 4:
                assert min(a) > max(c), 'Activation before complete K reduction'
            else:
                for rank in range(4):
                    assert min(activation[(cta,rank)][ordinal*per_activation:(ordinal+1)*per_activation]) > max(compute[(cta,rank)][ordinal*per_compute:(ordinal+1)*per_compute])
            groups.append((c, a))
            rows.append(dict(cta=cta, ordinal=ordinal, output_tile=cta+ordinal*case['ctas'],
                compute_start=min(c), compute_end=max(c), activation_start=min(a), activation_end=max(a)))
        for ordinal, (c, a) in enumerate(groups[:-1]):
            later = groups[ordinal+1][0]
            overlap += sum(min(later) <= t <= max(later) for t in a)
            same_tick += len(set(a) & set(later))
    assert transfers == case['output_tiles']*(case['k']//16)*2
    return dict(dxa_transfers=transfers, wgmma_events=sum(map(len, compute.values())),
        relu_events=sum(map(len, activation.values())), activation_events_in_next_compute_window=overlap,
        same_tick_with_next_compute=same_tick, rows=rows)


def main():
    specs = [(8, 'smoke')] + [(w,p) for w in (8,16) for p in ('matrix','validation','repeat','trace','best')] + [(16,'besttrace')]
    runs = {(w,p): json.loads(result_path(f'multicta_w{w}_{p}_results.json').read_text()) for w,p in specs}
    previous = json.loads(result_path('activation_w16_matrix_results.json').read_text())
    groups, hashes, matrix_cases = {}, {}, {}
    for (w,p), run in runs.items():
        assert run['status'] == 'complete', (w,p,run['status'])
        assert run['source_hashes'] == previous['source_hashes'] and run['binary_sha256'] == previous['binary_sha256']
        assert run['effective_config'] == dict(previous['effective_config'], VX_CFG_NUM_WARPS=str(w))
        for c in run['cases']:
            assert c['passed'] and min(c['coverage'].values()) > 0 and c['hardware_warps'] == w
            assert c['cta_warps'] == (5+c['activation_warps'] if c['mode']==4 else c['mode']+2)
            assert c['smem'] == (14336 if c['mode']==4 else 6144)
            counts = [1+(c['output_tiles']-1-i)//c['ctas'] for i in range(c['ctas'])]
            assert c['output_tiles'] == (c['m']//64)*(c['n']//32) and c['tiles_per_cta'] == counts
            assert sum(counts)==c['output_tiles'] and min(counts)>0
            key = (w,)+tuple(c[n] for n in FIELDS)
            groups.setdefault(key, []).append(c)
            hashes.setdefault(tuple(c[n] for n in ('m','n','k','seed')), set()).add(c['output_hash'])
            if p=='matrix':
                matrix_cases[(w,c['k'],c['ctas'],c['mode'],c['activation_warps'])] = c
    assert all(len(h)==1 for h in hashes.values()), 'Output mismatch'
    assert all(len({tuple(sorted(c['chain'].items())) for c in cs})==1 for cs in groups.values()), 'Counter instability'
    matrix_rows, best_rows, best_validation, repeat_rows = [], [], [], []
    for w in (8,16):
        for k in (64,256):
            for g in (1,2,4,8):
                cs = [matrix_cases[(w,k,g,v,a)] for v,a in VARIANTS]
                matrix_rows.append([w,k,g,16//g]+[c['chain']['cycles'] for c in cs])
            bests = [min((matrix_cases[(w,k,g,v,a)] for g in (1,2,4,8)), key=lambda c:c['chain']['cycles']) for v,a in VARIANTS]
            tri = min(bests[2:], key=lambda c:c['chain']['cycles'])
            best_rows.append([w,k]+[f"{c['chain']['cycles']} (g={c['ctas']})" for c in bests]
                +[f"{100*(tri['chain']['cycles']/bests[0]['chain']['cycles']-1):+.2f}%",f"{100*(tri['chain']['cycles']/bests[1]['chain']['cycles']-1):+.2f}%"])
            for c in bests:
                seed2=next(x for x in runs[(w,'best')]['cases'] if (x['k'],x['mode'],x['activation_warps'])==(k,c['mode'],c['activation_warps']))
                assert seed2['ctas']==c['ctas']
                best_validation.append([w,k,label(c['mode'],c['activation_warps']),c['ctas'],c['chain']['cycles'],seed2['chain']['cycles']])
        for g in (4,8):
            for v,a in VARIANTS:
                c=matrix_cases[(w,64,g,v,a)]
                cs=groups[(w,)+tuple(c[n] for n in FIELDS)]
                vals=[x['chain']['cycles'] for x in cs]
                repeat_rows.append([w,g,label(v,a),len(vals),min(vals),max(vals)])
    traces=[]
    for (w,p), run in runs.items():
        if p not in ('trace','besttrace'):
            continue
        for c in run['cases']:
            res, rows = residency(local(c['trace']), w, c['cta_warps'], c['ctas'])
            events=work(c)
            stem=f"w{w}_g{c['ctas']}_v{c['mode']}_a{c['activation_warps']}"
            for suffix, data in (('residency',rows),('activity',events['rows'])):
                with (HERE/f'{stem}_{suffix}.csv').open('w',newline='') as out:
                    writer=csv.DictWriter(out,fieldnames=data[0].keys())
                    writer.writeheader(); writer.writerows(data)
            events.pop('rows')
            traces.append(dict(warps=w,ctas=c['ctas'],mode=c['mode'],activation_warps=c['activation_warps'],
                residency=res,work=events,trace=c['trace']))
    peaks=[[t['warps'],t['ctas'],label(t['mode'],t['activation_warps'])]+[t['residency']['peak'][n] for n in ('live_warps','started_live_ctas','all_warps_live_ctas')] for t in traces]
    work_rows=[[t['warps'],t['ctas'],label(t['mode'],t['activation_warps'])]+[t['work'][n] for n in ('dxa_transfers','wgmma_events','relu_events','activation_events_in_next_compute_window','same_tick_with_next_compute')] for t in traces]
    summary=dict(successful_simx_executions=sum(len(r['cases']) for r in runs.values()),
        phases={f'w{w}_{p}':len(r['cases']) for (w,p),r in runs.items()},matrix=matrix_rows,best=best_rows,
        best_seed_validation=best_validation,repeats=repeat_rows,traces=traces,
        source_hashes=previous['source_hashes'],binary_sha256=previous['binary_sha256'],
        provenance={f'w{w}_{p}':r['logs'] for (w,p),r in runs.items()})
    (HERE/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (HERE/'measurements.csv').open('w',newline='') as out:
        writer=csv.writer(out)
        writer.writerow(['hardware_warps','phase']+list(FIELDS)+['cycles','instructions','reads','writes','smem_bytes','output_hash'])
        for (w,p),run in runs.items():
            for c in run['cases']:
                writer.writerow([w,p]+[c[n] for n in FIELDS]+[c['chain'][n] for n in ('cycles','instructions','reads','writes')]+[c['smem'],c['output_hash']])
    write_report(summary,matrix_rows,best_rows,best_validation,repeat_rows,peaks,work_rows)
    print(json.dumps(dict(runs=summary['successful_simx_executions'],best=best_rows,peaks=peaks),ensure_ascii=False))


def write_report(summary,matrix_rows,best_rows,best_validation,repeat_rows,peaks,work_rows):
    report=f'''# Project 20：多个 persistent CTA 的工作分配实验

本批 {summary['successful_simx_executions']} 次 timing SimX 运行全部通过数值检查。复用上一轮相同源码和 device binary，只改变总矩阵、K、CTA 数与 8/16 hardware warp 配置。三阶段的收益取决于阶段 warp 分配和 CTA 数的共同选择；本批出现优于两阶段的实测点，也保留了明显退化的配置。

## 1. 实验问题与控制条件

主矩阵固定 M=256、N=128，共 16 个 64×32 输出 tile。K=64/256，CTA 数 g=1/2/4/8，每 CTA 分别连续处理 16/8/4/2 个输出 tile。CTA i 处理 i、i+g、i+2g 等 tile，因此不会把流水线缩成每 CTA 只有一个输出。

- Vortex v3.0 固定提交 `7fe19369587c087f902c3e2bc2add612aa910a95`，工具链 v3.0.1。
- 1 核、32 lanes、issue width=4、4 TCU blocks、ALU/LSU/FPU 各 1 block、64 KiB LMEM；LSU pending=16、AMO RS=8、L2/L3 关闭。8/16 两套完整有效配置只有 NUM_WARPS 不同。
- 原样复用 `tests/regression/activation_balance`，没有修改应用、RTL 或 SimX 源码。全部运行的源码和 device binary SHA256 与上一轮一致：`{summary['binary_sha256']}`。
- 普通融合使用 4 warp、6 KiB SMEM；两阶段使用 4 compute + 1 loader、6 KiB；三阶段 A1/U4 使用 4 compute + 1 loader + 1 activation，A2/U4 使用 2 activation，总计 6/7 warp、14 KiB。A/B 深度=2、C 深度=1。
- 普通融合保留跨输出 tile 的预取。两阶段的 ReLU 留在 compute accumulator 内，三阶段经共享内存 C 队列交接。同步包含原有 BAR 的 LSU drain 及尾部排空。

改变 g 同时改变 CTA 管理次数、每 CTA 流水线长度、tile 遍历顺序、缓存复用和驻留竞争。这是实际调度方案的整体对照，不能把全部周期变化单独归因于 occupancy。增加 hardware warp 也有上下文与调度状态容量成本，本实验没有做等面积比较。

## 2. 正确性与进展

全部用例检查精确 CPU reference、正/负/零输入覆盖、NaN sentinel 和最终输出 hash。相同矩阵/seed 下，各执行方式、CTA 数及硬件 warp 配置的 hash 一致。

另测 9 个输出 tile 分给 4 个 CTA（3/2/2/2），以及 15 个 tile 分给 8 个 CTA（2/2/2/2/2/2/2/1），覆盖不均匀分配、奇数队列代数和最后一个 CTA 只有一个输出的排空情况。矩阵维度仍与 tile 对齐，没有测试部分边界 tile。

主扫描 64 例；多 CTA smoke 4 例；不均匀分配/seed 验证 24 例；关键重复 32 例；扫描最佳 CTA 数的第二 seed 复核 16 例；debug trace 8 例。相同配置的正式重复和 debug 周期/指令/全局请求数均一致。

## 3. 固定总工作量的完整扫描

周期越低越好。A1/A2 均为循环展开四组的三阶段。

{table(['hardware warps','K','CTA 数 g','每 CTA tiles','普通融合','两阶段','三阶段 A1/U4','三阶段 A2/U4'],matrix_rows)}

## 4. 给每种方法选择扫描范围内的最佳 CTA 数

以下各方法分别从 g=1/2/4/8 选择最小周期；括号给出对应 g。最后两列选择 A1/A2 中更快的三阶段，分别与最佳普通融合、最佳两阶段比较。负值表示三阶段周期更少。

{table(['hardware warps','K','最佳普通融合','最佳两阶段','最佳 A1/U4','最佳 A2/U4','最佳三阶段 vs 普通','最佳三阶段 vs 两阶段'],best_rows)}

主要观察：

- 8-warp、K=64：最佳三阶段为 A2/U4、g=1，80,797 cycles；最佳普通融合为 g=4、85,148。当前三阶段可以获得小幅实际收益，但增加到更多 CTA 会明显退化。
- 16-warp、K=64：最佳三阶段切换为 A1/U4、g=2，80,423 cycles，比最佳两阶段 83,192 少约 3.33%，比最佳普通融合 85,132 少约 5.53%。当前单 CTA 的最佳 activation 分配并非多 CTA 下始终最佳。
- 8-warp、K=256：最佳普通融合 242,342 cycles，仍快于最佳三阶段 267,631。单看 g=1 时普通融合的 381,134，会夸大流水化的相对优势；需要允许普通融合也选择合适的 CTA 数。
- 16-warp、K=256：最佳三阶段 236,663，与最佳两阶段 237,088 仅相差约 0.18%，应视为本批中很接近的结果，不作稳定显著优势的泛化结论。

这些最小值仅是本次扫描范围内的最佳点，不能当作全局最优调度策略。与上一轮 5 个输出 tile 相比，本批 M/N、footprint 与 tile 序列长度也变了，不能把新收益单独解释为“序列更长”。

## 5. 第二个 seed 复核最佳点

根据 seed=1 的完整矩阵选出每种方法的最佳 g，再用 seed=2026 复核，不重新为第二个 seed 挑选 g。

{table(['hardware warps','K','方法','选择的 g','seed1 cycles','seed2026 cycles'],best_validation)}

## 6. 实际 CTA 驻留和工作量

使用既有 dispatch 与 tmask=0 退出事件重建逻辑 CTA 生命周期，按 blockIdx 识别逻辑 CTA，允许硬件 slot ID 被后续 CTA 复用。每次分配/退出均配对、所有 rank 均出现，最后没有残留 warp。

{table(['hardware warps','g','方法','峰值 live warp','峰值已开始未结束 CTA','峰值全部 warp 同时存活 CTA'],peaks)}

dispatcher 逐 warp 分配，因此可以同时存在只入驻部分 warp 的 CTA。上表各峰值可能发生于不同时刻，不能当作时间平均 occupancy。dispatch 日志没有精确 tick，CSV 中 preceding_trace_cycle 仅用于相邻 trace 定位。

在 16-warp、g=2 下，A1 与 A2 都有两个 CTA 的全部 warp 同时存活，分别使用 12/14 个 warp。A1 的较低周期不能仅靠“完整驻留 CTA 更多”解释；不同阶段的调度、发射与访存竞争仍需进一步区分。

按物理 warp 当前所属的逻辑 CTA/rank 收集 WGMMA 与 FMIN_MAX，检查每个 CTA 的 tile 数、K 次数和所有参与者。以下全部是 M=256、N=128、K=64，因此工作量应相同。

{table(['hardware warps','g','方法','DXA transfers','WGMMA events','ReLU events','下一输出计算窗口内 activation events','按 CTA 累计同 tick 交集'],work_rows)}

三阶段所有输出的 activation 均在该输出完整 K 归约后开始；普通融合/两阶段检查每个计算 warp 完成自己的 K 后才处理其输出行。跨输出重叠仅在同一逻辑 CTA 内配对，避免把两个 CTA 的独立执行误判成该 CTA 的流水重叠。

事件窗口与同 tick 交集是重叠证据，不是利用率或纯 FPU 延迟。全局内存请求计数包含数据、指令与栈，不能直接解释为 tensor 字节、带宽或能耗。尚未单独量化共享内存、LSU drain、缓存与调度的各项贡献。

## 7. 重复运行

主矩阵、两次额外重复以及对应 debug 运行的周期范围如下；第二个 seed 在上节单独列出。

{table(['hardware warps','g','方法','相同输入观测次数','min cycles','max cycles'],repeat_rows)}

## 8. 当前结论与下一步边界

核内多 CTA 和不均匀 persistent tile 分配在本批测试中均正确完成。三阶段在一些配置中已经超过两阶段强基线，但收益有条件：阶段分配、CTA 数和硬件 warp 预算必须一起考虑。当前证据不支持固定采用更多 activation warp 或更多 CTA。

下一批可先形成小型选择表：根据 K、总 tile 数与 hardware warp 容量选择普通融合/两阶段/三阶段及 g，再用未参与选择的新形状检查泛化。若转向跨核队列，应单独验证原子队列的内存可见性、进展与 KMU placement，不由本批核内成功推断跨核正确。

本轮没有实现自动编译器/RFQ，没有重跑此前超时的 RTL build，也没有新增 RTL/XRT、FPGA 或 PPA 结论。

## 9. 复现与文件

```bash
python3 'experiments/multi_cta/run_multi.py' --phase smoke --warps 8
python3 'experiments/multi_cta/run_multi.py' --phase matrix --warps 8
python3 'experiments/multi_cta/run_multi.py' --phase matrix --warps 16
python3 'experiments/multi_cta/run_multi.py' --phase finish --warps 8
python3 'experiments/multi_cta/run_multi.py' --phase finish --warps 16
python3 'experiments/multi_cta/run_multi.py' --phase best --warps 8
python3 'experiments/multi_cta/run_multi.py' --phase best --warps 16
python3 'experiments/multi_cta/run_multi.py' --phase besttrace --warps 16
python3 'experiments/multi_cta/analyze_multi.py'
```

finish 顺序执行 validation、repeat、trace；best 依赖对应配置的完整 matrix。`measurements.csv` 保存全部逐例数据，`analysis.json` 保存汇总与日志来源，`*_residency.csv`/`*_activity.csv` 保存逐事件记录。源码、device binary/dump、完整配置和压缩 trace 均在各批日志中保留。
'''
    (HERE/'RESULTS.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    main()
