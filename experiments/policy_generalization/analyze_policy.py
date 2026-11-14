#!/usr/bin/env python3
"""Evaluate the frozen table against diagnostic holdout grids without updating it."""

import csv
import json
from statistics import geometric_mean,median

from freeze_policy import HERE,COURSE,CONFIG_FIELDS,predict,sha


def name(c):
    stage={2:'融合',3:'两阶段'}.get(c['mode'],f"三阶段 A{c['activation_warps']}/U{c['unroll']}")
    return f"{stage}, g={c['ctas']}"


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
        +['| '+' | '.join(map(str,row))+' |' for row in rows])


def percentage(ratio):
    return f'{100*(ratio-1):+.2f}%'


def config_id(c):
    return tuple(c[f] for f in CONFIG_FIELDS)


def main():
    policy=json.loads((HERE/'policy.json').read_text(encoding='utf-8'))
    manifest=json.loads((HERE/'holdout_manifest.json').read_text(encoding='utf-8'))
    assert sha(HERE/'policy.json')==manifest['policy_sha256']
    assert all(sha(COURSE/n)==h for n,h in policy['training_sources'].items())
    specs=[(w,p) for w in (8,16) for p in ('policy','oracle','validate')]
    runs={(w,p):json.loads((COURSE/f'policy_w{w}_{p}_results.json').read_text()) for w,p in specs}
    groups,hashes={},{}
    for (w,p),run in runs.items():
        assert run['status']=='complete'
        assert run['policy_sha256']==manifest['policy_sha256'] and run['manifest_sha256']==sha(HERE/'holdout_manifest.json')
        assert run['source_hashes']==policy['source_hashes'] and run['binary_sha256']==policy['binary_sha256']
        assert run['effective_config']==dict(policy['effective_config_16'],VX_CFG_NUM_WARPS=str(w))
        for c in run['cases']:
            assert c['passed'] and min(c['coverage'].values())>0 and c['hardware_warps']==w
            assert c['cta_warps']==(5+c['activation_warps'] if c['mode']==4 else c['mode']+2)
            assert c['smem']==(14336 if c['mode']==4 else 6144)
            key=(w,c['m'],c['n'],c['k'],c['seed'])+config_id(c)
            groups.setdefault(key,[]).append(c)
            hashes.setdefault((c['m'],c['n'],c['k'],c['seed']),set()).add(c['output_hash'])
    assert all(len(h)==1 for h in hashes.values()), 'Output mismatch'
    assert all(len({tuple(sorted(c['chain'].items())) for c in cs})==1 for cs in groups.values()), 'Repeated configuration changed counters'
    evaluation=[]
    validation=[]
    for w in (8,16):
        for shape in manifest['shapes']:
            m,n,k=(shape[f] for f in ('m','n','k'))
            selected=next(c for c in runs[(w,'policy')]['cases'] if (c['m'],c['n'],c['k'])==(m,n,k))
            expected=next(c for c in manifest['predictions'] if (c['hardware_warps'],c['m'],c['n'],c['k'])==(w,m,n,k))
            assert config_id(selected)==config_id(expected)==config_id(predict(policy,m,n,k,w))
            oracle_cases=[c for c in runs[(w,'oracle')]['cases'] if (c['m'],c['n'],c['k'])==(m,n,k)]
            expected_grid={config_id(c) for c in manifest['oracle_candidates'] if c['ctas']<=shape['tiles']}
            assert len(oracle_cases)==len(expected_grid) and {config_id(c) for c in oracle_cases}==expected_grid
            global_best=min(oracle_cases,key=lambda c:c['chain']['cycles'])
            fused=min((c for c in oracle_cases if c['mode']==2),key=lambda c:c['chain']['cycles'])
            two=min((c for c in oracle_cases if c['mode']==3),key=lambda c:c['chain']['cycles'])
            family=min((c for c in oracle_cases if (c['mode'],c['activation_warps'],c['unroll'])==(selected['mode'],selected['activation_warps'],selected['unroll'])),key=lambda c:c['chain']['cycles'])
            cyc=selected['chain']['cycles']
            item=dict(hardware_warps=w,m=m,n=n,k=k,tiles=shape['tiles'],choice=selected,best=global_best,
                fused=fused,two=two,family_best=family,policy_ratio_to_best=cyc/global_best['chain']['cycles'],
                policy_ratio_to_fused=cyc/fused['chain']['cycles'],
                cta_selection_ratio=cyc/family['chain']['cycles'],
                family_selection_ratio=family['chain']['cycles']/global_best['chain']['cycles'])
            evaluation.append(item)
            choices={config_id(c):c for c in (selected,global_best,fused)}
            for identity,c in choices.items():
                repeat=next(x for x in runs[(w,'validate')]['cases'] if (x['m'],x['n'],x['k'])==(m,n,k) and config_id(x)==identity)
                validation.append([w,f'{m}×{n}×{k}',name(c),c['chain']['cycles'],repeat['chain']['cycles']])
    def aggregate(items):
        ratios=[i['policy_ratio_to_best'] for i in items]
        fused=[i['policy_ratio_to_fused'] for i in items]
        return dict(cases=len(items),within_1pct=sum(r<=1.01 for r in ratios),within_5pct=sum(r<=1.05 for r in ratios),
            exact_grid_best=sum(r==1 for r in ratios),faster_than_best_fused=sum(r<1 for r in fused),
            geometric_ratio_to_best=geometric_mean(ratios),median_ratio_to_best=median(ratios),worst_ratio_to_best=max(ratios),
            geometric_ratio_to_best_fused=geometric_mean(fused))
    metrics={str(w):aggregate([i for i in evaluation if i['hardware_warps']==w]) for w in (8,16)}
    metrics['all']=aggregate(evaluation)
    summary=dict(successful_simx_executions=sum(len(r['cases']) for r in runs.values()),
        phases={f'w{w}_{p}':len(r['cases']) for (w,p),r in runs.items()},policy_sha256=manifest['policy_sha256'],
        manifest_sha256=sha(HERE/'holdout_manifest.json'),metrics=metrics,evaluation=evaluation,
        validation=validation,provenance={f'w{w}_{p}':r['logs'] for (w,p),r in runs.items()},
        note='Frozen v1 policy is not changed using holdout grid results; grid best is restricted to declared candidates.')
    (HERE/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (HERE/'measurements.csv').open('w',newline='') as out:
        writer=csv.writer(out)
        writer.writerow(['hardware_warps','phase','M','N','K','seed']+list(CONFIG_FIELDS)+['cycles','instructions','reads','writes','output_hash'])
        for (w,p),run in runs.items():
            for c in run['cases']:
                writer.writerow([w,p]+[c[f] for f in ('m','n','k','seed')]+[c[f] for f in CONFIG_FIELDS]
                    +[c['chain'][f] for f in ('cycles','instructions','reads','writes')]+[c['output_hash']])
    with (HERE/'evaluation.csv').open('w',newline='') as out:
        fields=['hardware_warps','M','N','K','policy','policy_cycles','grid_best','grid_best_cycles',
                'tuned_fused_cycles','policy_regret_pct','cta_selection_penalty_pct','family_selection_penalty_pct']
        writer=csv.writer(out);writer.writerow(fields)
        for i in evaluation:
            writer.writerow([i['hardware_warps'],i['m'],i['n'],i['k'],name(i['choice']),i['choice']['chain']['cycles'],
                name(i['best']),i['best']['chain']['cycles'],i['fused']['chain']['cycles'],
                100*(i['policy_ratio_to_best']-1),100*(i['cta_selection_ratio']-1),100*(i['family_selection_ratio']-1)])
    write_report(policy,manifest,summary)
    print(json.dumps(dict(runs=summary['successful_simx_executions'],metrics=metrics,
        rows=[[i['hardware_warps'],i['m'],i['n'],i['k'],name(i['choice']),i['choice']['chain']['cycles'],name(i['best']),
               i['best']['chain']['cycles'],percentage(i['policy_ratio_to_best'])] for i in evaluation]),ensure_ascii=True))


def write_report(policy,manifest,summary):
    evaluation=summary['evaluation']
    metrics=summary['metrics']
    anchors=[[a['hardware_warps'],a['tiles'],a['k'],a['candidates'],name(a['choice']),
              a['choice']['cycles'],a['fastest_cycles']] for a in policy['anchors']]
    rows=[[i['hardware_warps'],f"{i['m']}×{i['n']}×{i['k']}",name(i['choice']),i['choice']['chain']['cycles'],
           name(i['best']),i['best']['chain']['cycles'],percentage(i['policy_ratio_to_best'])] for i in evaluation]
    fused_rows=[[i['hardware_warps'],f"{i['m']}×{i['n']}×{i['k']}",i['choice']['chain']['cycles'],
                 name(i['fused']),i['fused']['chain']['cycles'],percentage(i['policy_ratio_to_fused'])] for i in evaluation]
    metric_rows=[[w,m['cases'],m['exact_grid_best'],m['within_1pct'],m['within_5pct'],
                  percentage(m['geometric_ratio_to_best']),percentage(m['worst_ratio_to_best']),
                  m['faster_than_best_fused'],percentage(m['geometric_ratio_to_best_fused'])]
                 for w,m in metrics.items()]
    diagnostics=[[i['hardware_warps'],f"{i['m']}×{i['n']}×{i['k']}",name(i['family_best']),
                  percentage(i['cta_selection_ratio']),percentage(i['family_selection_ratio'])] for i in evaluation]
    worst=sorted(evaluation,key=lambda i:i['policy_ratio_to_best'],reverse=True)[:3]
    observations='\n'.join(f"- {i['hardware_warps']}-warp、{i['m']}×{i['n']}×{i['k']}：规则选 {name(i['choice'])}，"
        f"相对网格最优多 {100*(i['policy_ratio_to_best']-1):.2f}% 周期。"
        f"其中，保持执行方式、仅重新选 CTA 数可对应消除 {100*(i['cta_selection_ratio']-1):.2f}% 的相对周期开销；"
        f"该执行方式在最佳 CTA 数下仍比全网格最优多 {100*(i['family_selection_ratio']-1):.2f}%。"
        for i in worst)
    report=f'''# Project 20：冻结配置选择表在新形状上的泛化实验

本批完成 {summary['successful_simx_executions']} 次 timing SimX 数值检查，均通过。先冻结历史配置表与六个未参与建表的形状，再执行规则选择，最后测量固定候选网格评估选择损失。没有用新网格结果更新本版规则。

12 个形状/硬件组合中，规则有 {metrics['all']['exact_grid_best']} 个达到网格最小周期、{metrics['all']['within_5pct']} 个落在最优 5% 以内；相对最优的周期比几何均值为 {metrics['all']['geometric_ratio_to_best']:.4f}，最大额外周期为 {100*(metrics['all']['worst_ratio_to_best']-1):.2f}%。

## 1. 本轮要回答的问题

上一轮为各形状分别扫描 CTA 数，已找到有条件优于普通融合/两阶段的三阶段点。本轮检验：仅利用历史点，能否在新的形状上直接选择执行方式、activation warp 数和 CTA 数，而不是为每个新形状先跑完整扫描。

继续复用固定 Vortex v3.0 提交 `7fe19369587c087f902c3e2bc2add612aa910a95`、工具链 v3.0.1，以及相同 `activation_balance` 应用/device binary。1 核、32 lanes、issue width=4、4 TCU blocks、ALU/LSU/FPU 各 1 block、64 KiB LMEM、LSU pending=16、AMO RS=8、L2/L3 关闭。8/16 hardware warp 之外的完整配置一致。

普通融合/两阶段为 4/5 warp、6 KiB SMEM；三阶段 A1/U4、A2/U4 为 6/7 warp、14 KiB。三阶段 C 队列固定深度 1。没有改变 kernel 或模拟器以适配新测试。

## 2. 冻结规则与训练数据

policy v1 SHA256：`{summary['policy_sha256']}`。

训练数据仅来自此前四份正式记录：activation 的 16-warp matrix、8-warp capacity，以及多 CTA 的 8/16-warp matrix。过滤后共 {len(policy['training_rows'])} 条已有测量、{len(policy['anchors'])} 个 `(hardware warps, tile 数 T, K)` 锚点。

规则如下，在首次新形状运行前写入不可覆盖的 `policy.json` 和 `holdout_manifest.json`：

1. 只在相同 hardware warp 配置中寻找锚点，距离为 `abs(log2(T/Ta)) + abs(log2(K/Ka))`。距离相同时依次取较小的 Ta、Ka。
2. 每个锚点中，把周期处于最快值 1% 以内的候选视为可取舍集合，依次优先少 CTA warp、少 SMEM、低周期、少 CTA 数。
3. 使用锚点的执行方式/activation warp/CTA 数；若 CTA 数超过新形状的 tile 数，向下限制到合法的 1/2/4/8。

1% 是事先设定的资源取舍阈值，不是统计显著性阈值，也不等于测得的 PPA 收益。规则没有 M/N 或 A/B footprint 特征；使用它们仍需后续版本验证。

{table(['hardware warps','T','K','已测候选数','选定配置','选定周期','锚点最快周期'],anchors)}

小形状锚点历史上仅测 g=1，大形状锚点覆盖 g=1/2/4/8。因此锚点候选覆盖并不均衡；当前规则继承了这一限制，不能把它视作训练充分的通用成本模型。

## 3. 未参与建表的形状与候选网格

六个形状为 `64×128×64`、`256×32×64`、`256×64×128`、`192×128×64`、`384×128×256`、`320×128×192`，分别在 8/16 hardware warp 下评估。

它们涵盖较小 tile 数、相同 T/K 的不同 M/N、未见 K=128/192，以及 T=20/24 的范围外延伸。全部维度仍与计算 tile 对齐。这里的“未见”是未参与该版本选择表建构，并非真实应用数据集上的泛化。

诊断网格在冻结的清单中事先指定：普通融合、两阶段、三阶段 A1/U4、A2/U4，各自扫描 g=1/2/4/8，并过滤 g>T 的非法配置。共 176 次网格测量；其最小值称为本报告的 oracle，仅代表这组有限候选的最快值。

两套硬件的规则选择都先执行完，再开始网格扫描。选择表及其训练记录 SHA256 在每批运行前核查；网格数据没有成为规则输入。随后用第二个 seed 复核规则选择、网格最快和调优普通融合的去重配置。

## 4. 冻结规则与范围内最优的差距

周期越低越好；额外周期为 `规则周期 / 网格最快周期 - 1`。

{table(['hardware warps','M×N×K','冻结规则配置','规则 cycles','网格最快配置','最快 cycles','规则额外周期'],rows)}

## 5. 与调好 CTA 数的普通融合比较

普通融合基线也从相同 CTA 候选中取最小周期，不用固定且不利的 CTA 数作为对照。

{table(['hardware warps','M×N×K','规则 cycles','最佳普通融合配置','融合 cycles','规则 vs 融合'],fused_rows)}

以下按每个形状/硬件组合等权计算周期比的几何均值；不是对总周期加权，也不是 benchmark suite 的代表性总体收益。

{table(['hardware warps','组合数','恰为最快','1% 内','5% 内','相对最优几何额外周期','最大额外周期','快于最佳融合数','相对融合几何周期变化'],metric_rows)}

规则在 host 实验脚本中查表后选择既有 kernel entry 和 CTA 数。这里只计单 kernel 的模拟设备周期，没有计入 Python 查表、文件读取或主机提交开销，也没有把 176 次 oracle 调优成本算作可部署规则的一部分。

## 6. 执行方式与 CTA 数的损失分解

诊断一：保持规则选定的执行方式和 activation warp，仅在测试网格里重新选最佳 g。诊断二：这种执行方式即使选了最佳 g，还比全网格最快慢多少。这些都是测试后的解释，不会回填 v1。

{table(['hardware warps','M×N×K','该执行方式的最佳 g','CTA 选择额外周期','执行方式选择额外周期'],diagnostics)}

两个比例满足乘法关系：`规则/最优 = (规则/同方式最佳g) × (同方式最佳g/最优)`，百分比不能直接相加。

`256×64×128` 是两套硬件共有的主要失误点，但损失来源不同：8-warp 下，仅重新选择 g 只能改善约 0.59%，还需要改执行方式；16-warp 下，同一两阶段组织重新选 g 对应约 7.38% 的相对开销，而该组织在最佳 g 下距全网格最优约 0.47%。同一个简单规则失误，可能需要不同的资源相关修正。

三阶段的收益也在未参与建表的 `192×128×64` 上复现：8-warp 网格最佳三阶段为 61,301 cycles，最佳普通融合为 63,759；16-warp 分别为 61,179/65,236。冻结规则在 16-warp 下选择的是稍慢的三阶段 62,330，但仍比调优普通融合少约 4.45% 周期。这支持继续研究适用条件，尚不足以把三阶段作为统一默认方案。

本组差距最大的三个组合：

{observations}

## 7. 正确性、重复与边界

全部输出均检查精确 CPU reference、正/负/零覆盖及 NaN sentinel，相同形状/seed 的输出 hash 跨方法/硬件配置一致。规则选定配置在 oracle 网格中再次运行，其核心周期、指令和全局请求数一致。第二个 seed 的复核如下：

{table(['hardware warps','M×N×K','配置','seed1 cycles','seed2026 cycles'],summary['validation'])}

当前规则仍是很小的历史查表启发式。有限网格、锚点分布、固定 tile/数据类型、没有部分边界 tile、单核和两套特定 warp 配置，都限制了结论。全局请求计数包含指令/数据/栈，不能直接折算能耗。

下一版可针对诊断出的缺口补齐训练候选或加入成本特征，再另选新的验证形状；本批六个形状若用于改规则，就不应继续作为同一新版的独立测试集。当前 v1 保持冻结。

本轮没有新增 CTA trace、RTL/XRT、FPGA 或 PPA 验证，也未实现 WASP 自动阶段切分、RFQ 或跨核队列/KMU placement。

## 8. 文件与复现

```bash
python3 'experiments/policy_generalization/freeze_policy.py'
python3 'experiments/policy_generalization/run_policy.py' --phase policy --warps 8
python3 'experiments/policy_generalization/run_policy.py' --phase policy --warps 16
python3 'experiments/policy_generalization/run_policy.py' --phase oracle --warps 8
python3 'experiments/policy_generalization/run_policy.py' --phase oracle --warps 16
python3 'experiments/policy_generalization/run_policy.py' --phase validate --warps 8
python3 'experiments/policy_generalization/run_policy.py' --phase validate --warps 16
python3 'experiments/policy_generalization/analyze_policy.py'
```

`policy.json` 与 `holdout_manifest.json` 为冻结内容，freeze 脚本拒绝覆盖不同版本；`evaluation.csv` 是逐组合选择质量，`measurements.csv` 是所有运行，`analysis.json` 保存汇总与日志路径。每批日志保留源码、device binary/dump、完整有效配置、策略与测试清单。
'''
    (HERE/'RESULTS.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    main()
