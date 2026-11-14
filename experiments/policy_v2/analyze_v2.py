#!/usr/bin/env python3
"""Compare frozen v1 and two v2 feature ablations on identical new holdouts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import result_path
import csv
import json
from statistics import geometric_mean

from policy_model import HERE,OLD,FIELDS,predict,sha

KINDS=('v1','v2_tk','v2_mnk')


def identity(c):return tuple(c[f] for f in FIELDS)


def label(c):
    family={2:'融合',3:'两阶段'}.get(c['mode'],f"三阶段 A{c['activation_warps']}/U{c['unroll']}")
    return f"{family}, g={c['ctas']}"


def change(x):return f'{100*(x-1):+.2f}%'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
        +['| '+' | '.join(map(str,row))+' |' for row in rows])


def metrics(items):
    ratios=[i['ratio_to_best'] for i in items]
    return dict(cases=len(items),exact=sum(r==1 for r in ratios),within_1pct=sum(r<=1.01 for r in ratios),
        within_5pct=sum(r<=1.05 for r in ratios),geomean_ratio=geometric_mean(ratios),worst_ratio=max(ratios),
        geomean_to_fused=geometric_mean(i['ratio_to_fused'] for i in items),
        faster_than_fused=sum(i['ratio_to_fused']<1 for i in items))


def main():
    p=json.loads((HERE/'experiment_plan.json').read_text(encoding='utf-8'))
    policy=json.loads((HERE/'policy_v2.json').read_text(encoding='utf-8'))
    manifest=json.loads((HERE/'holdout_manifest.json').read_text(encoding='utf-8'))
    assert p['v1_policy_sha256']==sha(OLD/'policy.json')
    assert policy['plan_sha256']==manifest['plan_sha256']==sha(HERE/'experiment_plan.json')
    assert manifest['policy_sha256']==sha(HERE/'policy_v2.json')
    assert all(sha(result_path(n))==h for n,h in policy['training_sources'].items())
    assert len(manifest['predictions'])==len(KINDS)*2*len(p['holdout_shapes'])
    specs=[(w,phase) for w in (8,16) for phase in ('train','policy','oracle','validate')]
    runs={(w,phase):json.loads(result_path(f'policy2_w{w}_{phase}_results.json').read_text()) for w,phase in specs}
    groups,hashes={},{}
    for (w,phase),run in runs.items():
        assert run['status']=='complete' and run['plan_sha256']==manifest['plan_sha256']
        assert run['source_hashes']==p['source_hashes'] and run['binary_sha256']==p['binary_sha256']
        assert run['effective_config']==dict(p['effective_config_16'],VX_CFG_NUM_WARPS=str(w))
        if phase!='train':assert run['policy_sha256']==manifest['policy_sha256']
        for c in run['cases']:
            assert c['passed'] and min(c['coverage'].values())>0 and c['hardware_warps']==w
            assert c['cta_warps']==(5+c['activation_warps'] if c['mode']==4 else c['mode']+2)
            assert c['smem']==(14336 if c['mode']==4 else 6144)
            key=(w,c['m'],c['n'],c['k'],c['seed'])+identity(c)
            groups.setdefault(key,[]).append(c)
            hashes.setdefault((c['m'],c['n'],c['k'],c['seed']),set()).add(c['output_hash'])
        if phase in ('train','oracle'):
            shapes=p['train_shapes'] if phase=='train' else p['holdout_shapes']
            expected={(s['m'],s['n'],s['k'],v['mode'],v['activation_warps'],v['unroll'],g)
                      for s in shapes for v in p['variants'] for g in p['cta_candidates']
                      if g<=(s['m']//64)*(s['n']//32)}
            observed={(c['m'],c['n'],c['k'])+identity(c) for c in run['cases']}
            assert observed==expected and len(observed)==len(run['cases'])
    assert all(len(h)==1 for h in hashes.values()),'Output mismatch'
    assert all(len({tuple(sorted(c['chain'].items())) for c in cs})==1 for cs in groups.values()),'Counter instability'
    evaluation=[];validation=[]
    for w in (8,16):
        for shape in p['holdout_shapes']:
            m,n,k=(shape[f] for f in ('m','n','k'))
            candidates=[c for c in runs[(w,'oracle')]['cases'] if (c['m'],c['n'],c['k'])==(m,n,k)]
            best=min(candidates,key=lambda c:c['chain']['cycles'])
            fused=min((c for c in candidates if c['mode']==2),key=lambda c:c['chain']['cycles'])
            chosen={}
            for kind in KINDS:
                pred=next(c for c in manifest['predictions'] if (c['selector'],c['hardware_warps'],c['m'],c['n'],c['k'])==(kind,w,m,n,k))
                assert identity(pred)==identity(predict(policy,kind,m,n,k,w))
                c=next(c for c in runs[(w,'policy')]['cases'] if (c['m'],c['n'],c['k'])==(m,n,k) and kind in c['selectors'])
                assert identity(c)==identity(pred)
                family=min((x for x in candidates if (x['mode'],x['activation_warps'],x['unroll'])==(c['mode'],c['activation_warps'],c['unroll'])),key=lambda c:c['chain']['cycles'])
                item=dict(selector=kind,hardware_warps=w,m=m,n=n,k=k,choice=c,best=best,fused=fused,family=family,
                    ratio_to_best=c['chain']['cycles']/best['chain']['cycles'],ratio_to_fused=c['chain']['cycles']/fused['chain']['cycles'],
                    cta_ratio=c['chain']['cycles']/family['chain']['cycles'],family_ratio=family['chain']['cycles']/best['chain']['cycles'])
                evaluation.append(item);chosen[identity(c)]=c
            chosen[identity(best)]=best;chosen[identity(fused)]=fused
            for key,c in chosen.items():
                checked=next(x for x in runs[(w,'validate')]['cases'] if (x['m'],x['n'],x['k'])==(m,n,k) and identity(x)==key)
                validation.append([w,f'{m}×{n}×{k}',label(c),c['chain']['cycles'],checked['chain']['cycles']])
    scores={kind:{str(w):metrics([i for i in evaluation if i['selector']==kind and i['hardware_warps']==w]) for w in (8,16)} for kind in KINDS}
    for kind in KINDS:scores[kind]['all']=metrics([i for i in evaluation if i['selector']==kind])
    paired={}
    for first,second in (('v2_tk','v1'),('v2_mnk','v1'),('v2_mnk','v2_tk')):
        ratios=[]
        for a in (i for i in evaluation if i['selector']==first):
            b=next(i for i in evaluation if i['selector']==second and (i['hardware_warps'],i['m'],i['n'],i['k'])==(a['hardware_warps'],a['m'],a['n'],a['k']))
            ratios.append(a['choice']['chain']['cycles']/b['choice']['chain']['cycles'])
        paired[first+'_vs_'+second]=dict(geomean_ratio=geometric_mean(ratios),worst_regression=max(ratios),
            faster=sum(r<1 for r in ratios),slower=sum(r>1 for r in ratios),same=sum(r==1 for r in ratios))
    summary=dict(successful_simx_executions=sum(len(r['cases']) for r in runs.values()),
        phases={f'w{w}_{phase}':len(r['cases']) for (w,phase),r in runs.items()},metrics=scores,paired=paired,
        evaluation=evaluation,validation=validation,plan_sha256=manifest['plan_sha256'],policy_sha256=manifest['policy_sha256'],
        v1_sha256=p['v1_policy_sha256'],training_rows=len(policy['training_rows']),anchors=len(policy['anchors']),
        provenance={f'w{w}_{phase}':r['logs'] for (w,phase),r in runs.items()})
    (HERE/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (HERE/'evaluation.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.writer(out);writer.writerow(['selector','hardware_warps','M','N','K','choice','cycles','grid_best','best_cycles','best_fused_cycles','extra_cycles_pct','cta_extra_pct','family_extra_pct'])
        for i in evaluation:
            writer.writerow([i['selector'],i['hardware_warps'],i['m'],i['n'],i['k'],label(i['choice']),i['choice']['chain']['cycles'],
                label(i['best']),i['best']['chain']['cycles'],i['fused']['chain']['cycles'],100*(i['ratio_to_best']-1),100*(i['cta_ratio']-1),100*(i['family_ratio']-1)])
    with (HERE/'measurements.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.writer(out);writer.writerow(['hardware_warps','phase','M','N','K','seed']+list(FIELDS)+['cycles','instructions','reads','writes','output_hash'])
        for (w,phase),run in runs.items():
            for c in run['cases']:
                writer.writerow([w,phase]+[c[f] for f in ('m','n','k','seed')]+[c[f] for f in FIELDS]+[c['chain'][f] for f in ('cycles','instructions','reads','writes')]+[c['output_hash']])
    write_report(p,policy,manifest,summary)
    print(json.dumps(dict(runs=summary['successful_simx_executions'],metrics=scores,paired=paired)))


def write_report(p,policy,manifest,summary):
    scores=summary['metrics'];evaluation=summary['evaluation']
    metric_rows=[[kind,w,m['exact'],m['within_1pct'],m['within_5pct'],change(m['geomean_ratio']),
                  change(m['worst_ratio']),change(m['geomean_to_fused'])]
                 for kind in KINDS for w,m in scores[kind].items()]
    paired_rows=[[key,change(v['geomean_ratio']),v['faster'],v['slower'],v['same'],change(v['worst_regression'])]
                 for key,v in summary['paired'].items()]
    perf_rows=[];config_rows=[]
    different=0
    for w in (8,16):
        for s in p['holdout_shapes']:
            es=[next(i for i in evaluation if (i['selector'],i['hardware_warps'],i['m'],i['n'],i['k'])==(kind,w,s['m'],s['n'],s['k'])) for kind in KINDS]
            shape=f"{s['m']}×{s['n']}×{s['k']}"
            perf_rows.append([w,shape,es[0]['best']['chain']['cycles']]+[f"{i['choice']['chain']['cycles']} ({change(i['ratio_to_best'])})" for i in es])
            config_rows.append([w,shape]+[label(i['choice']) for i in es]+[label(es[0]['best'])])
            different+=identity(es[1]['choice'])!=identity(es[2]['choice'])
    training_rows=[]
    for a in policy['anchors']:
        if any((a['m'],a['n'],a['k'])==(s['m'],s['n'],s['k']) for s in p['train_shapes']):
            training_rows.append([a['hardware_warps'],f"{a['m']}×{a['n']}×{a['k']}",a['candidates'],
                label(a['choice']),a['choice']['cycles'],a['fastest_cycles']])
    worst=sorted((i for i in evaluation if i['selector']!='v1'),key=lambda i:i['ratio_to_best'],reverse=True)[:6]
    diagnostic_rows=[[i['selector'],i['hardware_warps'],f"{i['m']}×{i['n']}×{i['k']}",label(i['choice']),
                     change(i['ratio_to_best']),change(i['cta_ratio']),change(i['family_ratio'])] for i in worst]
    status=[]
    for kind in ('v2_tk','v2_mnk'):
        a,b=scores[kind]['all'],scores['v1']['all']
        status.append(f"- {kind}：本组平均额外周期由 {100*(b['geomean_ratio']-1):.2f}% 变为 {100*(a['geomean_ratio']-1):.2f}%，"
                      f"最坏额外周期由 {100*(b['worst_ratio']-1):.2f}% 变为 {100*(a['worst_ratio']-1):.2f}%。")
    status_text='\n'.join(status)
    report=f'''# Project 20：v2 补充训练与几何特征对照（2026-11-02）

本批 {summary['successful_simx_executions']} 次 timing SimX 检查均通过。比较保持冻结的 v1，以及相同扩充训练数据上的 v2-TK、v2-MNK。三种规则在同一批新形状上评估，未按测试结果逐形状挑选规则赢家。

{status_text}

这些是同一组 12 个新形状/硬件组合的比较，不能直接拿上一批 v1 的 1.60%/7.88% 与这批 v2 比较。

## 1. 实验控制与数据分离

- 原样复用 Vortex v3.0 commit `7fe19369587c087f902c3e2bc2add612aa910a95`、工具链 v3.0.1、`activation_balance` 应用及 device binary。
- 1 核、32 lanes、issue width=4、4 TCU blocks、ALU/LSU/FPU 各 1 block、64 KiB LMEM、LSU pending=16、AMO RS=8、L2/L3 关闭；8/16 hardware warp 是完整配置中唯一变化。
- 普通融合/两阶段为 4/5 warp、6 KiB SMEM；三阶段 A1/U4、A2/U4 为 6/7 warp、14 KiB。保持 A/B 深度2、C深度1和跨输出预取、ready/free 及尾部排空协议。
- 新训练点与新验证形状在首次新训练运行前写入 `experiment_plan.json`。训练完成后冻结两个 v2 规则和全部预测，再运行新验证。v1 文件与结果保留。

计划 SHA256：`{summary['plan_sha256']}`；v2 SHA256：`{summary['policy_sha256']}`；v1 SHA256：`{summary['v1_sha256']}`。

v2 训练集由 v1 的 92 条历史测量、上一批六形状的 176 条网格测量，以及本批 96 条补充训练组成，共 {summary['training_rows']} 条、{summary['anchors']} 个精确 `(hardware warps,M,N,K)` 锚点。上一批测试形状现在是训练数据，不再算 v2 独立验证。

补充训练为 128×128×96、128×128×128、256×128×128，各在 8/16-warp 下比较四种执行方式和 g=1/2/4/8。

{table(['hardware warps','训练形状','候选数','1%资源取舍后的配置','配置周期','最快周期'],training_rows)}

## 2. 两个预先指定的 v2 规则

两个规则使用完全相同的锚点和候选选择：每个精确形状内，最快周期 1% 以内优先少 CTA warp、少 SMEM、低周期、少 CTA 数。这个阈值是人工资源取舍，不是统计显著性或 PPA 判据。

- v2-TK：距离 `abs(log2(T/Ta)) + abs(log2(K/Ka))`，其中 T 为输出 tile 数。
- v2-MNK：距离 `abs(log2(M/Ma)) + abs(log2(N/Na)) + abs(log2(K/Ka))`，各维等权。
- 距离相同时，两者都按锚点 T、K、M、N 递增取第一个。CTA 数限制到不超过输出 tile 数的合法 1/2/4/8。

锚点先按精确 M/N/K 分组，避免把不同形状的绝对周期混在一起取最小值。v2-TK 在特征距离中仍不区分相同 T/K 的不同长宽比；v2-MNK 才显式使用几何距离。这是小型查表启发式，没有训练连续性能模型或自动拆分 kernel。

## 3. 本批新的独立验证形状

`128×96×128`、`192×64×128`、`256×96×144`、`192×128×112`、`384×96×160`、`256×192×224`，分别测 8/16 hardware warp。前两个具有相同 T/K、不同 M/N；K=112/144/160/224 不与补充训练 K 重合。

全部形状均未参与 v2 建表，维度仍与完整 tile 对齐。K=112/144 包含奇数个 K tile，测试沿用原输入队列协议。本轮没有部分边界 tile、真实模型 tensor 或跨核通信。

三种规则先执行各自冻结配置；相同配置去重运行，结果同时归属相应规则。随后扫描预先指定的四种执行方式 × g=1/2/4/8（过滤 g>T），共176个评估用网格点。网格最小值是有限候选最优，不是所有实现的全局最优。

## 4. 同一批形状上的选择质量

每个规则每套硬件为6个组合，all为12个。所有几何均值按组合等权，额外周期定义为 `选择周期/候选网格最小周期-1`。

{table(['规则','hardware warps','恰为最快','1%内','5%内','几何平均额外周期','最坏额外周期','相对最佳普通融合几何周期变化'],metric_rows)}

三种规则的逐组合结果，括号内为相对网格最优的额外周期：

{table(['hardware warps','M×N×K','网格最优 cycles','v1 cycles','v2-TK cycles','v2-MNK cycles'],perf_rows)}

对应配置如下。三阶段的 U 均为4：

{table(['hardware warps','M×N×K','v1','v2-TK','v2-MNK','网格最快'],config_rows)}

## 5. 数据补充与特征变化的配对比较

下面直接比较相同形状的周期比；负值表示前一个规则更快。v2-TK vs v1 主要衡量补充数据后的变化，v2-MNK vs v2-TK 在相同扩充数据上比较特征距离。v1 的原锚点没有同 T/K 的不同训练形状；新数据中的几何重复点分别保留为精确锚点。

{table(['配对','几何周期变化','更快组合','更慢组合','相同组合','最大配对回退'],paired_rows)}

v2-TK 与 v2-MNK 在本组只有 {different}/12 个组合的预测配置不同。几何特征的本轮证据因此很有限，不能据此宣布普遍更优或无效。所有规则和权重保持冻结，没有根据测试调整阈值或回退路径。

本组主要观察：

- 补充数据后的 v2-TK 在 7/12 个组合中比 v1 更快，4 个相同，1 个小幅回退。回退点为 16-warp、256×96×144，周期从 v1 的 110,772 变为 111,381，约 +0.55%。
- 8-warp、192×64×128 中，v2-TK 选普通融合 g=2，为 54,459 cycles；v2-MNK 根据更接近的几何锚点选 g=4，为 56,841，慢约 4.37%。几何接近不自动意味着 CTA 数可以直接照搬。
- v2-TK 相对调优普通融合的几何周期比约为 1.00006，本组总体几乎持平。当前得到的是更稳定地接近候选最优的配置选择，尚无额外整体加速优势的证据。

## 6. v2 误选诊断

以下列出两个 v2 规则额外周期最大的六条记录。CTA项保持执行方式不变，只在网格内重选 g；方式项是该方式即使选了最佳 g 后距全网格最优的差距。两者为乘法关系，百分比不能直接相加。

{table(['规则','hardware warps','M×N×K','选择','总额外周期','CTA选择项','执行方式项'],diagnostic_rows)}

这些解释来自测试后的网格，不会回填本轮规则。若据此改 v3，需要新的独立形状；本轮六个形状一旦用于改规则，就应转为训练/诊断数据。

## 7. 验证与范围

所有运行检查精确 CPU reference、正/负/零覆盖、NaN sentinel。相同矩阵/seed 在各方法、规则和硬件配置下输出 hash 一致。规则配置在网格中重跑时，核心周期、指令和全局内存请求数一致。所有选定规则、网格最快、最佳普通融合的去重配置又用 seed=2026 检查：

{table(['hardware warps','M×N×K','配置','seed1 cycles','seed2026 cycles'],summary['validation'])}

这里只报告模拟设备执行周期，不含 host 查表、文件读取和提交开销。评估网格只用于离线研究，不是查表规则运行时必需的测量成本。资源取舍没有转换成面积、功耗或频率收益。

当前仍是单核 timing SimX、固定 tile/数据类型/执行单元配置。本轮未修改上游应用、RTL 或 SimX，没有新增 RTL/XRT、FPGA、PPA、自动阶段切分/RFQ 或跨核队列/KMU placement 结论。

## 8. 文件与复现

```bash
python3 'experiments/policy_v2/policy_model.py'
python3 'experiments/policy_v2/run_v2.py' --phase train --warps 8
python3 'experiments/policy_v2/run_v2.py' --phase train --warps 16
python3 'experiments/policy_v2/policy_model.py' --freeze
python3 'experiments/policy_v2/run_v2.py' --phase policy --warps 8
python3 'experiments/policy_v2/run_v2.py' --phase policy --warps 16
python3 'experiments/policy_v2/run_v2.py' --phase oracle --warps 8
python3 'experiments/policy_v2/run_v2.py' --phase oracle --warps 16
python3 'experiments/policy_v2/run_v2.py' --phase validate --warps 8
python3 'experiments/policy_v2/run_v2.py' --phase validate --warps 16
python3 'experiments/policy_v2/analyze_v2.py'
```

计划/模型/预测清单拒绝覆盖不同内容。`evaluation.csv` 是逐规则/组合结果，`measurements.csv` 保存全部运行，`analysis.json` 保存配置、指标与日志路径。每批日志保留源码、device binary/dump、完整配置和冻结文件。日志目录用 UTC 命名，本报告日期采用用户本地时间。
'''
    (HERE/'RESULTS.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
