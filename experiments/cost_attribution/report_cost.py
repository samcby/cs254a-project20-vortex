#!/usr/bin/env python3
"""Generate the source-backed Chinese attribution report after validation."""
import json
from pathlib import Path
import sys
from analyze_cost import HERE,label,table

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from project_paths import result_path


def main():
    result=json.loads((HERE/'analysis.json').read_text(encoding='utf-8'))
    normal={w:json.loads(result_path(f'cost_w{w}_normal_results.json').read_text(encoding='utf-8'))['cases'] for w in (8,16)}
    cap={w:json.loads(result_path(f'cost_capacity_w{w}_normal_results.json').read_text(encoding='utf-8'))['cases'] for w in (8,16)}
    def find(w,v,a=0,r=0,g=None):
        return next(c for c in normal[w] if (c['mode'],c['aw'],c['reserve'],c['ctas'])==(v,a,r,g or (4 if w==8 else 2)))
    def tr(w,v,g=None):return next(t for t in result['traces'] if (t['hardware_warps'],t['mode'],t['ctas'])==(w,v,g or (4 if w==8 else 2)))
    matrix=[]
    for c in normal[16][:12]:
        other=find(8,c['mode'],c['aw'],c['reserve'])
        matrix.append([label(c),c['cta_warps'],c['smem']//1024,c['chain']['cycles'],other['chain']['cycles']])
    capacities=[]
    for w in (8,16):
        for g in ((1,4) if w==8 else (2,)):
            cs=[next(c for c in cap[w] if c['ctas']==g and c['mode']==v) for v in (2,10,11)]
            capacities.append([w,cs[0]['k'],g]+[c['chain']['cycles'] for c in cs]+[f"{100*(cs[2]['chain']['cycles']/cs[1]['chain']['cycles']-1):+.2f}%"])
    overlaps=[]
    for w,a in ((16,1),(8,2),(16,2)):
        yes,no=find(w,4,a),find(w,8,a)
        overlaps.append([w,yes['k'],yes['ctas'],a,yes['chain']['cycles'],no['chain']['cycles'],f"{100*(no['chain']['cycles']/yes['chain']['cycles']-1):+.2f}%"])
    counters=[];sync=[];deps=[];res=[]
    for w in (16,8):
        for v in (2,3,4):
            t=tr(w,v);d=t['detail'];c=find(w,v,1 if w==16 and v==4 else 2 if v==4 else 0)
            counters.append([w,label(c),c['chain']['instructions'],d['bar_instructions'],d['lane_memory_operations'].get('SMEM_C/ST',0),d['lane_memory_operations'].get('SMEM_C/LD',0),d['lane_memory_operations']['Y/ST']])
            sync.append([w,label(c),d['gate_blocked_unique_ticks'],f"{100*d['gate_blocked_unique_ticks']/t['cycles']:.2f}%",d['suspended_warp_cycles'].get('C_ready/activation',0),d['suspended_warp_cycles'].get('C_free/compute',0)])
            q=d['residency']['average']
            res.append([w,label(c),f"{q['live_warps']:.3f}",f"{q['full_compute_ctas']:.3f}",f"{q['full_live_ctas']:.3f}"])
            if v!=2:
                q=d['suspended_warp_cycles_by_dependency']
                deps.append([w,label(c)]+[q.get('input_ready/'+s,0) for s in ('DXA','participants','participants_and_DXA')])
    capres=[]
    for t in result['capacity']['traces']:
        d=t['detail'];q=d['residency']['average']
        capres.append([t['hardware_warps'],label(t),f"{q['full_compute_ctas']:.3f}",f"{q['full_live_ctas']:.3f}",d['wgmma_events'],d['relu_events']])
    text=f'''# Project 20：流水线成本归因（2026-11-03）

本轮把已有结果推进到可检查的解释：8-warp 下的上下文容量门槛有直接对照支持；16-warp 获益配置确实依赖计算与输出阶段的重叠；三阶段增加的共享内存流量和动态工作量已被量化。不同成本会改变调度和访存时序，不能给它们各分配一个固定周期数后相加。

## 1. 研究范围与验证

固定 M=256、N=128，共16个64×32输出tile。重点解释两个已有场景：16 hardware warps、K=64、g=2 的三阶段 A1/U4；8 hardware warps、K=256、g=4 的三阶段 A2/U4。同时保留 g=1 和其他角色分配的少量对照。两组 K/g 不同，因此跨组绝对周期不能用于单独归因 hardware warp 容量。

- 固定 Vortex 提交 `7fe19369587c087f902c3e2bc2add612aa910a95`、toolchain v3.0.1。1核、32 lanes、issue width4、4 TCU blocks、ALU/LSU/FPU各1、LMEM64KiB、LSU pending16、AMO RS8；完整有效配置与前轮一致，两套配置只有 NUM_WARPS 不同。
- 新建独立应用 `cost_pipeline`；容量补充对照使用另一个独立应用 `cost_capacity`。原 `activation_balance` 和受版本控制的上游 SimX/RTL 保持不变。容量对照单独保存 binary；两份 binary 中的普通融合均精确复现旧基线计数。
- 最终 {result['runs']} 次 SimX 运行通过检查：主归因66次、开发smoke3次、容量补充21次。其中20份为完整 debug trace。包括关键配置重复运行和 seed=2026，CPU reference、混合正负/零值、NaN sentinel、同输入输出hash均通过。相同 seed 的正式重复和对应 debug 的周期、指令、全局请求以及18项 core MPM计数一致；第二 seed 的关键周期也一致。
- 独立观察副本 `/root/cs254a/project20/cost-attribution-observer/simx` 只添加日志，记录 BAR gate、barrier状态、dispatch/exit tick。每份 trace 的上述计数均与原 SimX 对应运行一致，观察代码没有改变本批模拟时序。源文件原始/观察 SHA256 见 `observer_sources.json`。
- 主归因 binary SHA256：`{result['binary_sha256']}`；容量对照 binary SHA256：`{result['capacity']['binary_sha256']}`。

## 2. 资源、搬运与重叠的控制实验

以下周期越小越好，所有行的 GEMM、ReLU、最终输出保持一致。输出队列深度1，输入双缓冲。

{table(['控制方式','CTA warp数','SMEM KiB','16w/K64/g2 cycles','8w/K256/g4 cycles'],matrix)}

各对照的含义及限制：

1. **只多预留共享内存**：两阶段仍用5个warp，SMEM从6增到14KiB，但不访问新增区域。两个场景的周期、动态指令和18项core计数完全一致。当前 g/warp 预算下，额外8KiB的容量预留不是性能退化的原因；这不代表其他形状或更多CTA也没有容量影响，更不代表没有面积/能耗成本。
2. **只增加 C 往返路径**：仍用5个warp，计算结果写入C共享内存，经4个compute warp的barrier后，按 WGMMA n-major布局以volatile读回，再在寄存器中ReLU并写Y。trace确认每个输出恰好一写一读。8w周期增加10,483（+3.73%）；16w反而减少1,216（-1.46%）。它测到的是“往返+同步+时序变化”的净效应，不能把前者当作普适的纯SMEM延迟。
3. **两阶段+闲置warp**：join5先测加入末尾join的净变化，再和join6/join7比较。16w从join5到join6/7小幅变慢；8w的join7却比join5更快。因此闲置上下文还会改变调度，不能由warp数量直接预测周期。该对照从5个warp起步，本来就跨过8w的双CTA容量门槛，需要下一节的4→5对照补足。
4. **限制输出阶段重叠**：把等待C槽释放的barrier从完成当前tile计算之后移到开始下一tile计算之前。角色、SMEM、输出操作和barrier次数保持不变，trace确认同一CTA前一tile activation与下一tile计算窗口的交集归零。编译器仍可能改变少量指令和参数回读：16w A1为33,352→33,320条动态指令，其他栈/参数load增加2,048个lane操作。因此这是可执行的调度消融，不是完全不变的机器码或无开销理想开关。

## 3. 容量门槛：普通融合也保留一个闲置 warp

复制普通融合原算法到共享的inline函数，分别以4/5个warp启动；rank0–3执行完全相同的普通融合，新增rank4只在末尾barrier等待。join4/join5都包含相同位置的末尾join，以测量一个额外上下文的影响。SMEM仍为6KiB，没有C队列。

{table(['hardware warps','K','g','原普通融合','普通融合+join4','普通融合+闲置warp+join5','join5 vs join4'],capacities)}

8w、g=4时，4→5 warp带来108,861 cycles、约43.73%的增加，动态指令仅84,424→84,500（约+0.09%）。g=1的相同变化仅+281 cycles、约0.07%。这强烈支持多CTA上下文容量门槛的因果作用，而非新增warp的少量启动指令。16w、g=2的对照为+2.87%，两个5-warp CTA仍可完整驻留；这个场景的K不同，不用于隔离8→16容量变化。

用精确dispatch/exit时间重建，驻留平均值的分母为首次dispatch至最后exit；“完整计算组”表示4个compute warp同时存活，**不是同时执行、就绪率或TCU利用率**。

{table(['hardware warps','控制方式','平均完整计算组CTA','平均全部warp存活CTA','WGMMA事件','ReLU事件'],capres)}

主场景中实际驻留如下。部分入驻CTA也计入各自warp，未把容量除法当作实际并发测量。

{table(['hardware warps','方法','平均live warps','平均完整计算组CTA','平均全部warp存活CTA'],res)}

8w普通融合可以保持约1.96个完整计算组，5/7-warp的两/三阶段约1.00。专用loader/activation会减少为其他CTA留下的上下文，独立CTA原本提供的延迟隐藏因此受到限制。实际两阶段比“普通融合+闲置warp”快，说明专用loader也在发挥作用；不能把闲置warp对照的43.73%直接等同于实际三阶段退化的某个比例。

## 4. 阶段重叠确实有收益，但取决于配置

{table(['hardware warps','K','g','activation warps','原三阶段cycles','提前等待C槽cycles','提前等待 vs 原三阶段'],overlaps)}

16w A1提前等待后多15,338 cycles，原流水相对该消融少约16.02%周期。trace里原A1有592个activation事件位于同CTA下一tile的计算窗口，提前等待后为0。8w A2限制重叠后也明显变慢；它已有的重叠收益仍不足以抵消相对普通融合的整体代价。

16w A2提前等待反而略快408 cycles（约0.50%）。阶段重叠并非在所有分配中都能转为净收益，活动窗口和“同tick出现”也不等同于硬件利用率。不能把A1的消融比例推广到A2或其他形状。

## 5. 可以精确数出的动态成本

{table(['hardware warps','方法','动态指令计数','BAR指令含expect/arrive','C写lane操作','C读lane操作','Y写lane操作'],counters)}

两个场景总输出都是32,768个FP32值，三阶段增加32,768次C写及32,768次C读，即128KiB写+128KiB读=**256KiB额外SMEM有效载荷**。这里是已知4字节FP32的程序级lane操作，不是DRAM流量、片上总线传输量或能耗；DXA本身也不经过这份普通LSU请求统计。输入A的打包共享内存load、WGMMA/activation事件与总输出工作量均逐trace核对。

16w两阶段→A1的动态指令由23,086增到33,352（约+44.47%）；8w两阶段→A2为76,476→87,740（约+14.73%）。增加来自队列交接、索引、循环、额外访存和同步等执行工作，不能全部称为ReLU算术开销。计算、ReLU事件总数保持不变。

只加共享内存往返的16w对照虽然工作更多，反而略快，同时input-ready的纯DXA等待由321,616降到214,568 warp-cycles，而BAR gate阻塞增加。这说明插入操作还改变了预取/同步节奏；现有结果不足以把“搬运本身”独立估成一个正的固定周期税。

## 6. 同步等待：LSU drain 与 barrier 等待分别看

现有SimX在BAR的eop进入Wctl之前要求 `core_->lsu_drained()`；它是全核LSU检查，因此不同warp/CTA可能互相影响。观察代码没有取消drain，也没有改变barrier语义。此处区分两类时间：

- **gate阻塞ticks**：同一tick至少有一个BAR由于LSU未排空而未进入barrier处理。这是实际gate活动，不表示该tick全核没有做其他工作，不能直接叫“损失周期”。
- **suspended warp-cycles**：warp已到达barrier后，直到被resume的累计等待。多个warp可同时等待，数值可以大于kernel elapsed cycles。它与gate、scoreboard和其他MPM指标也不能相加。

{table(['hardware warps','方法','LSU gate阻塞ticks','ticks / kernel cycles','activation等C-ready warp-cycles','compute等C-free warp-cycles'],sync)}

三阶段在16w从18,488降到3,463个gate阻塞tick，在8w从19,122降到8,623，但BAR数量反而更多。因此“更多barrier导致更多LSU drain，从而让三阶段变慢”不能充分解释当前结果。gate仍有影响，只是不能作为单一根因，更不能用gate差额直接解释净周期差。

input-ready处的等待按实际barrier计数和DXA event状态进一步分类：

{table(['hardware warps','方法','只缺DXA完成 warp-cycles','只缺参与者 warp-cycles','两者都缺 warp-cycles'],deps)}

16w两/三阶段input-ready处主要是在等DXA完成，三阶段A1只有134 warp-cycles属于“只缺参与者”；8w两阶段则有708,752、三阶段有290,634。三阶段这些参与者等待中，可缺compute、loader或两者；`analysis.json`记录具体缺失角色，不能统一称作loader供给不足。三阶段C-ready的最后参与者在两组各16个输出中均为compute，8w activation累计等C-ready达436,115 warp-cycles，也不支持把主要瓶颈简单归为activation算术速度。

## 7. 目前能解释到什么程度

**16w/K64/g2 A1为什么小幅获益：** 两个计算组仍能完整驻留，计算与activation存在真实的跨tile重叠；提前等待的消融明确损失15,338 cycles。同时该组织改变同步/LSU竞争节奏，gate阻塞显著减少。额外256KiB SMEM和更多动态指令抵消部分好处，最后比两阶段少2,769 cycles（3.33%）。这些变化相互影响，不能给15,338、gate差额、SMEM对照作算术相加。

**8w/K256/g4为什么三阶段输给普通融合：** 普通融合能借约两个完整计算组隐藏延迟；专用角色让完整组驻留降至约一个，并伴随较多参与者等待和额外交接。普通融合的闲置warp消融直接证明容量门槛可以造成大幅退化。三阶段虽从重叠中获益，仍为285,096 cycles，比普通融合242,342多17.64%。即使给三阶段选择此前扫描的最佳g=1，也为267,631，仍多10.44%。主因解释已从泛泛的“流水线有开销”，收敛为容量门槛、角色同步和额外动态工作共同作用。

**已经排除或限制的说法：** 当前不是单纯多预留SMEM导致变慢；也不是BAR drain次数越多就一定越慢；不能只数activation warp或只看单CTA结果；本批没有可加总的精确cycle分账，也没有证明所有形状的统一主瓶颈。

后续若只做一项设计探索，优先考虑在8w容量紧张时复用角色上下文或缩减专用warp需求，并保留普通融合强基线和相同预取策略。若选择16w获益场景继续解释，应补一个有针对性的DXA/LSU/TCU活动时间线来判断供给节奏，不宜先取消内存可见性语义、直接设计大规模跨核队列或继续无界扫描policy。本轮数据都限于单核 timing SimX，没有新增RTL、PPA、FPGA、自动编译器/RFQ实现结论。

## 8. 复现与产物

```bash
EXP='experiments/cost_attribution'
python3 "$EXP/run_cost.py" --phase smoke --warps 8
for w in 8 16; do
  for phase in normal validate trace; do
    python3 "$EXP/run_cost.py" --phase "$phase" --warps "$w"
    python3 "$EXP/run_cost.py" --phase "$phase" --warps "$w" --capacity
  done
done
python3 "$EXP/analyze_cost.py"
python3 "$EXP/report_cost.py"
```

`measurements.csv`为87条正式逐例数据，`analysis.json`包含20份trace的汇总和来源。每份trace有对应`*_residency.csv`、`*_activity.csv`、`*_waits.csv`；原始压缩trace、源码、kernel dump及binary保存在JSON指向的logs目录。计数口径来自固定提交的 `sfu_unit.cpp`、`wctl_unit.cpp`、`barrier_unit.cpp`、`scheduler.cpp`、`cta_dispatcher.cpp`、`core.cpp`，日志补丁由`prepare_cost.py`生成。

开发过程中首次C往返smoke因误用WMMA的m-major累加器回读而数值失败；已改成WGMMA n-major逆映射，失败日志保留，不纳入正式测量。另一次复用观察构建目录时的Makefile路径检查失败发生在SimX执行之前，已改成幂等检查。没有放宽CPU checker或计数一致性条件。
'''
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8')
    print(HERE/'RESULTS.md')


if __name__=='__main__':main()
