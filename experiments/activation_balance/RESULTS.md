# Project 20：activation 工作分配与循环组织实验

本批完成 78 次 timing SimX 运行，全部通过精确 CPU reference、正负/零值覆盖与输出 hash 检查。三阶段采用两个 activation warp 并展开四组元素后，大幅缩小了与两阶段的差距，但仍没有超过本批两阶段强基线。当前结论只覆盖单个 persistent CTA。

## 1. 研究问题与控制条件

上一轮三阶段实际发生阶段重叠，但比两阶段慢。本批分别改变 activation warp 数和循环展开，检查负载分配与指令组织能否改善结果。

- 固定 Vortex v3.0 commit `7fe19369587c087f902c3e2bc2add612aa910a95`、工具链 v3.0.1。
- 1 核、32 lanes、issue width=4、4 TCU blocks，ALU/LSU/FPU 各 1 block；64 KiB LMEM、LSU pending=16、AMO RS=8，L2/L3 关闭。主实验为 16 hardware warp，额外做 8-warp 对照。
- 新应用 `tests/regression/activation_balance` 保留普通融合、两阶段和三阶段。原 `persistent_pipeline` 应用及结果未覆盖。普通融合继续支持跨输出 tile 的 A/B 预取。
- 计算仍为 4 warp，加载仍为 1 warp。三阶段的 activation 为 1/2 warp，CTA 总计为 6/7 warp。更多 activation warp 没有增加 FPU/LSU block 数，但增加每 CTA 使用的 warp 上下文。
- 每个输出 tile 为 64×32；M=64，N=32×tile 数，K=16/64。A/B 深度固定为 2。C 队列深度 1/2 对应每 CTA 14/22 KiB SMEM；普通融合与两阶段为 6 KiB。

`A1/U4` 表示一个 activation warp，每次外层循环处理四组、每组每 lane 一个元素。它是编译期循环展开，没有把 SIMD 宽度扩大四倍。当前 tile 的元素数能被 `32 × activation_warps × unroll` 整除，因此展开没有越界尾部。

输出 ready/free 的参与者同步改为 `4 + activation_warps`；输入 ready/free 仍为 5。两个 activation warp 处理互不重叠的行，全部完成后才释放 C 槽位。沿用 BAR 的 LSU drain 语义，保留尾部排空。

## 2. 公平性与正确性

全部阶段与资源配置的源码哈希一致，8/16-warp 和 debug/normal 的 device binary 哈希也一致。普通融合和两阶段的全部六个形状均精确重现上一轮周期、指令、全局请求计数及输出 hash。

新三阶段循环为模板化分配实现，与上一轮的函数布局/代码生成略有区别；其新基线应使用本轮 `A1/U1`，不能将相对上一轮的微小变化归因于 warp 数。本轮主要改善均通过同一新应用内的对照衡量。

输入仍为 `[-16,16]/16` 的精确 FP16 数值，CPU 用整数点积再除以 256，包含零行/零列和正负结果。输出预填 NaN 检查漏写。主矩阵 36 例，smoke 6 例，阶段位置/seed/队列控制 14 例，8-warp 对照 6 例，关键重复 12 例，debug trace 4 例。

## 3. 主实验周期

16 hardware warp、C depth=1、seed=1。周期越低越好。

| 输出 tiles | K | 普通融合 | 两阶段 | A1/U1 | A1/U4 | A2/U1 | A2/U4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 3512 | 3809 | 12851 | 9624 | 8208 | 6570 |
| 1 | 64 | 6701 | 6792 | 15998 | 12600 | 11190 | 9553 |
| 2 | 16 | 6076 | 5958 | 23646 | 16664 | 13829 | 10648 |
| 2 | 64 | 12728 | 11605 | 26571 | 19843 | 17288 | 14087 |
| 5 | 16 | 12759 | 12065 | 54608 | 37406 | 30325 | 22285 |
| 5 | 64 | 28396 | 26660 | 57619 | 41232 | 35128 | 27586 |

5 个输出 tile、K=64 的资源与指令对照：

| 组织 | CTA warps | SMEM bytes | cycles | instructions | 周期 vs 两阶段 |
| --- | --- | --- | --- | --- | --- |
| 普通融合 | 4 | 6144 | 28396 | 7946 | +6.51% |
| 两阶段 | 5 | 6144 | 26660 | 7298 | +0.00% |
| A1/U1 | 6 | 14336 | 57619 | 11768 | +116.13% |
| A1/U4 | 6 | 14336 | 41232 | 10568 | +54.66% |
| A2/U1 | 7 | 14336 | 35128 | 11972 | +31.76% |
| A2/U4 | 7 | 14336 | 27586 | 10772 | +3.47% |

改变 activation warp 数和循环展开都能降低此三阶段原型的周期。循环展开在 5-tile、K=64 下减少 1200 条已退休指令，两个 warp 本身则略增加总指令数，但明显缩短执行周期。结果支持阶段并行度和循环组织值得优化；没有逐周期隔离调度、访存等待与代码生成的因果份额。

## 4. ReLU 放在哪里

额外对照在四个 compute warp 内对 accumulator 执行 ReLU，再通过相同 C 队列交给输出 warp；输出 warp 只读共享内存并写全局内存。最终仍验证相同 GEMM→ReLU 输出，并没有删除最终计算。

| 组织 | consumer ReLU cycles | producer ReLU + consumer copy cycles | 周期变化 |
| --- | --- | --- | --- |
| A1/U1 | 57619 | 58945 | +2.30% |
| A1/U4 | 41232 | 40647 | -1.42% |
| A2/U1 | 35128 | 36697 | +4.47% |
| A2/U4 | 27586 | 28626 | +3.77% |

移动 ReLU 没有稳定改善，不能认为当前开销主要来自单条 fmax。这个对照同时改变算术位置和调度关系，不能当成纯 FPU 延迟或纯共享内存开销的精确测量。输入搬运、完整 C tile 的共享内存存取、最终输出和握手都仍然存在。

## 5. 队列与 hardware warp 对照

展开版的 C depth=1/2：

| K | 组织 | depth1 cycles | depth2 cycles | 周期变化 |
| --- | --- | --- | --- | --- |
| 16 | A1/U4 | 37406 | 37082 | -0.87% |
| 64 | A1/U4 | 41232 | 40476 | -1.83% |
| 16 | A2/U4 | 22285 | 21574 | -3.19% |
| 64 | A2/U4 | 27586 | 27516 | -0.25% |

depth2 增加 8 KiB SMEM；本批没有改变三阶段仍慢于两阶段的结论。不能把输出队列深度当作阶段并行度的替代。

固定同一 CTA 工作量的 16/8 hardware warp 对照：

| 组织 | 16 hardware warp cycles | 8 hardware warp cycles |
| --- | --- | --- |
| 普通融合 | 28396 | 28456 |
| 两阶段 | 26660 | 26782 |
| A1/U1 | 57619 | 57605 |
| A1/U4 | 41232 | 41199 |
| A2/U1 | 35128 | 35262 |
| A2/U4 | 27586 | 27599 |

当前所有 CTA 均只需 4/5/6/7 warp，8-warp 配置足以容纳它们。这里只检查单 CTA 对硬件 warp 容量的敏感性，不代表多个 CTA 的驻留结果，也不是等面积比较。

## 6. 重复与 trace

以下同配置观测包含主矩阵、两次额外重复及对应 debug 运行；所有相同配置的 core/memory 计数均一致。第二个 seed 单独通过正确性检查。

| 组织 | 同配置观测次数 | min cycles | median cycles | max cycles |
| --- | --- | --- | --- | --- |
| 普通融合 | 3 | 28396 | 28396 | 28396 |
| 两阶段 | 3 | 26660 | 26660 | 26660 |
| A1/U1 | 4 | 57619 | 57619.0 | 57619 |
| A1/U4 | 4 | 41232 | 41232.0 | 41232 |
| A2/U1 | 4 | 35128 | 35128.0 | 35128 |
| A2/U4 | 4 | 27586 | 27586.0 | 27586 |

| 组织 | DXA transfers | WGMMA events | FMIN_MAX events | activation span median | 下一 compute 窗口内 activation events | 同 tick 数之和 |
| --- | --- | --- | --- | --- | --- | --- |
| A1/U1 | 40 | 2560 | 320 | 9581 | 74 | 9 |
| A1/U4 | 40 | 2560 | 320 | 6356 | 110 | 12 |
| A2/U1 | 40 | 2560 | 320 | 5162 | 143 | 9 |
| A2/U4 | 40 | 2560 | 320 | 3489 | 231 | 12 |

每个版本都执行相同数量的 DXA、WGMMA 和 ReLU 事件。每个输出 tile 的 activation 均在该 tile 全部 K 归约之后开始；按 warp 内顺序分组，并检查事件总数。窗口见 `activity_a*_u*.csv`。

activation span 是首末 FMIN_MAX 事件之间的跨度，包含访存、指令依赖和调度等待；不是纯 FPU 延迟、完整 epilogue 时间或硬件利用率。窗口重叠和同 tick 事件是实际重叠的证据，不是单独的性能收益证明。

## 7. 范围与下一步

这批实验表明，先前三阶段的大幅退化有相当部分可以通过 activation 工作分配和循环组织改善；但轻量 ReLU 仍未证明值得从 accumulator 融合路径拆出。普通融合/两阶段避免了额外完整 C tile 的共享内存交接，仍是重要对照。

下一批优先从单 CTA 扩到多个 persistent CTA，在 8/16 hardware warp 下检查 A2/U4 的改善是否能抵消每 CTA 从 5 增至 7 warp 的驻留成本；每个 CTA 仍需连续处理多个输出 tile。也可以延长 K 或 tile 序列，观察填充/排空成本与计算量的关系，再选择更重后处理或更细粒度输出块作为扩展。完成这些核内对照后，再扩展跨核队列/KMU placement。本轮没有修改上游 RTL/SimX，没有重跑此前超时的 RTL build，因此没有新增 RTL、XRT 或 PPA 结论。

## 8. 复现与文件

```bash
python3 'experiments/activation_balance/run_balance.py' --phase smoke --warps 16
python3 'experiments/activation_balance/run_balance.py' --phase matrix --warps 16
python3 'experiments/activation_balance/run_balance.py' --phase controls --warps 16
python3 'experiments/activation_balance/run_balance.py' --phase repeat --warps 16
python3 'experiments/activation_balance/run_balance.py' --phase trace --warps 16
python3 'experiments/activation_balance/run_balance.py' --phase capacity --warps 8
python3 'experiments/activation_balance/analyze_balance.py'
```

新源码镜像位于 `activation_balance/`，生成过程在 `prepare_sources.py`；旧源码哈希不符时停止。运行脚本保存逐例命令、源码、device binary/dump、完整有效配置与压缩 trace。`measurements.csv` 为全部逐例计数，`analysis.json` 保存验证后的汇总和日志路径。
