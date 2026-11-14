# Project 20: WSL 环境、研究起点与继续工作指引

当前仓库的构建与复现命令见 [README](../README.md#reproduce)。以下笔记保留最初 WSL 环境中的工作记录。

更新日期：2026-10-23。本文是技术交接指引，不是课程 proposal。

## 1. 当前范围和已验证状态

目标：为 CS254A Project 20 “Dataflow Pipelines on Vortex” 的 proposal 准备可复现的运行基础、源码依据和评估方法。当前只建立环境和测试上游示例，不实现新的流水线或修改 Vortex RTL/SimX。

2026-10-23 的现场检查结果：

- Codex Desktop 可以通过 PowerShell 调用 `wsl.exe`。受限执行最初返回 `Wsl/E_ACCESSDENIED`；经工具权限提升后，可以执行 Linux 命令、访问 GitHub、安装依赖和构建。
- 实际使用发行版：`Ubuntu-22.04`，Ubuntu 22.04.5 LTS，WSL2，用户 `root`。
- 另有发行版 `Ubuntu`，Ubuntu 24.04.1 LTS，用户 `samcby`。这是另一套文件系统，不要混用路径。
- 两套发行版都没有发现先前记录的 `/tmp/cs254a_lab2`；Windows 课程文件仍保留。这不表示以前的 Lab 2 源码或构建目录已恢复。
- 已安装必要系统开发包，并完成 v3.0 源码、子模块、RV32 工具链、第三方库和 kernel/runtime 基础构建。
- 环境与基线测试的状态、命令、退出码和日志位置见 [baseline_state.json](../results/baseline/baseline_state.json)。

本轮结束状态：`setup`、`smoke`、`measure`、`trace` 四阶段均完成。无需为了访问/构建能力迁移到另一窗口；也可以按下方路径在 WSL Codex 接续。

| 实际运行 | 指令数 | 设备周期 | 结果 |
|---|---:|---:|---|
| 默认配置 demo | 2,444 | 6,529 | PASSED |
| 默认配置 SGEMM 32×32 | 48,400 | 115,058 | PASSED |
| DXA+WGMMA 单缓冲，128×128×64 | 9,160 | 39,932 | PASSED |
| 同配置双缓冲 | 11,640 | 41,970 | PASSED |

WGMMA 两个版本分别以 perf class 7/16/11 运行，三个采集类别下各自的指令数和周期完全相同。单缓冲片外请求为 4,699（读 3,636 / 写 1,063）；双缓冲为 5,190（读 3,882 / 写 1,308）。双缓冲周期增加约 5.1%；这是一个配置/输入的观察结果，尚未完成因果分析，不能泛化为双缓冲总是更慢。

DXA trace 测试通过，已检查导出的 JSON 能解压解析并包含指令切片。Perfetto UI 视觉检查待完成。 [DXA trace](../archive/logs/baseline/trace_20261023T205131Z/dxa.perfetto.json.gz)

上游受版本控制的源码没有修改。`third_party/cocogfx` 中出现 `.o`、`.a` 和 `.depend` 等构建产物，因此 `git status` 会报告该子模块含未跟踪文件；不要把这误报成设计源码改动。

目录：

| 用途 | 路径 |
|---|---|
| Windows 资料、脚本与运行记录 | `D:\UCLA_Master\CS 254A\project` |
| 同一目录在 WSL 中的路径 | `/mnt/d/UCLA_Master/CS 254A/project` |
| 独立的持久工作目录 | `/root/cs254a/project20` |
| Vortex 源码 | `/root/cs254a/project20/vortex` |
| 普通 RV32 构建 | `/root/cs254a/project20/vortex/build32` |
| WGMMA 测量构建 | `/root/cs254a/project20/vortex/build32-wgmma` |
| 小规模 DXA trace 构建 | `/root/cs254a/project20/vortex/build32-dxa-trace` |
| 专用工具链 | `/root/cs254a/project20/tools-v3.0.1` |

后两个构建目录由相应脚本阶段创建；是否已经运行，以状态文件为准。源码与构建放在无空格的 Linux 路径，Windows 路径用于共享资料与日志。不要使用 `/tmp` 保存唯一的源码或工具链。

## 2. 固定版本

- 仓库：<https://github.com/vortexgpgpu/vortex>
- 标签：`v3.0`
- 实际源码提交：`7fe19369587c087f902c3e2bc2add612aa910a95`
- 该提交的 `VERSION`：`VORTEX_VERSION=3.0`，`TOOLCHAIN_REV=v3.0.1`。
- 安装的组件：LLVM、riscv32 GNU、libcrt32、libc32、Verilator。当前未为本项目安装 POCL、HIP、Triton、Yosys 或 OpenSTA。
- 实测工具版本：Clang 20.1.8、RISC-V GCC 12.2.0、Verilator 5.050。以实际二进制输出为准，不能从旧 release 描述推断实际工具版本。

本地 Lab 2 是 v2.3 项目。不要在本项目套用其补丁、旧宏名或 `source ci/toolchain_env.sh` 流程。v3.0 使用各 build 目录保存的工具路径和 TOML 生成配置。

## 3. 从 Windows 或 WSL 继续

从 PowerShell 进入正确发行版：

```powershell
wsl -d Ubuntu-22.04 -u root
```

进入后先检查：

```bash
cd /root/cs254a/project20/vortex
git rev-parse HEAD
git status --short
cat VERSION
cat AGENTS.md
```

所有测试从已配置的 build 目录执行。课程目录中的脚本可以从 WSL 调用：

```bash
python3 '/mnt/d/UCLA_Master/CS 254A/project/bootstrap_v3.py' --phase smoke
```

脚本阶段：

| 阶段 | 动作 |
|---|---|
| `setup` | 安装系统依赖，检出固定源码，安装五个预编译组件，构建第三方库和 kernel/runtime |
| `smoke` | 执行上游 SimX `demo` 与 `sgemm -n32` |
| `measure` | 在独立 WGMMA 构建中运行现有单缓冲/双缓冲版本，分别采集 memory、DXA、TCU 指标 |
| `trace` | 在独立构建中运行小规模 `dxa_copy -d1` 的 debug trace，并转换成 Perfetto JSON |

按顺序执行尚未完成的阶段，每个阶段完成后检查状态和日志。不要并发运行这些阶段，它们会更新同一个状态文件。

```bash
python3 '/mnt/d/UCLA_Master/CS 254A/project/bootstrap_v3.py' --phase measure
python3 '/mnt/d/UCLA_Master/CS 254A/project/bootstrap_v3.py' --phase trace
```

`setup` 需要 root，因为包含 apt 安装。已经完成 setup 时优先继续测试阶段；不要重复下载。不要另行执行会覆盖已修改源码的恢复操作。

脚本把每条命令、工作目录、退出状态和原始输出写入 `project/logs/`；`baseline_state.json` 记录最近状态和累计步骤。若失败，先读相应 `.log` 的最后部分，修复实际原因再继续。当前脚本只是环境/示例运行辅助，不是最终实验框架。

`environment_verified.json` 保存工具版本、源码/子模块状态和配置校验信息。首次 trace 检查曾因辅助脚本假设成功标记为 `PASSED!`、JSON 顶层为对象而报错；实际示例输出 `PASSED`，导出格式为数组。辅助脚本已按实际格式修正，旧日志保留，最后一次 trace 阶段通过。这些错误不属于 Vortex kernel 计算失败。

已发现的文档参数问题：AGENTS 的 SGEMM smoke 示例为 `-n10`，但本地 `sgemm` 要求矩阵尺寸能被默认 4×4 block 整除，故该输入被拒绝。脚本使用 `-n32`；原始失败日志保留，未修改上游计算代码。

## 4. 必须遵守的仓库规则

以固定提交中的 `AGENTS.md` 及子目录规则为准：

1. 使用 out-of-tree build；生成后的脚本与 Makefile 位于 build 目录。
2. 不同主要配置使用独立 build 目录。修改 Makefile、TOML、构建目录结构后重新 `configure`；测试前确认生成文件已更新。
3. 硬件配置以 `VX_config.toml` / `VX_types.toml` 为依据，宏使用 `VX_CFG_` 前缀。应用与模拟器必须使用匹配的配置。
4. 性能比较使用正常时序 SimX；`SIMX_FUNCTIONAL` 的周期不具有性能含义。
5. SimX 的模块只能通过 bound channels 通信，不能直接跨模块读取或修改内部状态。
6. 改动队列深度、仲裁、缓存或流水线时序时，安排相匹配的 SimX/RTL 模型与 parity 验证。
7. `rtlsim` 用于处理器 RTL 迭代；完整 AFU 集成验证走 `xrt` 路径。SimX PASS 不能证明 RTL 或 FPGA 集成通过。
8. 不手工修改上游 perf/synthesis golden baselines，不通过放宽误差门限掩盖问题。

## 5. 项目要求与研究范围

原始材料在当前目录：`project20.pdf`、WASP HPCA 2024 PDF、Kitsune TACO 2025 PDF。

课程项目的正常目标包含：

- load → GEMM → activation 的三个 v3.0 kernel 顺序 baseline。
- 核内 warp-specialized pipeline；共享内存 tile 队列和异步 barrier。
- 跨核队列，使用硬件 atomics，并支持 KMU 的阶段 CTA 放置。
- 内存流量、Tensor/DXA 活动和端到端延迟研究。
- 明确区分原论文机制复现、Vortex 适配和新增内容。

跨核队列/KMU placement 已在交付要求中，不能直接当作 stretch goal。完整自动编译器、寄存器文件队列 RFQ、按阶段动态寄存器分配和 CGRA 暂不承诺；需要向教授确认 WASP 的复现深度及 RTL 交付边界。

研究问题：何时 tile 流水线的局部性和执行重叠收益，能够抵消同步、队列和资源划分开销？重点变量是 tile 大小、队列深度、各阶段 warp/core 分配，以及核内/跨核通信。

## 6. 已从本地源码确认的起点

优先阅读 `tests/regression/sgemm_tcu_wg_dxa/{main.cpp,kernel.cpp,common.h,Makefile}`，并跟踪 `sw/kernel/include/{vx_dxa.h,vx_barrier.h,vx_tensor.h}`。

在固定提交的实际本地源码中：

- 默认路径是单缓冲：发起 DXA，等待搬运，WGMMA 计算，释放缓冲。
- 已有 `DXA_DOUBLE_BUFFER` 宏路径：两组 A/B 缓冲、两个 barrier 槽，通过指针与 barrier id 切换，预取下一 K tile。
- 第一个 warp 发出 DXA，但仍参与计算；这是已有软件预取流水线，不是独立 loader/consumer warp 的 WASP 式阶段划分。
- host 把 CTA 的活跃 warp 数设为 `ISSUE_WIDTH`，因为 WGMMA 按 warpgroup 协同执行。新增独立 loader warp 时必须分析该约束，不能只插入一个 `if` 分支。
- 双缓冲增加 LMEM 占用，可能减少 CTA 驻留。源码注释还描述了寄存器溢出和栈访问的历史性能问题；这些注释只能作为调试线索，不能当作本项目实测结论。

前面在线阅读仅描述了默认单缓冲分支；本地实际检出的源码补充确认了已有双缓冲分支。后续以固定提交的本地源码为实现依据。

## 7. 当前示例测量的定义

`measure` 阶段采用仓库 CI 中已有的 WGMMA+DXA 配置作为基础：

- RV32，1 核，32 threads/warp，8 个硬件 warp，issue width 4。
- 开启 TCU、WGMMA、DXA；`WGMMA_NRC=16`。
- ALU/LSU/FPU 各 1 block，其余资源按生成配置核查。
- 输入 fp16、输出 fp32；矩阵 `M=N=128, K=64`。
- 两个版本只改变软件宏 `DXA_DOUBLE_BUFFER`，但该宏会改变每 CTA 的 LMEM 需求，因此必须同时记录驻留/资源影响。

实际 perf class 从本地 `VX_types.toml` 核对：memory=7，TCU=11，DXA=16。某些旧文档/CI 参数中的 `--perf=6` 是 L3 类，不能当作 DXA 指标。

这组测量只是上游 GEMM 示例的单/双缓冲对照，不是完整 load/GEMM/activation 算子链，不是 WASP 复现结果，也不是项目最终 speedup。

## 8. 评估方法和接下来的设计

最终至少比较：三个顺序 kernel、普通融合 kernel、核内 warp-specialized pipeline、跨核 pipeline + placement。

先固定 `Y = ReLU(A B)` 的语义，并明确独立 load kernel 是 staging、复制还是布局变换，避免人为增加工作而夸大流水线收益。GEMM 输出 tile 必须完成 K 归约后才能交给 activation。

记录正确性误差、整链设备周期（含填充/排空/同步）、tile 吞吐、片外读写、TCU/DXA 活动、队列占用和空/满等待时间。host 墙钟时间不能替代模拟设备周期。多核周期不能把各核计数直接相加。

跨核队列需明确 payload、槽位所有权、序列号、发布/获取顺序、消费者 L1 可见性、终止条件和资源驻留保证。全局地址上的小缓冲区并不保证驻留 L2；应测量缓存与片外流量。避免生产者占满资源、消费者未驻留造成死锁。

WASP：重点阅读 §III（阶段命名、映射/调度、RFQ、TMA）和 §IV（阶段提取与同步）。手工 warp specialization 只是其中一部分。

Kitsune：重点阅读 §4.1（队列/内存一致性）、§4.2（异构 CTA 调度）、§5.3（负载均衡）、§6.7（消融）。其原始设计也利用异构 CTA 同驻一个 SM，不只是在不同核心之间分割算子。

优先的后续实现：先理解并测量已有单/双缓冲，再设计独立 producer/consumer warp 的两阶段最小原型；正确性及重叠被验证后加入 activation，再扩展跨核。

## 9. 工具与资料入口

- [v3.0 release](https://github.com/vortexgpgpu/vortex/releases/tag/v3.0)
- [AGENTS.md](https://github.com/vortexgpgpu/vortex/blob/v3.0/AGENTS.md)
- [安装](https://github.com/vortexgpgpu/vortex/blob/v3.0/docs/install_vortex.md)、[模拟](https://github.com/vortexgpgpu/vortex/blob/v3.0/docs/simulation.md)、[源码地图](https://github.com/vortexgpgpu/vortex/blob/v3.0/docs/codebase.md)
- [Perfetto](https://github.com/vortexgpgpu/vortex/blob/v3.0/docs/perfetto_analysis.md)：实际脚本是 `ci/perfetto.py`；文档中的 `vortex_perfetto.py` 是旧命名。指令/warp/cache 事件不等于完整的 tile 队列或请求生命周期，需要检查是否补埋点。
- [Roofline](https://github.com/vortexgpgpu/vortex/blob/v3.0/ci/roofline.py)：使用 memory profile；TCU 峰值必须显式给出。首先使用固定配置和 FLOP/cycle，避免把默认平台带宽/任意频率当作本机测量。
- [综合/PPA](https://github.com/vortexgpgpu/vortex/blob/v3.0/docs/synthesis_analysis.md)：新增 RTL 明确后再考虑面积与 pre-layout STA；功耗需解释 SAIF 覆盖和 workload。流量减少不是已测得能耗下降。
- [RISC-V 规范](https://riscv.org/specifications/ratified/)：跨核 atomics/fence 内存语义的参考，需结合 Vortex 实现验证。
- [POCL](https://portablecl.org/)、[HIP/chipStar](https://github.com/vortexgpgpu/vortex/blob/v3.0/docs/designs/hip_on_vortex_chipstar.md)、[Triton](https://github.com/triton-lang/triton)：背景/后续接口选择。目前原生 C++ kernel 最便于控制专用 intrinsic；不要假定上游 Triton 直接支持 Vortex，或当前 HIP/SPIR-V 路径能暴露 DXA/WGMMA/barrier。

## 10. 交给 WSL Codex 的任务提示

> 请先阅读 `/mnt/d/UCLA_Master/CS 254A/project/WSL_GUIDE.md`、`baseline_state.json` 和 `/root/cs254a/project20/vortex/AGENTS.md`。本项目从固定提交 `7fe19369587c087f902c3e2bc2add612aa910a95`、工具链 `v3.0.1` 起步。先核查已有日志，继续尚未完成的 smoke/measure/trace 阶段，不重建已完成的安装。保留 Lab 2 和现有源码。当前目标是 proposal 所需的可行性、baseline 和同步设计依据；暂不生成正式 proposal，不开始跨核或编译器的大规模实现。明确报告实测结果、未验证内容及下一批最小工作。

## 11. Proposal 前的有限探索与当前停止点

用户明确要求：现阶段只做 proposal 所需的初步探索，形成研究方向、预期效果和后续计划，不完成整个项目。环境准备已完成；本轮新增六个用例后结束探索，不继续自动实现独立 warp 阶段、activation、跨核队列或 RTL。

新增实验脚本 `proposal_probe.py` 复用固定源码和原配置，保持 M=N=128，只增加 K=32/128/256 的单/双缓冲各一次。K=64 复用前一批日志。未修改上游 kernel、数值判据或硬件实现。

| K | 单缓冲 cycles | 双缓冲 cycles | 双缓冲周期变化 |
|---|---:|---:|---:|
| 32 | 27,147 | 25,878 | -4.67% |
| 64（前一批） | 39,932 | 41,970 | +5.10% |
| 128 | 69,488 | 68,974 | -0.74% |
| 256 | 119,311 | 121,628 | +1.94% |

六个新增用例全部通过上游数值检查。改变 K 时上游默认 ULP 容限也按其既定函数变化，脚本没有放宽容限；单/双缓冲在同一个 K 的最大误差一致，分别为 2/3/5 ULP（K=32/128/256）。

结果不呈现“循环更长则双缓冲更好”的单调趋势。动态指令开销始终存在；缓存流量和共享内存竞争随输入及调度变化。K 同时改变循环长度、输入 footprint 和寻址跨度，故该探索不能分离各因素的因果贡献。没有跨随机种子、配置或 RTL 泛化验证。

结合本地配置进一步核查：当前 LMEM 为 64 KiB，每 CTA 3/6 KiB，dispatcher 的 LMEM slot 容量上限均为 8；每 CTA 使用 4 warp、硬件有 8 warp，故不能把此处退化直接归因于双缓冲减少容量允许的驻留数。独立重编译的单/双缓冲反汇编都有相同的一次直接栈存储，尚无双缓冲新增直接 stack spill 的证据。

当前足以支持 proposal 的研究问题是：在 Vortex 上如何组织算子流水线的阶段、缓冲和资源分配，使重叠与局部性收益超过管理与竞争开销？预期效果是减少完整中间结果的片外往返，并在合适负载上降低整链周期；尚不承诺统一加速幅度。

后续 checkpoint 可按相对顺序安排：

1. 定义并完成 load/GEMM/activation 三 kernel 基线及普通融合对照，冻结数据类型、尺寸和流量/周期口径。
2. 在 SimX 上实现手工划分的核内 producer/consumer 阶段，验证缓冲协议、终止和实际重叠。
3. 扩展跨核原子队列与 KMU placement，验证数据可见性和进展条件。
4. 比较四类执行方式，分析 tile/队列/阶段资源分配的敏感性，并解释退化案例。

阶段分配策略或阶段感知调度可选作 stretch。跨核队列与 KMU placement 保持为项目原定目标。完整 WASP 编译器/RFQ、RTL 覆盖边界和 load kernel 的确切语义仍需教授指导。

这批数据记录了 proposal 阶段的初步特性观察。 [结果 JSON](../results/baseline/proposal_probe_results.json) · [原始运行日志](../archive/logs/baseline/proposal_probe_20261023T220813Z/)

## 12. 2026-10-28：进入核内 producer–consumer 实验

用户已授权分析已有测试并开展进一步实验，以上 proposal 阶段的停止点是历史记录。最新完整结果见 [`experiments/pc_pipeline/RESULTS.md`](../experiments/pc_pipeline/RESULTS.md)，机器可读汇总见同目录 `analysis.json`。

- 使用原固定提交和工具链，新增独立构建 `build32-pipeline`。上游受版本控制的源码未修改。
- 新增应用为 `dxa_pc`、`sgemm_tcu_wg_pc`、`sgemm_tcu_wg_control5`、`sgemm_tcu_wg_pc_mask`；Windows 目录保留源码镜像、安装/运行脚本和逐批日志。
- 最终协议完成 ready/free barrier 及尾部排空。最终版本累计 59 次 SimX 运行通过，包括 2 次 GEMM debug trace；初版记录留存但未混入最终比较。
- 4 个计算 warp 加 1 个加载 warp 的 GEMM 组织在本批用例中运行正确。双缓冲 trace 确认 DXA 活动区间内出现 WGMMA 计算事件；单缓冲对应计数为 0。
- 额外 warp、同步和缓冲管理成本抵消了部分收益。将 buffer 索引从取余改为等价位运算后，128×128×64 原型为 47,699 cycles，仍慢于上游 4-warp 双缓冲的 41,970 cycles。
- 仅测试 kernel 软件改动；尚未进行 RTL parity、XRT、PPA，也未加入 activation 或跨核队列。

下一批可从最佳已验证索引版本继续，先建立 GEMM→ReLU 与普通融合的对照，再决定是否将 activation 拆为独立阶段。ReLU 正确性测试应使用正负混合输入；原上游随机输入非负，不能有效检验截断负值的行为。

复现命令、各次原始日志路径及首次 debug 日志路径故障的恢复说明均在 `RESULTS.md`。不必重新执行原 `setup` 或覆盖旧 baseline 记录。

## 13. 2026-10-28：GEMM→ReLU 对照与跨 launch 计数

最新一批见 [`experiments/gemm_relu/RESULTS.md`](../experiments/gemm_relu/RESULTS.md)，逐例表为同目录 `measurements.csv`，汇总为 `analysis.json`。

- 新增 `tests/regression/gemm_relu`，使用单一多入口 device module 比较纯 GEMM、GEMM 后接独立 ReLU、普通融合、独立 loader/compute 加融合 ReLU。另有 grid-stride 独立 ReLU 对照。
- 最终 45 次应用运行、63 次 kernel launch 通过精确数值检查。输入为 `[-16,16]/16` 的正负混合 FP16 数值，并包含零行、零列；三个 ReLU 路径和优化独立 ReLU 的最终输出 hash 一致。
- **计数器生命周期不同**：核心 `cycles/instrs` 跨 launch 累计保留，`ProcessorImpl` 全局内存请求数和 cache tags 每次 launch 重置。核心阶段成本用快照差值，内存流量逐阶段相加；整链周期应等于最终核心累计快照。主机 event profiling 的 ns 不是模拟设备周期。
- 首次 smoke/matrix 曾直接相加核心累计快照，统计重复计入 GEMM，已修正并重跑；正式数据带 `counter_semantics=core-delta_uncore-per-launch_v2`。旧日志仅用于排错。
- 普通融合在六个主配置中均快于 grid-stride 优化后的分离执行。64×32×256 单 CTA 下，PC 融合 18,999 cycles，普通融合 20,825，第二个 seed 复现；128×128×64/256 多 CTA 下 PC 仍慢约 19%/23%。
- 比较口径为模拟设备执行周期，不包含主机提交间隔，且分离 kernel 不保留 warm cache；尚未实现独立 activation warp 或跨核队列，也未进行 RTL parity。

后续优先考虑保持计算单元和 LMEM 不变的 8/16 hardware warp 对照，检查额外 loader 的驻留资源成本。该配置实验尚未执行；需独立 build 并核对全部有效资源。

## 14. 2026-10-28：8/16-warp 对照已完成

以上第 13 节提出的资源对照现已完成。完整结果见 [`experiments/warp_capacity/RESULTS.md`](../experiments/warp_capacity/RESULTS.md)，逐例数据见同目录 `measurements.csv`。

- 独立构建为 `build32-warp-capacity-8` 和 `build32-warp-capacity-16`；旧构建保留。32 个正常用例、4 个 debug 用例均通过。
- 固定 1 核、32 lanes、issue width=4、4 TCU blocks、ALU/LSU/FPU 各 1 block、64 KiB LMEM 和缓存配置。额外固定 LSU pending size=16、AMO RS size=8，防止 NUM_WARPS=16 时默认队列也扩大。
- 完整解析配置只有 NUM_WARPS 不同。应用源码原样复用，两个配置构建出的 device kernel binary 哈希相同。8-warp 结果重现上一轮相同用例。
- 128×128×64 中，8w 普通融合/PC 为 43,658/52,052 cycles；16w 为 44,181/43,031。128×128×256 中，PC 相对普通融合的差距由 +23.34% 缩小到 +0.99%。单 CTA 基本不变。
- Trace 重建确认：所有 warp 同时存活的 CTA 峰值，普通融合由 2→4，PC 由 1→3。dispatcher 逐 warp 分配，另可有部分入驻/部分退出的 CTA；容量除法不是实际全部 live CTA 数的上限。
- 这不是等面积比较。增加 hardware warp 自然增加寄存器/上下文及调度状态容量，尚未测量 PPA 或 RTL parity。

后续仍保留 8-warp 基线和 16-warp 敏感性对照；阶段队列设计应显式考虑 loader/compute/activation 的 warp 预算。独立 activation warp、跨核队列和 KMU placement 尚未实现。

## 15. 2026-10-28：连续输出 tile 的三阶段实验

第 14 节提出的独立 activation warp 现已实现。完整结果见 [`experiments/persistent_pipeline/RESULTS.md`](../experiments/persistent_pipeline/RESULTS.md)，逐例数据和活动窗口见同目录 CSV，原始日志目录见 `analysis.json`。

- 新增独立应用 `tests/regression/persistent_pipeline`，复用固定源码、工具链及 8/16-warp 构建；未修改受版本控制的上游 RTL/SimX。当前只有一个 persistent CTA，连续处理 1/2/5 个 64×32 输出 tile，K=16/64。
- 同一应用比较 4-warp 普通融合、4 compute + 1 loader 两阶段、4 compute + 1 loader + 1 activation 三阶段。A/B 使用双缓冲，三阶段另有 1/2 格完整 FP32 输出队列，CTA 共享内存分别为 6/14/22 KiB。
- 输入 ready/free、输出 ready/free 和尾部排空协议均已实现。沿用现有 BAR 的 LSU drain 语义；未使用零开销的理想队列。只有完成 K 归约的结果进入 ReLU。
- 最终 44 次 SimX 运行通过正负混合精确输入、CPU reference 与输出 hash 检查，包括 18 个主用例、输出深度、第二个 seed、8-warp 对照、2 个 debug trace 和 12 次关键用例重复。
- 公平性修正：普通融合也跨输出 tile 预取 A/B。修正前的首批日志保留但不纳入正式比较；最终结果 JSON 带 `ordinary_prefetch=cross-output-tile`。
- 16-warp、5-tile、K=64 的普通融合/两阶段/三阶段周期为 28,396/26,660/58,421。关键用例各重复三次，周期一致。输出队列从 1 格增到 2 格后为 58,222（约 -0.34%），但增加 8 KiB 共享内存。
- 三阶段 trace 中有 71 个 activation 执行事件位于下一 tile 的计算活动窗口，11 个不同 tick 同时出现前一 tile 的 activation 与下一 tile 的 TCU 事件。真实阶段重叠已观察到，但这不是利用率指标。单个 activation warp 的活动跨度较长，尚未隔离共享内存、LSU 排空、队列和调度的各项成本。
- RTL 首次模型 C++ 编译超过 600 秒上限，进程组已结束，没有进入 kernel 数值检查。日志提示 jobserver 不可用并退回 `-j1`。脚本现已分离模型构建与数值 smoke，并提供 `--rtl-build-timeout`（默认 1800 秒）；调整后的 RTL 流程本批尚未重跑。不能将构建超时视为 kernel 死锁，也不能将 SimX PASS 视为 RTL PASS。

下一批优先考虑 1/2 个 activation warp 的资源分配与共享内存交接成本对照；保留跨输出预取的强融合基线。多个 persistent CTA、跨核 L2 原子队列、KMU placement、完整 WASP 编译器/RFQ 和 PPA 仍未完成，不应由本批 SimX 结果推断。

## 16. 2026-10-29：activation 分配与循环展开对照完成

第 15 节提出的 1/2 个 activation warp 对照现已完成。新增独立应用 `tests/regression/activation_balance`，旧 `persistent_pipeline` 源码与结果保留。完整结果见 [`experiments/activation_balance/RESULTS.md`](../experiments/activation_balance/RESULTS.md)。

- 同一应用重跑普通融合、两阶段，以及 activation warp 数 1/2 × 循环展开组数 1/4 的四种三阶段组织。计算仍为 4 warp、加载仍为 1 warp，三阶段 CTA 为 6/7 warp；执行单元和缓存等容量保持不变。
- 最终 78 次 timing SimX 运行全部通过精确 CPU reference、正负/零值覆盖与输出 hash 检查：6 smoke、36 主用例、14 阶段位置/seed/深度控制、12 关键重复、4 trace、6 个 8-warp 对照。
- 全部 8/16-warp、normal/debug 运行使用同一源码和相同 device binary 哈希。普通融合和两阶段六个主形状的周期、指令、全局请求计数与上一轮完全一致。
- 16 hardware warp、5 个输出 tile、K=64、C depth=1：普通融合 28,396，两阶段 26,660；三阶段 A1/U1=57,619、A1/U4=41,232、A2/U1=35,128、A2/U4=27,586 cycles。A2/U4 相对本轮 A1/U1 降低约 52.1%，比普通融合少约 2.85%，仍比两阶段多约 3.47%。它使用 7 warp、14 KiB SMEM，两阶段为 5 warp、6 KiB。
- A1/U1 模板循环与旧三阶段代码生成/布局不同，不能将旧三阶段的微小周期变化算作本轮优化收益。主要比较均在新应用内部进行。展开 4 组是循环展开，没有扩大 SIMD 宽度。
- 4 份 trace 均为 40 次 DXA transfer、2560 个 WGMMA compute event、320 个 FMIN_MAX event，且所有 activation 在对应完整 K 归约后开始。activation 首末 FMIN_MAX 跨度的中位数从 A1/U1 的 9581 降到 A2/U4 的 3489 cycles；这不是纯 FPU 延迟或利用率。
- 把 ReLU 移回计算 warp、让输出 warp 只复制结果，没有稳定改善。该对照保留 C 队列和全部最终计算，说明不能把主要开销简单归给 fmax；尚未分离访存、握手与调度各项成本。
- 增加输出队列到 depth2 没有改变三阶段仍慢于两阶段的结论。8-warp 单 CTA 结果接近 16-warp（A2/U4 为 27,599 cycles），但尚未测试多 CTA 的资源竞争。

下一批优先检验多个 persistent CTA 的进展和驻留代价，每个 CTA 连续处理多个 tile，并保留两阶段/普通融合对照。跨核队列与 KMU placement 仍在后续范围。本轮没有重跑此前超时的 RTL 构建，没有新增 RTL/XRT/PPA 结论。

## 17. 2026-10-29：多个 persistent CTA 的扫描已完成

第 16 节提出的多 CTA 检查现已完成。见 [`experiments/multi_cta/RESULTS.md`](../experiments/multi_cta/RESULTS.md)，逐例数据、逻辑 CTA 活动及驻留事件也在同目录。没有修改 `activation_balance` 应用；源码和 device binary 与上一轮完全一致。

- 主矩阵为 M=256、N=128，共 16 个输出 tile，K=64/256；CTA 数为 1/2/4/8，因此每 CTA 连续处理 16/8/4/2 个 tile。比较普通融合、两阶段、三阶段 A1/U4、A2/U4，在 8/16 hardware warp 下运行。
- 最终 148 次 timing SimX 运行全部通过：64 主扫描、4 smoke、24 不均匀分配/seed 检查、32 关键重复、16 最佳 CTA 数第二 seed 复核、8 trace。
- 不均匀分配包括 9 tile/4 CTA 和 15 tile/8 CTA，覆盖奇数队列代数及单输出 CTA 的尾部排空；仍为完整 tile，没有验证部分边界 tile。
- 按每种方法分别选择扫描中的最佳 CTA 数：16-warp、K=64，普通融合 85,132（g=4）、两阶段 83,192（g=2）、最佳三阶段 A1/U4 80,423（g=2）。三阶段分别少约 5.53%/3.33% 周期，第二个 seed 重现。
- 8-warp、K=64 的最佳三阶段 A2/U4 为 80,797（g=1）；但把同一方法增到 g=8 后为 110,447。8-warp、K=256 的最佳普通融合为 242,342，最佳三阶段为 267,631，三阶段仍慢约 10.44%。不能只选对流水线有利的 CTA 数作为普通融合对照。
- 16-warp、K=256 的最佳三阶段为 236,663，最佳两阶段为 237,088，差约 0.18%，当前只视为接近。
- g=8 的 trace 显示：8→16 hardware warp 后，全部 warp 同时存活的 CTA 峰值，普通融合 2→4、两阶段 1→3、A2/U4 三阶段 1→2。部分入驻 CTA 也存在；峰值不是时间平均 occupancy。
- g=2、16-warp 的 A1/A2 都同时完整入驻两个 CTA，分别用 12/14 warp。因此 A1 较快不能仅用更多完整驻留 CTA 解释，尚未隔离调度、发射与 LSU 等竞争成本。
- 全部 8 份 K=64 trace 均为 128 次 DXA transfer、8192 个 WGMMA compute event、1024 个 FMIN_MAX event；按逻辑 CTA/rank 核对完整 K 后才激活、最终所有 warp 分配/退出均配对。slot ID 可复用，分析以 blockIdx 识别逻辑 CTA。

本批结果把研究问题进一步收敛为：结合 workload、阶段 warp 数与 CTA 数选择合适的执行方式。下一步可形成小型选择表并在新形状上验证泛化；跨核队列/KMU placement 仍需单独实现和验证。所有结果限于当前核内 SimX，没有新增 RTL/XRT、FPGA、PPA 或完整 WASP 自动编译器/RFQ 结论。

## 18. 2026-10-29：冻结配置表的未见形状检查完成

第 17 节提出的选择表与新形状验证现已完成，见 [`experiments/policy_generalization/RESULTS.md`](../experiments/policy_generalization/RESULTS.md)。没有修改 `activation_balance` 应用或 device binary。

- 用此前四份正式结果中的 92 条测量建立 11 个 `(hardware warps, output tile 数 T, K)` 锚点。按 log2(T/K) 距离查最近锚点；在最快周期 1% 内优先少 warp/SMEM。该 1% 是资源取舍阈值，没有统计显著性或 PPA 含义。
- 首次新形状运行前冻结 `policy.json`、`holdout_manifest.json` 和 12 个预测，policy SHA256 为 `e8958a4000df35b8ce4493d878f4f77975ed4e0295a83e0a9061383f04714d9b`。本版没有用测试网格结果更新规则。
- 六个未参与建表的形状：64×128×64、256×32×64、256×64×128、192×128×64、384×128×256、320×128×192，分别测 8/16 hardware warp。两套硬件的规则预测先执行，再做固定候选网格。
- 最终 214 次 timing SimX 检查通过：12 规则执行、176 评估网格、26 个第二 seed 去重配置。相同形状/seed 的输出 hash 一致；规则配置在网格中重跑的核心与全局内存计数也一致。
- 12 个组合中，6 个规则选择达到网格最小周期，8 个在 1% 内，10 个在 5% 内。相对网格最优的周期比几何均值为 1.0160，最坏为 1.0788。相对调好 CTA 数的普通融合，规则在 8/12 个组合中严格更快，几何周期比约 0.9908；这是这组小样本等权统计，不是通用 benchmark 收益。
- 主要失误为 256×64×128：8-warp 选两阶段 g=1，77,599 cycles，网格最佳普通融合 g=4 为 72,613（约 +6.87%）；16-warp 规则为 77,566，最佳普通融合 g=2 为 71,899（约 +7.88%）。诊断显示，8-warp 更需要改执行方式，16-warp 的主要损失来自 CTA 数选择。
- 新形状 192×128×64 继续出现三阶段优势：8-warp 网格最佳三阶段/普通融合为 61,301/63,759；16-warp 为 61,179/65,236。冻结规则在 16-warp 选三阶段 62,330，仍比最佳普通融合少约 4.45% 周期。
- 当前特征只有 T、K 和 hardware warp。小形状训练锚点只测过 g=1，候选覆盖有限；M/N、footprint 等特征及更完整训练网格都仍是待验证方向。仅靠本轮结果不能断言某一新特征已能修复失误。

若改进 v2，应把本批作为可用训练/诊断数据，同时另选新的独立验证形状；不要修改冻结 v1 后继续把同一六个形状当作新版未见测试。本轮只实现 host 实验脚本查表选择既有 kernel/CTA 数，没有自动阶段切分、RFQ 或跨核队列，也没有新增 RTL/XRT/FPGA/PPA 验证。

## 19. 2026-11-02：v2 补充训练与几何特征对照完成

第 18 节提出的 v2 检查已完成。见 [`experiments/policy_v2/RESULTS.md`](../experiments/policy_v2/RESULTS.md)。v1 文件与此前结果保留，应用源码和 device binary 与此前一致。

- 新训练/验证计划在训练前冻结。补测 128×128×96、128×128×128、256×128×128，每个形状在 8/16 hardware warp 下各测四种组织和 g=1/2/4/8，共 96 次训练运行。
- 扩充训练集为 v1 历史92条 + v1测试网格176条 + 本批96条，共364条测量、29个精确 M/N/K 锚点。v1 原测试集已经转作 v2 训练数据，不再是独立验证。
- 两版 v2 使用同一扩充训练集、相同1%资源取舍阈值，只改变距离：v2-TK 用 T/K，v2-MNK 分别用 M/N/K。按精确形状选锚点配置，避免跨不同长宽比直接最小化绝对周期。v2 SHA256 为 `9ae4a5e4fdb3972572bf2d7896c8f5f54e53fb95810068b82b4493133e037314`。
- 新验证形状为 128×96×128、192×64×128、256×96×144、192×128×112、384×96×160、256×192×224。两套硬件的三种规则预测先执行，再测候选网格；规则没有根据这些测试回填。
- 本批 323 次 timing SimX 运行全部通过：96 训练、21 个去重预测配置、176 评估网格、30 个第二 seed 去重复核。全部输出 hash 与 CPU reference 检查通过，相同配置重跑的周期、指令和全局请求数一致。
- 在同一批12个新组合上，v1/v2-TK/v2-MNK 的几何平均额外周期分别为 3.03%/0.94%/1.30%，最坏额外周期为 7.77%/2.73%/4.37%。落在候选最优5%内的组合为9/12、12/12、12/12。不能用旧测试集的 v1 1.60%/7.88% 与本批 v2 直接比较。
- v2-TK 相对本批 v1 在7个组合更快、4个相同、1个回退约0.55%（16-warp、256×96×144）。它相对调好 CTA 数的普通融合总体几乎持平（几何周期比1.00006）。
- 两版 v2 只有一个组合的预测不同。8-warp、192×64×128 的 TK 选融合g=2、54,459 cycles；MNK 选g=4、56,841，约慢4.37%。当前不足以证明几何特征普遍优劣，至少说明近邻几何不保证 CTA 数可以直接照搬。

若继续改规则，需要再次另选独立验证形状。当前 v2/计划/预测文件保持冻结。本批没有修改上游应用、RTL/SimX，没有新增 RTL/XRT、FPGA、PPA、自动阶段切分/RFQ 或跨核队列/KMU placement 结果。日志目录按 UTC 命名，报告日期采用用户本地日期。

## 20. 2026-11-03：核内流水线的成本归因完成

完整结果见 [`experiments/cost_attribution/RESULTS.md`](../experiments/cost_attribution/RESULTS.md)。本批没有继续调整冻结policy，重点解释已有两个场景：M256/N128，16w/K64/g2 的三阶段 A1/U4获益，以及8w/K256/g4 的三阶段 A2/U4退化。

- 新增独立应用 `cost_pipeline` 和容量补充对照 `cost_capacity`；原 `activation_balance` 与受版本控制的Vortex源码保持不变。普通融合、两阶段和原三阶段均精确复现此前计数。两份新binary分别记录哈希，没有把它们视为相同binary。
- 最终90次SimX运行通过数值检查：87次正式测量/复核/trace，加3次开发smoke。20份trace来自独立的日志观察副本；周期、指令、全局请求及18项core计数与对应原模型运行一致。关键对照包含重复和第二seed，检查器未放宽。
- 只把两阶段SMEM预留从6增到14KiB，两组周期和core计数都不变。三阶段确实增加128KiB C写+128KiB C读，共256KiB额外共享内存有效载荷，但不能把这一流量直接换算成固定周期损失。
- 16w A1限制计算/activation重叠后从80,423增到95,761 cycles；8w A2从285,096增到313,471。相同CTA跨tile重叠归零。编译器仍会改变少量指令/参数回读，因此属于可执行调度消融；16w A2的同类控制反而略快，不能普遍化为“重叠越多越好”。
- 新的普通融合容量对照保持同一计算算法、6KiB SMEM和匹配末尾join，只多保留一个闲置warp。8w/g4下4→5 warp从248,911增到357,772 cycles，约+43.73%；g1只增约0.07%。Trace中平均完整计算组CTA从1.994降到1.000，支持上下文容量门槛的因果作用。该退化比例不等于实际三阶段的损失占比。
- 主场景8w普通融合/两阶段/三阶段的平均完整计算组CTA约为1.964/1.003/1.001；16w三者约为1.962/1.953/1.975。平均值按精确dispatch/exit tick积分，只表示各组4个compute warp同时存活，不是就绪率、执行占用或TCU利用率。
- 记录了BAR进入前的全核LSU drain与进入后的barrier等待。两阶段→三阶段的gate阻塞tick在16w从18,488降到3,463，在8w从19,122降到8,623，尽管三阶段BAR更多。8w较多input-ready等待还包含参与者未到达，不能统一归为DXA内存等待或activation算术瓶颈。gate ticks、累计warp等待、scoreboard计数不能与kernel周期直接相加。

当前证据支持：专用角色的重叠收益，需要与上下文容量、同步节奏和额外动态工作一起考虑。若继续设计，优先在8w容量紧张时探索角色复用/较少专用warp；若继续解释16w获益点，再针对DXA/LSU/TCU供给时间线做一项小对照。本批不包含新增RTL/PPA/FPGA、完整WASP自动编译器/RFQ或跨核队列实现。

开发时的首次C回读smoke因WMMA/WGMMA累加器布局不同而失败，修正为WGMMA n-major逆映射后复核通过；失败日志保留，不纳入正式数据。
