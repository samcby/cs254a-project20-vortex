# Vortex Dataflow Pipelines · CS 254A Project 20

This repository contains our single-core timing SimX experiments on load–GEMM–ReLU pipelines in Vortex v3.0. The kernels use DXA, asynchronous barriers, and WGMMA. They compare ordinary fusion with two-stage and three-stage pipelines, then investigate how shared-memory traffic, cross-tile overlap, and CTA residency affect the outcome.

## Results at a glance

| Experiment, M = 256 and N = 128 | Comparison | SimX cycles |
| --- | --- | ---: |
| 16 hardware warps, K = 64, 2 CTAs | Two stages → three stages with one activation warp and four-way unrolling | 83,192 → 80,423 |
| Same three-stage case | Move the output-slot wait earlier to restrict cross-tile overlap | 80,423 → 95,761 |
| 8 hardware warps, K = 256, 4 CTAs | Matched ordinary-fusion control, add one idle warp to each four-warp CTA | 248,911 → 357,772 |
| Same idle-warp control with 1 CTA | Add one idle warp | 380,910 → 381,191 |

The three-stage design reduces cycles by 3.33% in the selected 16-warp case. In the matched 8-warp capacity control, adding the idle warp lowers the average number of resident complete compute groups from about 1.99 to 1.00. Increasing unused shared-memory reservation from 6 to 14 KiB leaves cycles unchanged in the two tested configurations. Actual three-stage C exchange adds 128 KiB of shared-memory stores and 128 KiB of loads. The cost study contains **90 numerically checked runs, 87 formal measurement rows, and 20 timing traces**. [Cost study report](experiments/cost_attribution/RESULTS.md) · [CSV](experiments/cost_attribution/measurements.csv) · [Details](experiments/cost_attribution/analysis.json)

The configuration selector was also checked on held-out shapes. The v2-TK policy was within 5% of the fastest point in the measured candidate grid for all 12 shape/warp configurations in the v2 study. [Policy v2 report](experiments/policy_v2/RESULTS.md)

## Repository layout

| Path | Contents |
| --- | --- |
| [experiments/](experiments/) | Kernel and host code, experiment runners, reports, and summary CSVs |
| [results/](results/) | Run records by study, including command lines, configurations, hashes, and counters |
| [archive/logs/](archive/logs/) | Original logs, traces, binaries, and source snapshots by study and run date |
| [scripts/](scripts/) | Vortex setup, application installation, and shared paths |
| [docs/](docs/) | Proposal, WSL notebook, and individual run instructions |

## Experiment files

| Study | Report and files |
| --- | --- |
| DXA producer–consumer and indexing controls | [pc_pipeline](experiments/pc_pipeline/RESULTS.md) |
| GEMM and ReLU kernel launches and ordinary fusion | [gemm_relu](experiments/gemm_relu/RESULTS.md) |
| Controlled 8/16-warp resource sweep | [warp_capacity](experiments/warp_capacity/RESULTS.md) |
| Persistent three-stage pipeline and output buffer depth | [persistent_pipeline](experiments/persistent_pipeline/RESULTS.md) |
| Activation warp count and loop unrolling | [activation_balance](experiments/activation_balance/RESULTS.md) |
| Persistent CTAs and tile traversal | [multi_cta](experiments/multi_cta/RESULTS.md) |
| Frozen configuration rule and held-out shapes | [policy_generalization](experiments/policy_generalization/RESULTS.md) |
| Expanded training data and feature comparison | [policy_v2](experiments/policy_v2/RESULTS.md) |
| Cost attribution, including occupancy and synchronization controls | [cost_attribution](experiments/cost_attribution/RESULTS.md) |

The archive includes failed development and RTL build attempts with their original statuses. The trace runner generates the C++ observer overlay from the pinned Vortex source. The performance results are from single-core timing SimX. Cross-core queues, KMU placement, RTL execution and PPA remain follow-up work. [Observer source](experiments/cost_attribution/simx-observer/) · [Overlay generator](experiments/cost_attribution/prepare_cost.py)

## Reproduce

### Recompute the reports from the saved data

Use Python 3.10 or later. The archive pass needs the compressed logs and Python standard library. It recomputes the nine analyses in a separate directory and checks the cost-study rows and trace summaries against the committed results.

```bash
git clone https://github.com/samcby/cs254a-project20-vortex.git
cd cs254a-project20-vortex
python3 reproduce.py archive
```

### Build Vortex and rerun the cost study

Run the following in Ubuntu 22.04 x86-64 or its WSL2 distribution. The compiler and simulator paths should be on the Linux filesystem without spaces. The complete archive is about 630 MiB and full reruns need several additional GB for tools, builds, and traces.

```bash
sudo apt-get update
sudo apt-get install -y git wget python3 build-essential cmake ccache pkg-config \
  zlib1g-dev libtinfo-dev libncurses-dev uuid-dev libboost-serialization-dev \
  libpng-dev libhwloc-dev bzip2 unzip

export PROJECT20_WORK_ROOT="$HOME/cs254a/project20"
mkdir -p "$PROJECT20_WORK_ROOT"
git clone --depth=1 --branch v3.0 --recurse-submodules --shallow-submodules \
  https://github.com/vortexgpgpu/vortex.git "$PROJECT20_WORK_ROOT/vortex"
test "$(git -C "$PROJECT20_WORK_ROOT/vortex" rev-parse HEAD)" = \
  7fe19369587c087f902c3e2bc2add612aa910a95

(
  set -e
  mkdir -p "$PROJECT20_WORK_ROOT/vortex/build32"
  cd "$PROJECT20_WORK_ROOT/vortex/build32"
  ../configure --xlen=32 --tooldir="$PROJECT20_WORK_ROOT/tools-v3.0.1"
  ./ci/toolchain_install.sh --llvm --riscv32 --libcrt32 --libc32 --verilator
  make -j4 third_party
  make -j4 software
)

# Back in this repository's root directory.
python3 reproduce.py quick   # 23 runs across the two headline configurations
python3 reproduce.py full    # 90 runs including trace and repeat batches
```

Both profiles compare each new run's configuration, binary SHA-256, numerical output hash, cycles, instructions, global memory requests, and 18 core counters with the archive. `full` also matches the derived trace summaries. First-time builds and traces take longer than individual measured kernels. `--out /path/to/new-directory` selects the output location. `PROJECT20_TOOLCHAIN_DIR` points at an existing `v3.0.1` toolchain if you have one. [Upstream Vortex installation guide](https://github.com/vortexgpgpu/vortex/blob/7fe19369587c087f902c3e2bc2add612aa910a95/docs/install_vortex.md)

Release verification on Ubuntu 22.04.5 under WSL2 used a fresh upstream clone, freshly downloaded tools, and new build directories. All 90 cost-study executions and the 20 derived trace summaries matched. All nine reports were also recomputed from the packaged data, and the 11 selected 8-warp policy v2 runs matched their archived counters and output hashes.

For the original experiment-by-experiment commands, work in a separate copy of this checkout and run `python3 scripts/install_apps.py` after setup. This places the nine apps under the pinned Vortex regression tree. Follow the commands in the study reports from the root of that copy. The runners write updated records under `results/<study>/` and new run batches under `archive/logs/<study>/`. Re-run the configuration selector's freeze steps only when building a new model from new training records. [Individual run notes](docs/individual-runs.md)

## Project documents and references

- [Original proposal source with student IDs removed](docs/proposal/main.tex) · [Lab notebook and original WSL setup notes](docs/WSL_GUIDE.md)
- [WASP, HPCA 2024](https://doi.org/10.1109/HPCA57654.2024.00086) · [Kitsune, ACM TACO 2025](https://doi.org/10.1145/3777466)
- [Vortex v3.0](https://github.com/vortexgpgpu/vortex/releases/tag/v3.0) · [Vortex Apache 2.0 license](licenses/Vortex-Apache-2.0.txt)

The public archive contains project-generated materials and links to third-party publications. Course handouts, publisher PDFs, and the original submission with student IDs are kept with the course records.
