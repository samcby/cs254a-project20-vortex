# Run an individual study

Use the pinned upstream and toolchain from [the setup instructions](../README.md#build-vortex-and-rerun-the-cost-study). Run commands from the root of a disposable copy of this repository so the original measurements and frozen inputs remain available for comparison.

```bash
python3 install_apps.py
```

The archived JSON includes the exact historical commands and local paths for provenance. The analyzers resolve archived logs relative to the checkout. The launch scripts resolve the Vortex workspace using `PROJECT20_WORK_ROOT` and the toolchain using `PROJECT20_TOOLCHAIN_DIR` when set. The original experiment reports contain each batch's commands and configuration.

| Study | Inputs to keep when rerunning an individual batch |
| --- | --- |
| [Producer–consumer](../experiments/pc_pipeline/RESULTS.md) | Run `micro`, `gemm`, `diagnostic`, `indexing`, then `trace` for the complete set. |
| [GEMM/ReLU](../experiments/gemm_relu/RESULTS.md) | Uses the `sgemm_tcu_wg_pc_mask` source and `pc_indexing_results.json`. |
| [Warp capacity](../experiments/warp_capacity/RESULTS.md) | Uses the validated `gemm_relu` source and its analysis JSON. |
| [Persistent pipeline](../experiments/persistent_pipeline/RESULTS.md) and [activation balance](../experiments/activation_balance/RESULTS.md) | Use the preceding validated source hashes and configuration records included in the checkout. The optional `--phase rtl` is a separate build and correctness attempt. |
| [Multi-CTA](../experiments/multi_cta/RESULTS.md) | Uses the saved activation-balance source and validated build configuration. |
| [Policy v1](../experiments/policy_generalization/RESULTS.md) and [v2](../experiments/policy_v2/RESULTS.md) | Keep the committed training records, frozen plan, policy and manifest together. Run the `policy`, `oracle`, and `validate` phases in order for each warp count. For v2, use a separate checkout when rerunning `train` because the committed v2 policy identifies its original training files by byte digest. |
| [Cost attribution](../experiments/cost_attribution/RESULTS.md) | Use `python3 reproduce.py full` to compare every new run with the archived record and recompute the trace summaries. |

Historical JSON records include time stamps and machine paths. If retraining a selector on new data, use a separate copy and generate a new plan, policy, and manifest from the new records in the documented order.
