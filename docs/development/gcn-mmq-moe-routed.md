# GCN MoE MMQ dispatch by expert load

This is an experimental follow-up to `feature_GCN_MOE_MMQ_DIAGNOSTICS`. GPU correctness and performance must be checked on the MI50 before retaining this policy. There is no model-name condition.

## Evidence

The supplied `mmq-diagnostic-v2` run contains 160 passing numerical checks on Q8_0 and IQ4_XS, plus 840 operation timings: 28 cases, 10 target widths and three repetitions. The largest max/min timing spread within one case and target is about 3.1%.

Representative medians in microseconds, lower is better:

| Type | M | K | Tokens | Experts | Routing | J=8 | J=16 | J=32 | Old auto |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Q8_0 | 1536 | 512 | 32 | 40 | dispersed | 158.84 | 317.11 | 204.26 | 317.13 |
| Q8_0 | 1536 | 512 | 32 | 40 | concentrated | 110.38 | 147.29 | 65.53 | 146.41 |
| Q4_K | 2048 | 512 | 128 | 256 | dispersed | 629.55 | 796.34 | 1185.62 | 629.94 |
| Q4_K | 2048 | 512 | 128 | 256 | concentrated | 669.78 | 310.73 | 222.18 | 670.20 |

All cases use eight experts per token. Dispersed routing uses seeded random IDs. Concentrated routing sends every token to the same eight experts. Old auto uses the capped 4/3 average.

Q8_0 J=16 is slower than both J=8 and J=32 in every measured Q8_0 case. This does not isolate the cause inside that kernel. At identical tensor dimensions, the preferred width also changes with the routing. A single multiplier cannot capture that difference. These synthetic measurements do not prove the actual routing distribution of a model.

## Candidate policy

Keep the existing kernels and select from the supported widths 8, 32, 64 and 128. Skip unavailable configurations and configurations that exceed the shared-memory limit. Stop when a width covers the full token count. Each expert uses the smallest retained width that covers its actual count, or the largest retained width if more column tiles are needed.

The existing GPU `expert_bounds` array provides those counts. Each width launches once; blocks reject experts outside their interval before shared-memory initialization. The intervals are disjoint and cover every nonempty expert. The CPU does not read the counts, synchronize, or time kernels. Captured graphs use the current counts on every replay.

The compact launch grid uses the virtual prefix `P[e] = floor(expert_bounds[e] / J) + e`. A binary search maps a column block to its expert and local tile. Each expert reserves `ceil(count / J)` tiles or one extra tile. Extra tiles return without writing. The total column-block count is `floor(total_assignments / J) + expert_count`, including padding for empty experts. This replaces `ceil(token_count / J) * expert_count` when smaller and within the grid Y limit. Otherwise the original grid remains valid, with the same interval filter.

The largest retained width handles arbitrarily concentrated routing with multiple tiles. The grid never assumes that an average is a maximum. Partial columns in the destination-ID load are guarded.

This initial width family limits dispatch to at most four launches and avoids J=16. It is not a proven optimum for all quantizations, shapes or GCN devices. The measurements also contain cases where J=48 wins. Extra launches, binary searches, and compiler register allocation can offset the benefit; the same-build controls below measure that risk. No kernel configuration table is changed.

## Controls

- `GGML_CUDA_MMQ_MOE_ROUTED` unset or `=1`, with `GGML_CUDA_MMQ_MOE_NCOLS` unset: new dispatch.
- `GGML_CUDA_MMQ_MOE_ROUTED=0`: previous 4/3 policy.
- `GGML_CUDA_MMQ_MOE_NCOLS=0`: original full-token selector, bypassing routed dispatch.
- `GGML_CUDA_MMQ_MOE_NCOLS=8..128`, multiples of eight: existing single-width diagnostic target, bypassing routed dispatch.
- `GGML_CUDA_MMQ_MOE_TRACE=1`: include each kernel's expert-count interval in the existing configuration trace.

Dense MMQ and other architectures do not enable expert filtering. Use the controls with the same binary to detect costs introduced by the kernel changes themselves. Keep the old diagnostic binary or old results for comparison too.

## Validation

Reuse the already configured diagnostic build directory after switching to this branch. The script's default path stays the same intentionally:

```sh
git fetch origin feature_GCN_MOE_MMQ_ROUTED
git switch --track origin/feature_GCN_MOE_MMQ_ROUTED
cmake --build build-feature_GCN_MOE_MMQ_DIAGNOSTICS --target test-backend-ops llama-bench -j 2

HIP_VISIBLE_DEVICES=0 GGML_CUDA_MMQ_MOE_ROUTED=1 python3 scripts/bench-gcn-mmq-moe.py \
    --mode test --params '^mmq_moe=1,' --widths legacy,auto --out mmq-routed-correctness

HIP_VISIBLE_DEVICES=0 GGML_CUDA_MMQ_MOE_ROUTED=1 python3 scripts/bench-gcn-mmq-moe.py \
    --mode perf --params '^mmq_moe=1,' --widths legacy,auto --out mmq-routed-performance

HIP_VISIBLE_DEVICES=0 GGML_CUDA_MMQ_MOE_ROUTED=0 python3 scripts/bench-gcn-mmq-moe.py \
    --mode perf --params '^mmq_moe=1,' --widths auto --out mmq-average-control

tar -czf mmq-routed-results.tar.gz mmq-routed-correctness mmq-routed-performance mmq-average-control
```

Expect 76 correctness cases per target: the original 28 shapes/routings plus 48 cases for Q8_0, IQ4_XS and Q4_K at partial row tiles and token counts 17, 33, 65 and 129. Both routing modes are covered. The performance suite keeps the same 28 cases to permit direct comparison. `legacy` selects the original policy; `auto` selects routed dispatch unless `GGML_CUDA_MMQ_MOE_ROUTED=0` is set.

Host checks compiled the actual tile-mapping helper and launch-grid calculation with undefined-behavior sanitization. They checked 12,213 count distributions and 183,195 width sets, including empty experts, skew, missing widths, and the grid-limit fallback. Every required tile was covered once. These checks do not compile HIP or validate GPU arithmetic.

After the numerical suite passes, compare operation timings with both controls. Then rerun the original Granite IQ4_XS and Q8_0 ubatch sweep and the Qwen35 MoE Q4_K sweep with `llama-bench`, without tracing. Recovery of the Granite regressions and retention of the Qwen gains remain acceptance requirements, not measured outcomes of this branch.
