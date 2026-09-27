# GCN MoE MMQ tile diagnostics

This branch measures the tile-selection problem. It does not claim to fix the regressions or change the default 4/3 policy.

The current selector minimizes ceil(ncols_opt / J). A candidate also changes I, thread count, shared memory use and register use. Average tokens per expert alone cannot describe those costs or an uneven routing distribution.

The reported MI50 results include a Q8_0 regression at ubatch 32 (871.27 to 640.90 tokens/s, -26.44%) and an IQ4_XS regression at ubatch 128 (1976.51 to 1914.38 tokens/s, -3.14%). For 40 experts and top-8 routing, the selected widths change from 32 to 16 and from 64 to 40, respectively. These are candidate causes, not a per-kernel timing diagnosis.

## Controls

Both controls apply only to GCN MoE MMQ.

- GGML_CUDA_MMQ_MOE_NCOLS unset: existing 4/3 policy.
- GGML_CUDA_MMQ_MOE_NCOLS=0: baseline selection using the full token count.
- GGML_CUDA_MMQ_MOE_NCOLS=8..128, multiples of 8: set the target width.
- GGML_CUDA_MMQ_MOE_TRACE=1: report the actual configuration once per shape and template instance, up to 256 shapes per thread.

The override is a target width, not a forced kernel. For example, target 128 can select J=64 when the quantization has no wider configuration. The log contains the actual I, J, threads, shared memory and fallback selection. The launch grid still uses the full token count so concentrated routing remains covered.

No GPU synchronization, timing, routing readback or runtime calibration is added to inference. Tracing is disabled by default. Run end-to-end benchmarks without tracing.

## Build and run

Reconfigure the existing HIP build so its ROCm settings are retained:

    cmake -S . -B build-feature_GCN_MOE_MMQ_TILES -DLLAMA_BUILD_TESTS=ON
    cmake --build build-feature_GCN_MOE_MMQ_TILES --target test-backend-ops llama-bench -j 8

    HIP_VISIBLE_DEVICES=0 python3 scripts/bench-gcn-mmq-moe.py --mode test --out mmq-correctness
    HIP_VISIBLE_DEVICES=0 python3 scripts/bench-gcn-mmq-moe.py --mode perf --out mmq-performance

The default slice covers Q8_0 and IQ4_XS at 32 and 128 tokens, with 40 experts, top-8 routing, and both projection directions. Each shape has seeded random routing and concentrated routing. Concentrated routing sends every token to the same eight experts.

The script varies candidate order, keeps raw output, checks that GCN MoE MMQ was reached, and produces samples.csv and summary.csv. Performance runs are repeated three times. A positive speed_vs_legacy_pct means faster than the baseline selection. An existing output directory is rejected so prior results are not overwritten.

Run the larger expert-count Q4_K control separately:

    HIP_VISIBLE_DEVICES=0 python3 scripts/bench-gcn-mmq-moe.py --mode perf \
        --params '^mmq_moe=1,.*n_mats=256,' --out mmq-control

These are synthetic operation timings including dispatch, routing and quantization. They do not measure a model's actual routing distribution, and they do not replace the original Granite and Qwen llama-bench tests.

## Durable selection

Use these measurements to distinguish a poor kernel configuration from a poor choice among configurations. A slow J=16 configuration may need a kernel/configuration improvement instead of a routing multiplier.

For a general selector, compare supported configurations using quantization, M, K, token count, expert count, top-k and device resources. Relevant quantities include row tiles ceil(M/I), active column tiles sum_e ceil(tokens_e/J), padded arithmetic and repeated weight loads. The launch_bounds occupancy value is a compiler target, not measured residency. A cost model needs measurements; assigning arbitrary coefficients would replace one unvalidated heuristic with another.

Keep the baseline as a candidate. Select an alternative only where repeated measurements support it, validate both routing distributions and nearby shapes, then rerun the original end-to-end tests. Retain baseline behavior outside the validated domain. Do not use model names.

Runtime autotuning is another option, but it requires a separate design for graph capture, startup cost, representative routing, cache lifetime and concurrent contexts. Timing one first call and caching its winner is not sufficient.

The acceptance gate is numerical agreement with the CPU reference, recovery of the two Granite regressions, and retention of the measured Qwen gains across the full ubatch sweep. Full HIP compilation and GPU validation must run on the target machine.
