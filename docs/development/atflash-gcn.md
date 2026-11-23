# Experimental ATFlash on AMD GCN

This branch provides an opt-in experiment for per-RoPE-wavelength attention pruning in the existing ggml Flash Attention operator. The default behavior is unchanged unless the environment variable is set.

## Run

Build the branch with the project's HIP configuration and enable Flash Attention. Set a positive integer in the range 1-64 (2 is the ATFlash paper default; 1 corresponds to the WavePrune one-period cutoff):

```bash
LLAMA_ATFLASH_PERIODS=2 HIP_VISIBLE_DEVICES=0 ./build-feature_GCN_ATFLASH/bin/llama-bench \\
    -m "$MODEL" -ngl 999 -sm none -fa on -p 2048 -n 0 -b 2048 -ub 512 -r 5 | tee bench-atflash-pp2048.log
```

For a baseline, run exactly the same command with `LLAMA_ATFLASH_PERIODS` unset. Validate quality with `llama-perplexity` and long-context retrieval tests before drawing conclusions from throughput.

## Supported scope

- Experimental HIP GCN devices without MFMA or WMMA (such as gfx906), with the tiled Flash Attention kernel.
- Standard full-dimension RoPE in either normal or NeoX layout, without frequency scaling.
- FP16 K and V cache, a single sequence stream, and a head dimension up to 256.
- No retraining is performed: attention logits change and model quality may regress.
- No compact KV cache, paged-KV support, hybrid sliding-window attention, or support for other GPU backends.

Other configurations are not validated. The `GGML_OP_FLASH_ATTN_EXT` CPU path includes a slow reference implementation for correctness testing.

## Validation

ATFlash tests in `test-backend-ops` are opt-in because GPU backends other than GCN do not implement this experimental operator mode. Run a HIP-vs-CPU comparison:

```bash
GGML_TEST_ATFLASH=1 ./build-feature_GCN_ATFLASH/bin/test-backend-ops test -b HIP -o FLASH_ATTN_EXT | tee test-atflash.log
```

The tests exercise one- and two-period windows, normal/NeoX RoPE layouts, and decode/prefill-sized query batches. They use logical KV positions that differ from physical KV-cache indices.

## Important limitations

The initial GPU implementation **skips selected QK multiply-accumulate operations but still loads full K tiles**. It is a correctness-oriented prototype, not the complete memory-traffic-avoiding implementation described in ATFlash. Speedups are not established and the additional branching may cause regressions. A later optimization should skip fully inactive K chunks before global-memory loads and benchmark dense-off overhead independently.

The feature assumes full GPU offload. Other GPU backends do not yet implement the modified attention semantics, so do not use this experiment with Vulkan, Metal or mixed backend offload. No branch of this experiment should be proposed upstream without local compilation, backend tests, accuracy measurements and a maintainer discussion.
