from pathlib import Path

vec = Path('ggml/src/ggml-cuda/fattn-vec.cuh')
common = Path('ggml/src/ggml-cuda/fattn-common.cuh')
tests = Path('tests/test-backend-ops.cpp')

s = vec.read_text()
old = '''#if defined(GGML_USE_HIP) && defined(GCN)
                    if constexpr (nthreads_quantize <= WARP_SIZE) {
                        if (threadIdx.x < WARP_SIZE) {
                            quantize_q8_1_to_shared<float2, nthreads_quantize>
                                (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                        }
                    } else {
                        quantize_q8_1_to_shared<float2, nthreads_quantize>
                            (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                    }
#else
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
#endif // defined(GGML_USE_HIP) && defined(GCN)
'''
new = '''#if defined(GGML_USE_HIP) && defined(GCN)
                    const bool quantize_lane_active = nthreads_quantize > WARP_SIZE || threadIdx.x < WARP_SIZE;
#else
                    constexpr bool quantize_lane_active = true;
#endif // defined(GGML_USE_HIP) && defined(GCN)
                    if (quantize_lane_active) {
                        quantize_q8_1_to_shared<float2, nthreads_quantize>
                            (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                    }
'''
assert s.count(old) == 1
s = s.replace(old, new)

marker = '''        if constexpr (oob_check) {
            if (k_VKQ_0 + nthreads > k_VKQ_max) {
'''
replacement = '''        // Keep bounds checks out of the full-tile hot path. They cause a large regression on GCN.
        if constexpr (oob_check) {
            if (k_VKQ_0 + nthreads > k_VKQ_max) {
'''
assert s.count(marker) == 1
s = s.replace(marker, replacement)
vec.write_text(s)

s = common.read_text()
debug = '''#if defined(GGML_USE_HIP)
    if constexpr (DV == 256) {
        if (Q->ne[1] == 1 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0) {
            fprintf(stderr,
                "FATTN_DEBUG D=%d kv=%lld nb=%lld block=[%u,%u] max_blocks_per_sm=%d ntiles_KV=%d parallel_blocks=%d grid=[%u,%u,%u]\\n",
                DV, (long long) K->ne[1], (long long) Q->ne[1], block_dim.x, block_dim.y,
                max_blocks_per_sm, ntiles_KV, parallel_blocks, blocks_num.x, blocks_num.y, blocks_num.z);
        }
    }
#endif // defined(GGML_USE_HIP)

'''
assert s.count(debug) == 1
s = s.replace(debug, '')
if 'fprintf(' not in s:
    s = s.replace('#include <cstdio>\n', '')
common.write_text(s)

s = tests.read_text()
marker = '    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 20000, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));\n'
assert s.count(marker) == 1
addition = '''
    // GCN wave64 fattn-vec tail coverage around a 10k context.
    for (int kv : {9984, 9985, 10016, 10048, 10080, 10111, 10112}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, kv, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    }

    // Compare aligned vec and unaligned tile paths before enabling KV tails for more quant types.
    for (const ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},  9984, 1, true, false, 0, 0, GGML_PREC_F32, type, type));
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 10000, 1, true, false, 0, 0, GGML_PREC_F32, type, type));
    }
'''
s = s.replace(marker, marker + addition)
tests.write_text(s)
