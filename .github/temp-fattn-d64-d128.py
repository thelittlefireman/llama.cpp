from pathlib import Path

fattn = Path('ggml/src/ggml-cuda/fattn.cu')
vec = Path('ggml/src/ggml-cuda/fattn-vec.cuh')
tests = Path('tests/test-backend-ops.cpp')

s = fattn.read_text()
old = '''#if defined(GGML_USE_HIP)
    const bool gcn_vec_tail_type = K->type == V->type &&
        (K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q4_1 || K->type == GGML_TYPE_Q5_1 || K->type == GGML_TYPE_Q8_0);
    const bool can_use_gcn_vec_tail = GGML_CUDA_CC_IS_GCN(cc) && Q->ne[0] == 256 && Q->ne[1] == 1 &&
        gcn_vec_tail_type && K->ne[1] >= FATTN_KQ_STRIDE;
#else
    constexpr bool can_use_gcn_vec_tail = false;
#endif // defined(GGML_USE_HIP)
'''
new = '''#if defined(GGML_USE_HIP)
    const bool can_use_gcn_vec_tail = GGML_CUDA_CC_IS_GCN(cc) && Q->ne[0] == 256 && Q->ne[1] == 1 &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && K->ne[1] >= FATTN_KQ_STRIDE;
#else
    constexpr bool can_use_gcn_vec_tail = false;
#endif // defined(GGML_USE_HIP)
'''
assert s.count(old) == 1
fattn.write_text(s.replace(old, new))

s = vec.read_text()
old = '''#if defined(GGML_USE_HIP)
    constexpr bool gcn_vec_tail_type = type_K == type_V &&
        (type_K == GGML_TYPE_Q4_0 || type_K == GGML_TYPE_Q4_1 || type_K == GGML_TYPE_Q5_1 || type_K == GGML_TYPE_Q8_0);
    if constexpr (D == 256 && gcn_vec_tail_type) {
'''
new = '''#if defined(GGML_USE_HIP)
    if constexpr (D == 256 && type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
'''
assert s.count(old) == 1
vec.write_text(s.replace(old, new))

s = tests.read_text()
marker = '''    for (const ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 20000, 1, true, false, 0, 0, GGML_PREC_F32, type, type));
    }
'''
assert s.count(marker) == 1
addition = '''
    // Characterize GCN wave64 vector-vs-tile behavior across D and quant type.
    for (int hs : {64, 128}) {
        for (const ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
            test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 8, {8, 1},  9984, 1, true, false, 0, 0, GGML_PREC_F32, type, type));
            test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 8, {8, 1}, 10000, 1, true, false, 0, 0, GGML_PREC_F32, type, type));
        }
    }
'''
tests.write_text(s.replace(marker, marker + addition))
