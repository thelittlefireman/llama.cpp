from pathlib import Path
import subprocess
import textwrap

base = "8d25e1f0602a467439c174541cb2ab38e4cea574"
vec_path = Path("ggml/src/ggml-cuda/fattn-vec.cuh")
cu_path = Path("ggml/src/ggml-cuda/fattn.cu")

current = vec_path.read_text()
loop_header = "    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,\n             // Increment pointers after each loop:\n             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {\n"
loop_start = current.index(loop_header)
full_line = "        const bool full_kv_tile = k_VKQ_0 + nthreads <= k_VKQ_max;\n"
full_start = current.index(full_line, loop_start)
body_start = full_start + len(full_line)
loop_end_marker = "\n    }\n\n    if (sinks && blockIdx.y == 0)"
loop_end = current.index(loop_end_marker, body_start)
tail_body = current[body_start:loop_end]
tail_body = tail_body.replace(
    "                const bool valid_KQ = !allow_kv_tail || full_kv_tile || k_VKQ_0 + i_KQ < k_VKQ_max;",
    "                const bool valid_KQ = k_VKQ_0 + i_KQ < k_VKQ_max;")
tail_body = tail_body.replace(
    "            if constexpr (allow_kv_tail) {\n                if (!full_kv_tile && k_VKQ_0 + k >= k_VKQ_max) {\n                    continue;\n                }\n            }\n\n",
    "            if (k_VKQ_0 + k >= k_VKQ_max) {\n                continue;\n            }\n\n")
assert "allow_kv_tail" not in tail_body
assert "full_kv_tile" not in tail_body

subprocess.run(["git", "checkout", base, "--", str(vec_path), str(cu_path)], check=True)
s = vec_path.read_text()

old = "template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size"
new = "template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool oob_check> // D == head size"
assert s.count(old) == 1
s = s.replace(old, new)

loop_start = s.index(loop_header)
insert_marker = "\n        // Calculate KQ tile and keep track of new maximum KQ values:"
insert_pos = s.index(insert_marker, loop_start)
nested_tail = textwrap.indent(tail_body.lstrip("\n"), "        ")
tail_block = (
    "\n        if constexpr (oob_check) {\n"
    "            if (k_VKQ_0 + nthreads > k_VKQ_max) {\n"
    + nested_tail +
    "\n                continue;\n"
    "            }\n"
    "        }\n"
)
s = s[:insert_pos] + tail_block + s[insert_pos:]

old = "template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>\nvoid ggml_cuda_flash_attn_ext_vec_case_impl"
new = "template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool oob_check>\nvoid ggml_cuda_flash_attn_ext_vec_case_impl"
assert s.count(old) == 1
s = s.replace(old, new)

old = "fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;"
new = "fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, oob_check>;"
assert s.count(old) == 1
s = s.replace(old, new)

old_call = "ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);"
assert s.count(old_call) == 4
s = s.replace(old_call, "ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap, false>(ctx, dst);")

marker = "    float logit_softcap;\n    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));\n\n    if (Q->ne[1] == 1) {"
assert s.count(marker) == 1
replacement = '''    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    if constexpr (D == 256 && type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
        const ggml_tensor * K = dst->src[1];
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (GGML_CUDA_CC_IS_GCN(cc) && Q->ne[1] == 1 && K->ne[1] % FATTN_KQ_STRIDE != 0) {
            constexpr int cols_per_block = 1;
            if (logit_softcap == 0.0f) {
                constexpr bool use_logit_softcap = false;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap, true>(ctx, dst);
            } else {
                constexpr bool use_logit_softcap = true;
                ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap, true>(ctx, dst);
            }
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

    if (Q->ne[1] == 1) {'''
s = s.replace(marker, replacement)
assert "allow_kv_tail" not in s
assert s.count("bool oob_check") == 2
assert s.count("use_logit_softcap, false>(ctx, dst)") == 4
assert s.count("use_logit_softcap, true>(ctx, dst)") == 2
vec_path.write_text(s)

s = cu_path.read_text()
old = "    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;"
new = '''#if defined(GGML_USE_HIP)
    const bool can_use_gcn_vec_tail = GGML_CUDA_CC_IS_GCN(cc) && Q->ne[0] == 256 && Q->ne[1] == 1 &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 && K->ne[1] >= FATTN_KQ_STRIDE;
#else
    constexpr bool can_use_gcn_vec_tail = false;
#endif // defined(GGML_USE_HIP)
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 &&
        (K->ne[1] % FATTN_KQ_STRIDE == 0 || can_use_gcn_vec_tail);'''
assert s.count(old) == 1
s = s.replace(old, new)
cu_path.write_text(s)
