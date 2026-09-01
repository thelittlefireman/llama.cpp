from pathlib import Path

path = Path('ggml/src/ggml-cuda/fattn-tile.cuh')
text = path.read_text()

old = '''template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, typename T_vec_dot>
'''
new = '''template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, typename T_vec_dot>
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup, lane_id, warp_id);
'''
new = '''    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int load_warp_size = ggml_cuda_fattn_tile_get_physical_warp_size_device(DKQ, DV);
    constexpr int load_nwarps    = nwarps*warp_size/load_warp_size;
    static_assert((nwarps*warp_size) % load_warp_size == 0, "bad load warp size");

    flash_attn_tile_load_tile<load_warp_size, load_nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup, threadIdx.x, threadIdx.y);
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = 'flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, '
new = 'flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, DV, '
assert text.count(old) == 2
text = text.replace(old, new)

old = '''    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.
'''
new = '''    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int load_warp_size = ggml_cuda_fattn_tile_get_physical_warp_size_device(DKQ, DV);
    constexpr int load_nwarps    = nwarps*warp_size/load_warp_size;
    static_assert((nwarps*warp_size) % load_warp_size == 0, "bad load warp size");

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''        flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0, lane_id, warp_id);
'''
new = '''        flash_attn_tile_load_tile<load_warp_size, load_nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0, threadIdx.x, threadIdx.y);
'''
assert text.count(old) == 1
text = text.replace(old, new)

path.write_text(text)
Path('.github/workflows/temp-fattn-tile-wave64-load.yml').unlink()
Path('.github/scripts/temp-fattn-tile-wave64-load.py').unlink()
