from pathlib import Path

path = Path('ggml/src/ggml-cuda/fattn-tile.cuh')
text = path.read_text()

old = '''    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (lane_id == 0) {
            KQ_max_new_shared[warp_id] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(warp_id & ~(np-1)) + lane_id % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }
'''
new = '''#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (DKQ == 256 && DV == 256 && np > 1 && np % 2 == 0) {
        static_assert(cpw == 1, "bad cpw");
        KQ_max_new[0] = fmaxf(KQ_max_new[0], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[0], 32, 64));
        if constexpr (np > 2) {
            __shared__ float KQ_max_new_shared[nwarps/2];
            if ((warp_id & 1) == 0 && lane_id == 0) {
                KQ_max_new_shared[warp_id/2] = KQ_max_new[0];
            }
            __syncthreads();
            KQ_max_new[0] = KQ_max_new_shared[(warp_id & ~(np-1))/2 + lane_id % (np/2)];
            KQ_max_new[0] = warp_reduce_max<np/2>(KQ_max_new[0]);
        } else {
            __syncthreads();
        }
    } else
#endif // defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (lane_id == 0) {
            KQ_max_new_shared[warp_id] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(warp_id & ~(np-1)) + lane_id % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }
'''
assert text.count(old) == 1
text = text.replace(old, new)

start = text.index('    if constexpr (np > 1) {\n        static_assert(cpw == 1, "bad cpw");\n        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");', text.index('KQ_sum[jc0] = warp_reduce_sum<warp_size>'))
end = text.index('\n\n    // Attention sink:', start)
old = text[start:end]
new = '''    if constexpr (np > 1) {
        static_assert(cpw == 1, "bad cpw");
        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, "KV_tmp too small");

#if defined(GGML_USE_HIP) && defined(GCN)
        if constexpr (DKQ == 256 && DV == 256 && np % 2 == 0) {
            KQ_sum[0] += __shfl_xor_sync(0xFFFFFFFF, KQ_sum[0], 32, 64);
#ifdef FAST_FP16_AVAILABLE
#pragma unroll
            for (int i = 0; i < cpw*((DVp/2)/warp_size); ++i) {
                const uint32_t other_u32 = __shfl_xor_sync(0xFFFFFFFF, __builtin_bit_cast(uint32_t, VKQ[i]), 32, 64);
                VKQ[i] += __builtin_bit_cast(half2, other_u32);
            }
#else
#pragma unroll
            for (int i = 0; i < cpw*((DVp/2)/warp_size); ++i) {
                VKQ[i].x += __shfl_xor_sync(0xFFFFFFFF, VKQ[i].x, 32, 64);
                VKQ[i].y += __shfl_xor_sync(0xFFFFFFFF, VKQ[i].y, 32, 64);
            }
#endif // FAST_FP16_AVAILABLE

            if (warp_id & 1) {
                return;
            }

            if constexpr (np > 2) {
#ifdef FAST_FP16_AVAILABLE
                half2 * VKQ_combine = (half2 *) KV_tmp;
#else
                float * VKQ_combine = (float *) KV_tmp;
#endif // FAST_FP16_AVAILABLE
                float * KQ_sum_combine = (float *) Q_tmp;

                if (warp_id % np != 0) {
#ifdef FAST_FP16_AVAILABLE
                    constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
                    for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                        ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[(warp_id/2)*(DVp/2) + i0 + lane_id*cpy_ne_D], &VKQ[i0/warp_size]);
                    }
#else
                    constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
                    for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                        ggml_cuda_memcpy_1<cpy_ne_D*4>(
                            &VKQ_combine[(warp_id/2)*DVp + i0 + lane_id*cpy_ne_D], ((const float *) VKQ) + i0/warp_size);
                    }
#endif // FAST_FP16_AVAILABLE

                    if (lane_id == 0) {
                        KQ_sum_combine[warp_id/2] = KQ_sum[0];
                    }
                    return;
                }

                __syncthreads();

#pragma unroll
                for (int ip = 2; ip < np; ip += 2) {
#ifdef FAST_FP16_AVAILABLE
                    constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
                    for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                        __align__(16) half2 tmp[cpy_ne_D];
                        ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[((warp_id + ip)/2)*(DVp/2) + i0 + lane_id*cpy_ne_D]);
#pragma unroll
                        for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                            VKQ[i0/warp_size + i1] += tmp[i1];
                        }
                    }
#else
                    constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
                    for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                        __align__(16) float tmp[cpy_ne_D];
                        ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[((warp_id + ip)/2)*DVp + i0 + lane_id*cpy_ne_D]);
#pragma unroll
                        for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                            ((float *)VKQ)[i0/warp_size + i1] += tmp[i1];
                        }
                    }
#endif // FAST_FP16_AVAILABLE
                    KQ_sum[0] += KQ_sum_combine[(warp_id + ip)/2];
                }
            }
        } else
#endif // defined(GGML_USE_HIP) && defined(GCN)
        {
#ifdef FAST_FP16_AVAILABLE
            half2 * VKQ_combine    = (half2 *) KV_tmp;
#else
            float * VKQ_combine    = (float *) KV_tmp;
#endif // FAST_FP16_AVAILABLE
            float * KQ_sum_combine = (float *) Q_tmp;

            if (warp_id % np != 0) {
#ifdef FAST_FP16_AVAILABLE
                constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(&VKQ_combine[warp_id*(DVp/2) + i0 + lane_id*cpy_ne_D], &VKQ[i0/warp_size]);
                }
#else
                constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(
                        &VKQ_combine[warp_id*DVp + i0 + lane_id*cpy_ne_D], ((const float *) VKQ) + i0/warp_size);
                }
#endif // FAST_FP16_AVAILABLE

                if (lane_id == 0) {
                    KQ_sum_combine[warp_id] = KQ_sum[0];
                }
                return;
            }

            __syncthreads();

#pragma unroll
            for (int ip = 1; ip < np; ++ip) {
#ifdef FAST_FP16_AVAILABLE
                constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                    __align__(16) half2 tmp[cpy_ne_D];
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(warp_id + ip)*(DVp/2) + i0 + lane_id*cpy_ne_D]);
#pragma unroll
                    for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                        VKQ[i0/warp_size + i1] += tmp[i1];
                    }
                }
#else
                constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                    __align__(16) float tmp[cpy_ne_D];
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(tmp, &VKQ_combine[(warp_id + ip)*DVp + i0 + lane_id*cpy_ne_D]);
#pragma unroll
                    for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                        ((float *)VKQ)[i0/warp_size + i1] += tmp[i1];
                    }
                }
#endif // FAST_FP16_AVAILABLE
                KQ_sum[0] += KQ_sum_combine[warp_id + ip];
            }
        }
    }'''
text = text[:start] + new + text[end:]

path.write_text(text)
Path('.github/workflows/temp-fattn-tile-wave64-np.yml').unlink()
Path('.github/scripts/temp-fattn-tile-wave64-np.py').unlink()
