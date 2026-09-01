from pathlib import Path
import re

path = Path('ggml/src/ggml-cuda/fattn-tile.cuh')
text = path.read_text()

old = '''    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.
'''
new = '''    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr bool use_wave64_vkq = DKQ == 256 && DV == 256 && np > 1 && np % 2 == 0;
#else
    constexpr bool use_wave64_vkq = false;
#endif // defined(GGML_USE_HIP) && defined(GCN)
    constexpr int vkq_warp_size = use_wave64_vkq ? 64 : warp_size;

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
'''
new = '''    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr bool use_wave64_vkq = DKQ == 256 && DV == 256 && np > 1 && np % 2 == 0;
#else
    constexpr bool use_wave64_vkq = false;
#endif // defined(GGML_USE_HIP) && defined(GCN)
    constexpr int vkq_warp_size = use_wave64_vkq ? 64 : warp_size;
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
#else
    __shared__ float Q_tmp[ncols * DKQ];
    __shared__ float KV_tmp[nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV];
    __shared__ float KQ[ncols * nbatch_fa];
    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};
'''
new = '''    __align__(16) half2 VKQ[cpw * ((DVp/2)/vkq_warp_size)] = {{0.0f, 0.0f}};
#else
    __shared__ float Q_tmp[ncols * DKQ];
    __shared__ float KV_tmp[nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV];
    __shared__ float KQ[ncols * nbatch_fa];
    __align__(16) float2 VKQ[cpw * ((DVp/2)/vkq_warp_size)] = {{0.0f, 0.0f}};
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
'''
new = '''#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < (DVp/2)/vkq_warp_size; ++i0) {
                VKQ[jc*((DVp/2)/vkq_warp_size) + i0] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < (DVp/2)/vkq_warp_size; ++i0) {
                VKQ[jc*((DVp/2)/vkq_warp_size) + i0].x *= KQ_max_scale;
                VKQ[jc*((DVp/2)/vkq_warp_size) + i0].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
'''
assert text.count(old) == 2
text = text.replace(old, new)

start = text.index('''#ifdef FAST_FP16_AVAILABLE
#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {''', text.index('// VKQ = V @ KQ matrix multiplication:'))
end_marker = '''#endif // FAST_FP16_AVAILABLE

        __syncthreads();'''
end = text.index(end_marker, start) + len('#endif // FAST_FP16_AVAILABLE')
old = text[start:end]
new = '''#ifdef FAST_FP16_AVAILABLE
        if constexpr (use_wave64_vkq) {
            constexpr int vkq_ne = (DVp/2)/vkq_warp_size;
            constexpr int cpy_ne_D = cpy_ne/2 < vkq_ne ? cpy_ne/2 : vkq_ne;
            const int physical_lane_id = threadIdx.x;
            const int pair = (warp_id % np) & ~1;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_V; k1 += np) {
#pragma unroll
                for (int ip = 0; ip < 2; ++ip) {
                    __align__(16) half2 V_k[vkq_ne];
                    __align__(16) half2 KQ_k[cpw];
                    const int row = k1 + pair + ip;
#pragma unroll
                    for (int i0 = 0; i0 < vkq_ne; i0 += cpy_ne_D) {
                        ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0], &KV_tmp[row*(DV/2) + physical_lane_id*vkq_ne + i0]);
                    }
#pragma unroll
                    for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                        const int jc_KQ = jc_VKQ_0/KQ_cs + (warp_id / np)*(cpw/KQ_cs);
                        __align__(16) half tmp[KQ_cs];
                        ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
                            &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + row)*KQ_cs);
#pragma unroll
                        for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                            KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
                        }
                    }
#pragma unroll
                    for (int i0 = 0; i0 < vkq_ne; ++i0) {
#pragma unroll
                        for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                            VKQ[jc_VKQ_0*vkq_ne + i0] += V_k[i0]*KQ_k[jc_VKQ_0];
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int k1 = 0; k1 < nbatch_V; k1 += np) {
                __align__(16) half2 V_k[(DVp/2)/warp_size];
                __align__(16) half2 KQ_k[cpw];

                constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/warp_size], &KV_tmp[(k1 + warp_id % np)*(DV/2) + i0 + lane_id*cpy_ne_D]);
                }
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                    const int jc_KQ = jc_VKQ_0/KQ_cs + (warp_id / np)*(cpw/KQ_cs);

                    __align__(16) half tmp[KQ_cs];
                    ggml_cuda_memcpy_1<KQ_cs*sizeof(half)>(
                        &tmp, KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + warp_id % np)*KQ_cs);
#pragma unroll
                    for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_cs; ++jc_VKQ_1) {
                        KQ_k[jc_VKQ_0+jc_VKQ_1] = __half2half2(tmp[jc_VKQ_1]);
                    }
                }

#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                    for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                        VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];
                    }
                }
            }
        }
#else
        if constexpr (use_wave64_vkq) {
            constexpr int vkq_ne = (DVp/2)/vkq_warp_size;
            constexpr int cpy_ne_D = cpy_ne/2 < vkq_ne ? cpy_ne/2 : vkq_ne;
            const int physical_lane_id = threadIdx.x;
            const int pair = (warp_id % np) & ~1;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_V; k1 += np) {
#pragma unroll
                for (int ip = 0; ip < 2; ++ip) {
                    __align__(16) float2 V_k[vkq_ne];
                    __align__(16) float KQ_k[cpw];
                    const int row = k1 + pair + ip;
#pragma unroll
                    for (int i0 = 0; i0 < vkq_ne; i0 += cpy_ne_D) {
                        ggml_cuda_memcpy_1<cpy_ne_D*sizeof(float2)>(
                            &V_k[i0], &KV_tmp[row*DV + 2*(physical_lane_id*vkq_ne + i0)]);
                    }
#pragma unroll
                    for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                        const int jc_KQ = jc_VKQ_0/KQ_cs + (warp_id / np)*(cpw/KQ_cs);
                        ggml_cuda_memcpy_1<KQ_cs*sizeof(float)>(
                            &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + row)*KQ_cs);
                    }
#pragma unroll
                    for (int i0 = 0; i0 < vkq_ne; ++i0) {
#pragma unroll
                        for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                            VKQ[jc_VKQ_0*vkq_ne + i0].x += V_k[i0].x*KQ_k[jc_VKQ_0];
                            VKQ[jc_VKQ_0*vkq_ne + i0].y += V_k[i0].y*KQ_k[jc_VKQ_0];
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int k1 = 0; k1 < nbatch_V; k1 += np) {
                __align__(16) float2 V_k[(DVp/2)/warp_size];
                __align__(16) float  KQ_k[cpw];

                constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + warp_id % np)*DV + i0 + lane_id*cpy_ne_D]);
                }
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                    const int jc_KQ = jc_VKQ_0/KQ_cs + (warp_id / np)*(cpw/KQ_cs);

                    ggml_cuda_memcpy_1<KQ_cs*sizeof(float)>(
                        &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + warp_id % np)*KQ_cs);
                }

#pragma unroll
                for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                    for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                        VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[jc_VKQ_0];
                        VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[jc_VKQ_0];
                    }
                }
            }
        }
#endif // FAST_FP16_AVAILABLE'''
text = text[:start] + new + text[end:]

pattern = re.compile(r'''#if defined\(GGML_USE_HIP\) && defined\(GCN\)\n        if constexpr \(DKQ == 256 && DV == 256 && np % 2 == 0\) \{.*?        \} else\n#endif // defined\(GGML_USE_HIP\) && defined\(GCN\)''', re.S)
matches = list(pattern.finditer(text))
assert len(matches) == 1
new = '''#if defined(GGML_USE_HIP) && defined(GCN)
        if constexpr (use_wave64_vkq) {
            constexpr int physical_np = np/2;
            constexpr int vkq_ne = (DVp/2)/vkq_warp_size;
            const int physical_warp_id = warp_id/2;
            KQ_sum[0] += __shfl_xor_sync(0xFFFFFFFF, KQ_sum[0], 32, 64);

            if constexpr (physical_np > 1) {
#ifdef FAST_FP16_AVAILABLE
                half2 * VKQ_combine = (half2 *) KV_tmp;
#else
                float * VKQ_combine = (float *) KV_tmp;
#endif // FAST_FP16_AVAILABLE
                float * KQ_sum_combine = (float *) Q_tmp;
                const int physical_warp_pos = physical_warp_id % physical_np;

                if (physical_warp_pos != 0) {
#ifdef FAST_FP16_AVAILABLE
                    ggml_cuda_memcpy_1<vkq_ne*sizeof(half2)>(
                        &VKQ_combine[physical_warp_id*(DVp/2) + threadIdx.x*vkq_ne], VKQ);
#else
                    ggml_cuda_memcpy_1<vkq_ne*sizeof(float2)>(
                        &VKQ_combine[physical_warp_id*DVp + 2*threadIdx.x*vkq_ne], VKQ);
#endif // FAST_FP16_AVAILABLE
                    if (threadIdx.x == 0) {
                        KQ_sum_combine[physical_warp_id] = KQ_sum[0];
                    }
                }

                __syncthreads();
                if (physical_warp_pos != 0) {
                    return;
                }

#pragma unroll
                for (int ip = 1; ip < physical_np; ++ip) {
#ifdef FAST_FP16_AVAILABLE
                    __align__(16) half2 tmp[vkq_ne];
                    ggml_cuda_memcpy_1<vkq_ne*sizeof(half2)>(
                        tmp, &VKQ_combine[(physical_warp_id + ip)*(DVp/2) + threadIdx.x*vkq_ne]);
#pragma unroll
                    for (int i = 0; i < vkq_ne; ++i) {
                        VKQ[i] += tmp[i];
                    }
#else
                    __align__(16) float2 tmp[vkq_ne];
                    ggml_cuda_memcpy_1<vkq_ne*sizeof(float2)>(
                        tmp, &VKQ_combine[(physical_warp_id + ip)*DVp + 2*threadIdx.x*vkq_ne]);
#pragma unroll
                    for (int i = 0; i < vkq_ne; ++i) {
                        VKQ[i].x += tmp[i].x;
                        VKQ[i].y += tmp[i].y;
                    }
#endif // FAST_FP16_AVAILABLE
                    KQ_sum[0] += KQ_sum_combine[physical_warp_id + ip];
                }
            }
        } else
#endif // defined(GGML_USE_HIP) && defined(GCN)'''
text = pattern.sub(new, text, count=1)

old = '''#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + lane_id*cpy_ne_D < DV/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + lane_id*(2*cpy_ne_D)], tmp);
            }
        }
#else
        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + lane_id*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + lane_id*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && lane_id == 0) {
'''
new = '''#ifdef FAST_FP16_AVAILABLE
        if constexpr (use_wave64_vkq) {
            constexpr int vkq_ne = (DVp/2)/vkq_warp_size;
            __align__(16) float2 tmp[vkq_ne];
#pragma unroll
            for (int i = 0; i < vkq_ne; ++i) {
                tmp[i] = __half22float2(VKQ[jc0*vkq_ne + i]);
                tmp[i].x *= scale;
                tmp[i].y *= scale;
            }
            ggml_cuda_memcpy_1<sizeof(tmp)>(
                &dst[j_dst_unrolled*DV + 2*threadIdx.x*vkq_ne], tmp);
        } else {
            constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
                __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                    tmp[i1].x *= scale;
                    tmp[i1].y *= scale;
                }
                if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + lane_id*cpy_ne_D < DV/2) {
                    ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + lane_id*(2*cpy_ne_D)], tmp);
                }
            }
        }
#else
        if constexpr (use_wave64_vkq) {
            constexpr int vkq_ne = (DVp/2)/vkq_warp_size;
            __align__(16) float2 tmp[vkq_ne];
#pragma unroll
            for (int i = 0; i < vkq_ne; ++i) {
                tmp[i] = VKQ[jc0*vkq_ne + i];
                tmp[i].x *= scale;
                tmp[i].y *= scale;
            }
            ggml_cuda_memcpy_1<sizeof(tmp)>(
                &dst[j_dst_unrolled*DV + 2*threadIdx.x*vkq_ne], tmp);
        } else {
            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                if (i0 + warp_size*cpy_ne_D <= DV || i0 + lane_id*cpy_ne_D < DV) {
#pragma unroll
                    for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                        VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                        VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                    }
                    ggml_cuda_memcpy_1<cpy_ne_D*4>(
                        &dst[j_dst_unrolled*DV + i0 + lane_id*cpy_ne_D],
                        &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
                }
            }
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && (use_wave64_vkq ? threadIdx.x == 0 : lane_id == 0)) {
'''
assert text.count(old) == 1
text = text.replace(old, new)

path.write_text(text)
Path('.github/workflows/temp-fattn-tile-wave64-vkq.yml').unlink()
Path('.github/scripts/temp-fattn-tile-wave64-vkq.py').unlink()
