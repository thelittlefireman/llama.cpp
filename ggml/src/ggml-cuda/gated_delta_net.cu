#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // Apply the scalar decay once. kv uses the decayed state and the
            // same values are then reused for the rank-1 correction below.
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] *= g_val;
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - (g*S)^T k[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] += k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] += k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // Apply the per-row decay once. kv uses the decayed state and the
            // same values are then reused for the rank-1 correction below.
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r] *= expf(g_t[i]);
                kv_shard += s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] += k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] += k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}


template <int S_v>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_replay_cuda(const float * q,
                            const float * k,
                            const float * v,
                            const float * g,
                            const float * beta,
                            const float * checkpoint,
                            const float * replay,
                            const int32_t * state_copy,
                            float * replay_all,
                            float * dst,
                            float * checkpoint_out,
                            int64_t H,
                            int64_t n_tokens,
                            int64_t n_seqs,
                            int64_t sq1,
                            int64_t sq2,
                            int64_t sq3,
                            int64_t sv1,
                            int64_t sv2,
                            int64_t sv3,
                            int64_t sb1,
                            int64_t sb2,
                            int64_t sb3,
                            const uint3 neqk1_magic,
                            const uint3 rq3_magic,
                            float scale,
                            int replay_buffer_size,
                            int mem_size,
                            int state_head,
                            int64_t replay_seq_stride,
                            int64_t replay_all_seq_stride,
                            int K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int lane = threadIdx.x;
    const int col  = blockIdx.z * blockDim.y + threadIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int rows_per_lane = S_v / warp_size;

    const int physical_size = 2 * replay_buffer_size;
    const int physical_mask = physical_size - 1;
    const int record_stride = 2 * S_v + 1;
    const int head_stride   = 2 + physical_size * record_stride;

    const int encoded_state = state_copy[sequence];
    const int rollback      = encoded_state / mem_size;
    const int src_cell      = encoded_state % mem_size;
    const int dst_cell      = state_head + sequence;
    const bool moved        = src_cell != dst_cell;

    const float * replay_src = replay + sequence * replay_seq_stride + h_idx * head_stride;
    float * replay_dst       = replay_all + dst_cell * replay_all_seq_stride + h_idx * head_stride;

    int base  = (int) replay_src[0] & physical_mask;
    int count = (int) replay_src[1];
    count = rollback < count ? count - rollback : 0;

    const int rollback_window = K > 1 ? K - 1 : 0;
    const int total_count     = count + (int) n_tokens;
    int n_flush = 0;
    if (total_count > replay_buffer_size) {
        n_flush = count - rollback_window;
        if (n_flush < total_count - replay_buffer_size) {
            n_flush = total_count - replay_buffer_size;
        }
        n_flush = n_flush > 0 ? n_flush : 0;
    }

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * checkpoint_head = checkpoint + (sequence * H + h_idx) * S_v * S_v;
    float * checkpoint_dst = checkpoint_out + (sequence * H + h_idx) * S_v * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        s_shard[r] = checkpoint_head[col * S_v + i];
    }

    if (moved && n_flush == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            checkpoint_dst[col * S_v + i] = s_shard[r];
        }
    }

    if (moved) {
        for (int j = 0; j < count; ++j) {
            const int slot = (base + j) & physical_mask;
            const float * src_record = replay_src + 2 + slot * record_stride;
            float * dst_record = replay_dst + 2 + slot * record_stride;

            if (lane == 0) {
                dst_record[1 + S_v + col] = src_record[1 + S_v + col];
            }
            if (blockIdx.z == 0 && threadIdx.y == 0) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int i = r * warp_size + lane;
                    dst_record[1 + i] = src_record[1 + i];
                }
                if (lane == 0) {
                    dst_record[0] = src_record[0];
                }
            }
        }
    }

    for (int j = 0; j < count; ++j) {
        const int slot = (base + j) & physical_mask;
        const float * record = replay_src + 2 + slot * record_stride;
        const float decay = expf(record[0]);
        const float delta_col = record[1 + S_v + col];

#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            s_shard[r] = decay * s_shard[r] + record[1 + i] * delta_col;
        }

        if (j + 1 == n_flush) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r * warp_size + lane;
                checkpoint_dst[col * S_v + i] = s_shard[r];
            }
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float beta_val = beta[gb_offset];
        const float g_val    = g[gb_offset];
        const float decay    = expf(g_val);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
            s_shard[r] *= decay;
            kv_shard += s_shard[r] * k_reg[r];
        }

        const float kv_col = warp_reduce_sum<warp_size>(kv_shard);
        const float delta_col = (v_t[col] - kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            s_shard[r] += k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }

        const float attn_col = warp_reduce_sum<warp_size>(attn_partial);
        if (lane == 0) {
            attn_data[col] = attn_col * scale;
        }
        attn_data += S_v * H;

        const int slot = (base + count + t) & physical_mask;
        float * record = replay_dst + 2 + slot * record_stride;
        if (lane == 0) {
            record[1 + S_v + col] = delta_col;
        }
        if (blockIdx.z == 0 && threadIdx.y == 0) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r * warp_size + lane;
                record[1 + i] = k_reg[r];
            }
            if (lane == 0) {
                record[0] = g_val;
            }
        }
    }

    if (blockIdx.z == 0 && threadIdx.y == 0 && lane == 0) {
        replay_dst[0] = (float) ((base + n_flush) & physical_mask);
        replay_dst[1] = (float) (total_count - n_flush);
    }
}

static void launch_gated_delta_net_replay(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * checkpoint_d,
        const float * replay_d, const int32_t * state_copy_d, float * replay_all_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale,
        int replay_buffer_size, int mem_size, int state_head,
        int64_t replay_seq_stride, int64_t replay_all_seq_stride, int K,
        cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);

#define GGML_CUDA_GDN_REPLAY_CASE(S) \
    case S: ggml_cuda_kernel_launch(gated_delta_net_replay_cuda<S>, launch_params, \
        q_d, k_d, v_d, g_d, b_d, checkpoint_d, replay_d, state_copy_d, replay_all_d, dst_d, state_d, H, \
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, \
        replay_buffer_size, mem_size, state_head, replay_seq_stride, replay_all_seq_stride, K); break

    switch (S_v) {
        GGML_CUDA_GDN_REPLAY_CASE(16);
        GGML_CUDA_GDN_REPLAY_CASE(32);
        GGML_CUDA_GDN_REPLAY_CASE(64);
        GGML_CUDA_GDN_REPLAY_CASE(128);
        default: GGML_ABORT("fatal error");
    }
#undef GGML_CUDA_GDN_REPLAY_CASE
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    const bool use_replay = cache != nullptr && !kda && dst->src[6] != nullptr;
    if (use_replay) {
        ggml_tensor * src_replay          = dst->src[6];
        ggml_tensor * src_state_copy      = dst->src[7];
        ggml_tensor * src_checkpoint      = dst->src[8];
        ggml_tensor * src_replay_all      = dst->src[9];
        const int replay_buffer_size      = ggml_get_op_params_i32(dst, 1);
        const int mem_size                = ggml_get_op_params_i32(dst, 2);
        const int state_head              = ggml_get_op_params_i32(dst, 3);
        const int64_t replay_seq_stride   = src_replay->nb[1] / sizeof(float);
        const int64_t replay_all_stride   = src_replay_all->nb[1] / sizeof(float);

        GGML_ASSERT(src_replay->type == GGML_TYPE_F32 && src_replay_all->type == GGML_TYPE_F32);
        GGML_ASSERT(src_state_copy->type == GGML_TYPE_I32 && src_checkpoint->type == GGML_TYPE_F32);
        launch_gated_delta_net_replay(q_d, k_d, v_d, g_d, b_d,
            (const float *) src_checkpoint->data, (const float *) src_replay->data,
            (const int32_t *) src_state_copy->data, (float *) src_replay_all->data,
            dst_d, cache->data, S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, replay_buffer_size, mem_size, state_head,
            replay_seq_stride, replay_all_stride, K, stream);
        return;
    }

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
