#include "common.cuh"

template<int block_size>
static __device__ __forceinline__ float block_reduce_sum_lane0_ct(float val, float * shared_vals) {
    static_assert(block_size == 32 || block_size == 128 || block_size == 512);

    val = warp_reduce_sum(val);

    if constexpr (block_size == WARP_SIZE) {
        return val;
    }

    constexpr int nwarps = block_size / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        shared_vals[warp_id] = val;
    }

    __syncthreads();

    if (warp_id != 0) {
        return val;
    }

    val = lane_id < nwarps ? shared_vals[lane_id] : 0.0f;

    if constexpr (nwarps == 4) {
        return warp_reduce_sum<4>(val);
    } else if constexpr (nwarps == 16) {
        return warp_reduce_sum<16>(val);
    } else {
        return warp_reduce_sum(val);
    }
}

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template <bool norm, int block_size>
static __global__ void reduce_rows_f32(const float * x_ptr, float * dst_ptr, const int ncols) {
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    ggml_cuda_pdl_sync();
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = x[row * ncols + i];
            } else {
                temp[j] = 0.0f;
            }
            i += block_size;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    // Sum up partial sums. Only thread 0 consumes the final block result.
    __shared__ float shared_vals[32];
    sum = block_reduce_sum_lane0_ct<block_size>(sum, shared_vals);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}
