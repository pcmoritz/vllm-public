
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 32
#define GROUP_SIZE_M 8

#define OFFSET_TOKEN_ID(i) (pid_m * BLOCK_SIZE_M + (i))
#define OFFS_BN(i) ((pid_n * BLOCK_SIZE_N + i) % N)

namespace vllm {

template<typename scalar_t>
__global__ void fused_moe_kernel(
    scalar_t *a,
    scalar_t *b,
    scalar_t *c,
    int32_t *sorted_token_ids,
    int32_t *expert_ids,
    int32_t *total_tokens_post_pad,
    const int M,
    const int N,
    const int K,
    const int EM,
    const int num_valid_tokens,
    // The stride variables represent how much to increase the ptr by when moving by 1
    // element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    // by to get the element one row down (A has M rows).
    const int stride_am,
    const int stride_ak,
    const int stride_be,
    const int stride_bk,
    const int stride_bn,
    const int stride_cm,
    const int stride_cn,
    const int stride_weight,
    const int stride_token_id
) {
    // Calculate the global thread ID
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the number of PIDs in the M and N dimensions
    int num_pid_m = (EM + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    // Compute the number of PIDs in a group
    int num_pid_in_group = GROUP_SIZE_M * num_pid_n;

    // Calculate the group ID and the first PID in M dimension for this group
    int group_id = pid / num_pid_in_group;
    int first_pid_m = group_id * GROUP_SIZE_M;

    // Calculate the size of the group in M dimension
    int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);

    // Calculate the PID in the M and N dimensions
    int pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
    int pid_n = (pid % num_pid_in_group) / group_size_m;

    // ----------------------------------------------------------
    // Create pointers for the first blocks of A and B.
    // We will advance this pointer as we move in the K direction
    // and accumulate
    // `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    // `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers

       // Load number of tokens post padded
    int num_tokens_post_padded = *total_tokens_post_pad;
    if (pid_m * BLOCK_SIZE_M >= num_tokens_post_padded) {
        return;
    }

    // Load sorted token ids and create token mask
    bool token_mask[BLOCK_SIZE_M];
    for (int i = 0; i < BLOCK_SIZE_M; ++i) {
        int offs_token = sorted_token_ids_ptr[OFFSET_TOKEN_ID(i)];
        token_mask[i] = offs_token < num_valid_tokens;
    }

    // Calculate pointers for A and B matrices
    float* a_ptrs[BLOCK_SIZE_M][BLOCK_SIZE_K];
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            int offs_token = sorted_token_ids_ptr[OFFSET_TOKEN_ID(m)];
            a_ptrs[m][k] = a_ptr + (offs_token / top_k * stride_am) + (k * stride_ak);
        }
    }

    int off_experts = expert_ids_ptr[pid_m] * stride_be;
    float* b_ptrs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    for (int k = 0; k < BLOCK_SIZE_K; ++k) {
        for (int n = 0; n < BLOCK_SIZE_N; ++n) {
            b_ptrs[k][n] = b_ptr + off_experts + (k * stride_bk) + (OFFS_BN(n) * stride_bn);
        }
    }

    // -----------------------------------------------------------
    // Iterate to compute a block of the C matrix.
    // We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    // of fp32 values for higher accuracy.
    // `accumulator` will be converted back to fp16 after the loop.

    // Initialize the accumulator
    float accumulator[BLOCK_SIZE_M][BLOCK_SIZE_N] = {0};

    // Loop over K dimension in blocks
    for (int k = 0; k < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++k) {
        float a[BLOCK_SIZE_M][BLOCK_SIZE_K] = {0};
        float b[BLOCK_SIZE_K][BLOCK_SIZE_N] = {0};

        // Load the next block of A and B
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
            for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
                if (token_mask[m] && kk < K - k * BLOCK_SIZE_K) {
                    a[m][kk] = *(a_ptrs[m][kk]);
                }
            }
        }
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
            for (int n = 0; n < BLOCK_SIZE_N; ++n) {
                if (kk < K - k * BLOCK_SIZE_K) {
                    b[kk][n] = *(b_ptrs[kk][n]);
                }
            }
        }

        // Dot product and accumulate
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
            for (int n = 0; n < BLOCK_SIZE_N; ++n) {
                float sum = 0;
                for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
                    sum += a[m][kk] * b[kk][n];
                }
                accumulator[m][n] += sum;
            }
        }

        // Advance the pointers to the next K block
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
            for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
                a_ptrs[m][kk] += BLOCK_SIZE_K * stride_ak;
            }
        }
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
            for (int n = 0; n < BLOCK_SIZE_N; ++n) {
                b_ptrs[kk][n] += BLOCK_SIZE_K * stride_bk;
            }
        }
    }

    // Conditional loading of weights and multiplication
    if (MUL_ROUTED_WEIGHT) {
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
            if (token_mask[m]) {
                int weight_index = sorted_token_ids_ptr[OFFSET_TOKEN_ID(m)] * stride_weight;
                float moe_weight = topk_weights_ptr[weight_index];
                for (int n = 0; n < BLOCK_SIZE_N; ++n) {
                    accumulator[m][n] *= moe_weight;
                }
            }
        }
    }

    // Write back the block of the output
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        if (token_mask[m]) {
            for (int n = 0; n < BLOCK_SIZE_N; ++n) {
                int c_index = (stride_cm * sorted_token_ids_ptr[OFFSET_TOKEN_ID(m)]) + (stride_cn * (pid_n * BLOCK_SIZE_N + n));
                if (pid_n * BLOCK_SIZE_N + n < N) {
                    c_ptr[c_index] = accumulator[m][n];
                }
            }
        }
    }
}

} // namespace