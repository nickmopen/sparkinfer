// int8-KV variant of the Gemma 4 LOCAL flash-decode (hd256, sliding window 1024).
//
// The local KV pool is read every decode step over the whole window (<=1024
// tokens) — ~209 MB/token at bf16 across the 25 local layers. Storing K/V as
// int8 with a per-(token, kv_head) fp16 scale halves that traffic; decode is
// bandwidth-bound, so the read halving converts ~directly to speed at ctx>=1024.
//
// Quant mirrors the merged Qwen3.6 hd256 int8 path (#284): per-head symmetric
// amax -> int8, one fp16 scale per (physical token slot, kv_head). K is already
// RoPE'd by launch_rope before it reaches the append kernel here (Gemma 4 keeps
// rope and kv-append as separate launches), so this append only quantizes.
//
// Opt-in from the runtime via SPARKINFER_GEMMA4_KV_INT8; default path stays bf16.
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// ---- int8 append: per-head amax quant of (already-RoPE'd) K and raw V --------
// grid = (2*n_kv_heads, n_tokens); blockDim = head_dim (one thread per dim).
// unit in [0,n_kv_heads) quantizes K; [n_kv_heads,2*n_kv_heads) quantizes V.
__global__ void gemma4_local_kv_append_int8_kernel(
    const __nv_bfloat16* __restrict__ k,   // [n_tokens, n_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,
    signed char* __restrict__ k_pool,      // [n_blocks, block_size, n_kv_heads, head_dim]
    signed char* __restrict__ v_pool,
    __half* __restrict__ k_scale,          // [n_blocks, block_size, n_kv_heads]
    __half* __restrict__ v_scale,
    const int* __restrict__ block_table,   // [n_tokens, max_blocks_per_seq]
    const int* __restrict__ write_pos,     // [n_tokens]
    int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq
) {
    const int tok  = blockIdx.y;
    const int unit = blockIdx.x;
    const int t    = threadIdx.x;          // 0..head_dim-1
    const bool is_k = unit < n_kv_heads;
    const int  hh   = is_k ? unit : (unit - n_kv_heads);

    const int pos    = write_pos[tok];
    const int blk    = pos / block_size, within = pos % block_size;
    const int phys   = block_table[tok * max_blocks_per_seq + blk];
    const size_t ctok = (size_t)(phys * block_size + within);

    const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim + t;
    const size_t dst  = (ctok * n_kv_heads + hh) * head_dim + t;
    const float val = __bfloat162float((is_k ? k : v)[base]);

    // per-head symmetric amax across the head_dim lanes
    __shared__ float s_red[8];             // head_dim/32 warp partials (<=256 -> <=8)
    float amax = fabsf(val);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    if ((t & 31) == 0) s_red[t >> 5] = amax;
    __syncthreads();
    if (t == 0) {
        float a = 0.f;
        for (int w = 0; w < (head_dim >> 5); w++) a = fmaxf(a, s_red[w]);
        s_red[0] = a;
    }
    __syncthreads();

    const float d  = s_red[0] / 127.0f;
    const int   qi = (s_red[0] == 0.f) ? 0 : (int)roundf(val / d);
    if (is_k) {
        k_pool[dst] = (signed char)qi;
        if (t == 0) k_scale[ctok * n_kv_heads + hh] = __float2half(d);
    } else {
        v_pool[dst] = (signed char)qi;
        if (t == 0) v_scale[ctok * n_kv_heads + hh] = __float2half(d);
    }
}

// ---- int8 read: same math as flash_decode_local_kernel, int8 KV + scale ------
__device__ __forceinline__ float loci8_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

template <int HEAD_DIM, int NWARPS, int BLOCK_SIZE>
__global__ void flash_decode_local_int8_kernel(
    const __nv_bfloat16* __restrict__ q,        // [num_seqs, num_q_heads, HEAD_DIM]
    const signed char* __restrict__ k_pool,     // [num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM]
    const signed char* __restrict__ v_pool,
    const __half* __restrict__ k_scale,         // [num_blocks, BLOCK_SIZE, num_kv_heads]
    const __half* __restrict__ v_scale,
    const int* __restrict__ block_table,        // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out,            // [num_seqs, num_q_heads, HEAD_DIM]
    const float scale,
    const int num_q_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int window
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq     = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int warp    = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int q_head  = kv_head * NWARPS + warp;

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = s_k + BLOCK_SIZE * HEAD_DIM;

    float q_reg[ELEMS];
    const __nv_bfloat16* qp = q + (size_t)(seq * num_q_heads + q_head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) q_reg[e] = __bfloat162float(qp[lane + e * 32]);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    const int seq_len      = seq_lens[seq];
    const int window_start = max(0, seq_len - window);
    const int start_blk    = window_start / BLOCK_SIZE;
    const int n_blocks     = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int blk = start_blk; blk < n_blocks; blk++) {
        const int phys     = block_table[seq * max_blocks_per_seq + blk];
        const int blk_base = blk * BLOCK_SIZE;
        const int hi       = min(BLOCK_SIZE, seq_len - blk_base);   // valid rows present
        const int lo       = max(0, window_start - blk_base);       // first attended row

        // dequant int8 KV -> shared. One scale per (token row, kv_head).
        for (int i = threadIdx.x; i < hi * HEAD_DIM; i += blockDim.x) {
            const int within = i / HEAD_DIM;
            const int d      = i % HEAD_DIM;
            const size_t row  = (size_t)(phys * BLOCK_SIZE + within) * num_kv_heads + kv_head;
            const size_t base = row * HEAD_DIM + d;
            s_k[i] = (float)k_pool[base] * __half2float(k_scale[row]);
            s_v[i] = (float)v_pool[base] * __half2float(v_scale[row]);
        }
        __syncthreads();

        for (int t = lo; t < hi; t++) {
            float partial = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                partial += q_reg[e] * s_k[t * HEAD_DIM + lane + e * 32];
            const float score = loci8_warp_sum(partial) * scale;

            const float m_new = fmaxf(m, score);
            const float corr  = __expf(m - m_new);
            const float p     = __expf(score - m_new);
            l = l * corr + p;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                acc[e] = acc[e] * corr + p * s_v[t * HEAD_DIM + lane + e * 32];
            m = m_new;
        }
        __syncthreads();
    }

    const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    __nv_bfloat16* op = out + (size_t)(seq * num_q_heads + q_head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) op[lane + e * 32] = __float2bfloat16(acc[e] * inv_l);
}

template __global__ void flash_decode_local_int8_kernel<256, 2, 16>(
    const __nv_bfloat16*, const signed char*, const signed char*,
    const __half*, const __half*, const int*, const int*, __nv_bfloat16*,
    float, int, int, int, int);

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_gemma4_local_kv_append_int8(
    const void* k_new, const void* v_new,
    void* k_pool, void* v_pool, void* k_scale, void* v_scale,
    const int* block_table, const int* write_pos,
    int num_seqs, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, cudaStream_t stream
) {
    dim3 grid(2 * num_kv_heads, num_seqs);
    gemma4_local_kv_append_int8_kernel<<<grid, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_new),
        reinterpret_cast<const __nv_bfloat16*>(v_new),
        reinterpret_cast<signed char*>(k_pool), reinterpret_cast<signed char*>(v_pool),
        reinterpret_cast<__half*>(k_scale), reinterpret_cast<__half*>(v_scale),
        block_table, write_pos, num_kv_heads, head_dim, block_size, max_blocks_per_seq);
}

void launch_flash_decode_local_hd256_int8(
    const void* q, const void* k_pool, const void* v_pool,
    const void* k_scale, const void* v_scale,
    const int* block_table, const int* seq_lens, void* out,
    int num_seqs, int num_kv_heads,
    int block_size, int max_blocks_per_window,
    float scale, cudaStream_t stream
) {
    constexpr int HEAD_DIM = 256, NWARPS = 2, BLOCK_SIZE = 16, WINDOW = 1024;
    const int num_q_heads = num_kv_heads * NWARPS;
    dim3 grid(num_seqs, num_kv_heads);
    size_t smem = 2 * BLOCK_SIZE * HEAD_DIM * sizeof(float);   // 32 KB
    flash_decode_local_int8_kernel<HEAD_DIM, NWARPS, BLOCK_SIZE><<<grid, NWARPS * 32, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const signed char*>(k_pool),
        reinterpret_cast<const signed char*>(v_pool),
        reinterpret_cast<const __half*>(k_scale),
        reinterpret_cast<const __half*>(v_scale),
        block_table, seq_lens, reinterpret_cast<__nv_bfloat16*>(out),
        scale, num_q_heads, num_kv_heads, max_blocks_per_window, WINDOW);
    (void)block_size;
}
#endif

} // namespace kernels
} // namespace sparkinfer
