// ninfer::ops::detail - Volta (sm_70) vision attention via the vendored
// llama.cpp MMA flash-attention kernel.
//
// The Ampere+ vision kernel (ops/kernel/vision_attention.cuh) is built on
// ldmatrix/mma.m16n8k16 and has no SIMT sibling, so below sm_80 it reaches
// ops/common/mma.cuh's __trap() and vision is simply unavailable. This routes it
// through the same vendored kernel the prefill path uses.
//
// Three properties of vision attention make it a good fit, and one make it
// awkward:
//   - 16 q-heads and 16 kv-heads, so gqa_ratio == 1 -> ncols2 == 1.
//   - non-causal, which is the kernel's natural behaviour: causality in
//     llama.cpp is expressed entirely through the mask, never implied.
//   - head_dim 72 is not one of the kernel's supported sizes, but 128 is, and
//     zero-padding the feature axis is exact: padded Q/K contribute nothing to a
//     dot product and padded V columns land in output lanes we discard.
//   - segments are variable-length (cu_seqlens), which the kernel has no notion
//     of. Expressed here as a block-diagonal mask, which is what a mask is for.
//
// This is one launch over all P patches. The mask is therefore O(P^2), and the
// kernel computes P^2 scores where only sum(len_i^2) are needed -- exact, but
// wasteful in proportion to the segment count. For the common single-image case
// there is no waste at all. See the note in docs/volta-port.md before raising
// the patch ceiling.

#include "fattn-mma-f16.cuh" // vendored; must precede ninfer headers (defines WARP_SIZE)

#include "core/tensor.h"
#include "ops/launcher/vision_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

constexpr int kVisionHeadDim = 72;  // real feature width
constexpr int kPaddedHeadDim = 128; // what the kernel is instantiated for
constexpr int kVisionHeads   = 16;

constexpr int kNcols2 = 1; // gqa_ratio == 1
constexpr int kNcols1 = 32; // ncols1*ncols2 must be >= 32 on Volta
constexpr int kNcols  = kNcols1 * kNcols2;

constexpr int kKeyPad     = FATTN_KQ_STRIDE;
constexpr int kMaskRowPad = 64;

constexpr int round_up(int n, int m) { return ((n + m - 1) / m) * m; }

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

// BF16 [72,16,P] (token stride may be padded) -> FP32 [128,16,P] contiguous.
// The kernel indexes Q as float2; feature lanes 72..127 are written zero so they
// contribute nothing to any score.
__global__ void vision_pad_q_kernel(const __nv_bfloat16* __restrict__ q, std::int64_t token_stride,
                                    float* __restrict__ out, int patches) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int d     = threadIdx.x;
    if (token >= patches) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kPaddedHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * token);
    if (d >= kVisionHeadDim) {
        out[dst] = 0.0f;
        return;
    }
    const std::int64_t src = static_cast<std::int64_t>(token) * token_stride +
                             static_cast<std::int64_t>(head) * kVisionHeadDim + d;
    out[dst] = __bfloat162float(q[src]);
}

// Same for K/V, into FP16, over the padded key extent. Rows past the true patch
// count are zeroed: the mask gives them weight zero and 0 * NaN would still be NaN.
__global__ void vision_pad_kv_kernel(const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v, std::int64_t token_stride,
                                     half* __restrict__ k_out, half* __restrict__ v_out,
                                     int patches, int patches_padded) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int d     = threadIdx.x;
    if (token >= patches_padded) { return; }

    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kPaddedHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * token);
    if (token >= patches || d >= kVisionHeadDim) {
        k_out[dst] = __float2half(0.0f);
        v_out[dst] = __float2half(0.0f);
        return;
    }
    const std::int64_t src = static_cast<std::int64_t>(token) * token_stride +
                             static_cast<std::int64_t>(head) * kVisionHeadDim + d;
    k_out[dst] = __float2half(__bfloat162float(k[src]));
    v_out[dst] = __float2half(__bfloat162float(v[src]));
}

// Block-diagonal mask: query i attends key j exactly when they share a segment.
//
// Serves both public entry points. With cu_seqlens != nullptr the bounds are read
// on the device, so no host round trip is needed; with segment_length > 0 the
// uniform overload's bounds are pure arithmetic and no descriptor is needed at
// all. Rows past the patch count get an empty range, hence a fully masked row.
__global__ void vision_segment_mask_kernel(const std::int32_t* __restrict__ cu_seqlens, int segments,
                                           int segment_length, half* __restrict__ mask, int patches,
                                           int keys_padded) {
    const int row = blockIdx.x;

    int begin = 0;
    int end   = 0;
    if (row < patches) {
        if (cu_seqlens == nullptr) {
            begin = (row / segment_length) * segment_length;
            end   = begin + segment_length;
        } else {
            // Segment counts are small (one per image), so a linear scan costs
            // less than the setup a search would need.
            for (int s = 0; s < segments; ++s) {
                const int lo = cu_seqlens[s];
                const int hi = cu_seqlens[s + 1];
                if (row >= lo && row < hi) {
                    begin = lo;
                    end   = hi;
                    break;
                }
            }
        }
    }

    half* mask_row = mask + static_cast<std::int64_t>(row) * keys_padded;
    for (int key = threadIdx.x; key < keys_padded; key += blockDim.x) {
        const bool visible = key >= begin && key < end;
        mask_row[key]      = visible ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

// FP32 [128,16,P] -> BF16 [72,16,P] contiguous, dropping the padded lanes.
__global__ void vision_unpad_out_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                        int patches) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int d     = threadIdx.x;
    if (token >= patches || d >= kVisionHeadDim) { return; }

    const std::int64_t src =
        static_cast<std::int64_t>(d) + kPaddedHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * token);
    const std::int64_t dst =
        static_cast<std::int64_t>(d) + kVisionHeadDim * (head + static_cast<std::int64_t>(kVisionHeads) * token);
    out[dst] = __float2bfloat16(in[src]);
}

// ---------------------------------------------------------------------------

struct VisionFlashConfig {
    int    nthreads      = 0;
    int    nwarps        = 0;
    int    nbatch_fa     = 0;
    size_t nbytes_shared = 0;
    int    blocks_per_sm = 0;
    int    nsm           = 0;
};

const VisionFlashConfig& vision_flash_config() {
    static const VisionFlashConfig config = [] {
        VisionFlashConfig c;

        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        c.nsm = prop.multiProcessorCount;

        const int cc = prop.major * 100 + prop.minor * 10;

        const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2, cc);

        c.nthreads  = ggml_cuda_fattn_mma_get_nthreads (kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        c.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        c.nwarps    = c.nthreads / WARP_SIZE;

        const int cols_per_warp = std::min(kNcols, get_cols_per_warp(cc));

        const size_t shared_KV_1stage = size_t(c.nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_KV_2stage = size_t(c.nbatch_fa) *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_Q         = size_t(kNcols)      * (kPaddedHeadDim/2 + 4)                  * sizeof(half2);
        const size_t shared_mask      = size_t(kNcols1)     * (c.nbatch_fa/2 + 4)                     * sizeof(half2);
        const size_t shared_combine   = size_t(c.nwarps)*cols_per_warp * (nbatch_combine + 4)         * sizeof(half2);
        const size_t shared_KV        = nstages <= 1 ? shared_KV_1stage : shared_KV_2stage;

        c.nbytes_shared = std::max(shared_combine, Q_in_reg
            ? std::max(shared_Q, shared_KV + shared_mask)
            :          shared_Q + shared_KV + shared_mask);

        auto kernel = flash_attn_ext_f16<kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2, false, false>;
        cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, c.nbytes_shared);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &c.blocks_per_sm, reinterpret_cast<const void *>(kernel), c.nthreads, c.nbytes_shared);
        if (c.blocks_per_sm <= 0) { c.blocks_per_sm = 1; }
        return c;
    }();
    return config;
}

} // namespace

std::size_t vision_attention_volta_flash_meta_elements(std::int32_t patches) {
    const VisionFlashConfig& c = vision_flash_config();
    const int ntiles_x         = (patches + kNcols1 - 1) / kNcols1;
    const int ntiles_dst       = ntiles_x * kVisionHeads;
    const int nblocks          = std::max(c.blocks_per_sm * c.nsm, ntiles_dst);
    return static_cast<std::size_t>(nblocks) * kNcols * (2 + kPaddedHeadDim / 2);
}

void vision_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor* cu_seqlens, std::int32_t segment_length,
                                         Tensor& q_f32, Tensor& k_f16, Tensor& v_f16, Tensor& mask,
                                         Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                         cudaStream_t stream) {
    const VisionFlashConfig& c = vision_flash_config();

    const std::int32_t patches  = q.ne[2];
    const std::int32_t segments = cu_seqlens != nullptr ? cu_seqlens->ne[0] - 1 : 0;
    const std::int32_t keys     = round_up(patches, kKeyPad);
    const std::int32_t rows     = round_up(patches, kMaskRowPad);

    // Token stride in elements; the caller may hand us a padded qkv plane.
    const std::int64_t token_stride = q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));

    vision_pad_q_kernel<<<dim3(patches, kVisionHeads), kPaddedHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), token_stride,
        static_cast<float*>(q_f32.data), patches);

    vision_pad_kv_kernel<<<dim3(keys, kVisionHeads), kPaddedHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        token_stride, static_cast<half*>(k_f16.data), static_cast<half*>(v_f16.data), patches, keys);

    vision_segment_mask_kernel<<<rows, 256, 0, stream>>>(
        cu_seqlens != nullptr ? static_cast<const std::int32_t*>(cu_seqlens->data) : nullptr,
        segments, segment_length, static_cast<half*>(mask.data), patches, keys);

    const std::int64_t out_elements =
        static_cast<std::int64_t>(patches) * kVisionHeads * kPaddedHeadDim;
    cudaMemsetAsync(out_f32.data, 0, static_cast<std::size_t>(out_elements) * sizeof(float), stream);

    // Stream-K decomposition, matching launch_fattn.
    const int ntiles_x   = (patches + kNcols1 - 1) / kNcols1;
    const int ntiles_dst = ntiles_x * kVisionHeads;
    const int ntiles_KV  = (keys + c.nbatch_fa - 1) / c.nbatch_fa;
    const int max_blocks = c.blocks_per_sm * c.nsm;

    const int raw     = std::min(max_blocks, ntiles_KV * ntiles_dst);
    const int rounded = (raw / ntiles_dst) * ntiles_dst;
    const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    const int nblocks = loss <= 5 ? rounded : raw;

    const std::int32_t nb01 = kPaddedHeadDim * kVisionHeads * sizeof(float);
    const std::int32_t nb02 = kPaddedHeadDim                * sizeof(float);
    const std::int32_t nb11 = kPaddedHeadDim * kVisionHeads * sizeof(half);
    const std::int32_t nb12 = kPaddedHeadDim                * sizeof(half);
    const std::int32_t nb31 = keys                          * sizeof(half);

    const uint3 ne01_fd = init_fastdiv_values(patches);

    // 1/sqrt(72) over the real feature width, not the padded one.
    const float scale = 1.0f / sqrtf(static_cast<float>(kVisionHeadDim));

    flash_attn_ext_f16<kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2, false, false>
        <<<dim3(nblocks, 1, 1), dim3(WARP_SIZE, c.nwarps, 1), c.nbytes_shared, stream>>>(
            reinterpret_cast<const char *>(q_f32.data),
            reinterpret_cast<const char *>(k_f16.data),
            reinterpret_cast<const char *>(v_f16.data),
            reinterpret_cast<const char *>(mask.data),
            nullptr, nullptr, static_cast<float*>(out_f32.data),
            static_cast<float2*>(dst_meta.data),
            scale, 0.0f, 1.0f, 1.0f, /*n_head_log2=*/16u, 0.0f,
            kPaddedHeadDim, ne01_fd, kVisionHeads, 1, nb01, nb02, 0,
            kPaddedHeadDim, keys, kVisionHeads, 1, nb11, nb12, 0,
            nb11, nb12, 0,
            patches, 1, 1,
            nb31, 0, 0);

    if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
        const uint3 fd0 = init_fastdiv_values(ntiles_x * kVisionHeads);
        const uint3 fd1 = init_fastdiv_values(ntiles_x);
        const uint3 fd2 = init_fastdiv_values(ntiles_x);

        flash_attn_stream_k_fixup_uniform<kPaddedHeadDim, kNcols1, kNcols2>
            <<<dim3((unsigned) ntiles_dst, kNcols1, kNcols2), dim3(kPaddedHeadDim, 1, 1), 0, stream>>>(
                static_cast<float*>(out_f32.data), static_cast<float2*>(dst_meta.data), patches,
                kVisionHeads, kVisionHeads, nblocks, 1, nblocks / ntiles_dst, fd0, fd1, fd2);
    } else if (ntiles_dst % nblocks != 0) {
        const int total_work = ntiles_KV * ntiles_dst;

        const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * kVisionHeads);
        const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x);
        const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
        const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

        flash_attn_stream_k_fixup_general<kPaddedHeadDim, kNcols1, kNcols2>
            <<<dim3((unsigned) nblocks, kNcols1, kNcols2), dim3(kPaddedHeadDim, 1, 1), 0, stream>>>(
                static_cast<float*>(out_f32.data), static_cast<float2*>(dst_meta.data), patches,
                kVisionHeads, 1, total_work, fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
    }

    vision_unpad_out_kernel<<<dim3(patches, kVisionHeads), kPaddedHeadDim, 0, stream>>>(
        static_cast<const float*>(out_f32.data), static_cast<__nv_bfloat16*>(out.data), patches);
}

} // namespace ninfer::ops::detail
