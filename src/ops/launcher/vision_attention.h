#pragma once

#include "core/tensor.h"

#include <cstddef>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream);

std::int32_t vision_attention_uniform_tile(std::int32_t segment_length);

void vision_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out, cudaStream_t stream);

void vision_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               Tensor& out, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// Volta (sm_70) vision attention through the vendored llama.cpp flash-attention
// kernel. Defined in vision_attention_volta_flash.cu, the only vision translation
// unit that sees the vendored headers. See docs/volta-port.md.
//
// The staging is sized by the patch count: Q is padded to head_dim 128 in FP32,
// K/V in FP16, and the segment mask is O(P^2) because variable-length segments
// are expressed as a block-diagonal mask over one launch.
inline constexpr std::int32_t kVisionFlashPaddedHeadDim = 128;
inline constexpr std::int32_t kVisionFlashKeyPad        = 256;
inline constexpr std::int32_t kVisionFlashMaskRowPad    = 64;

std::size_t vision_attention_volta_flash_meta_elements(std::int32_t patches);

// Exactly one of cu_seqlens (variable segments) or segment_length (uniform) is
// supplied; pass nullptr / 0 for the other.
void vision_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor* cu_seqlens, std::int32_t segment_length,
                                         Tensor& q_f32, Tensor& k_f16, Tensor& v_f16, Tensor& mask,
                                         Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                         cudaStream_t stream);
#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
