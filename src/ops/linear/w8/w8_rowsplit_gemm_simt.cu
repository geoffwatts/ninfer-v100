#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include "ops/common/math.h"
#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlockDefault = 8;
constexpr int kStages              = 2;

#ifdef NINFER_VOLTA_BUILD
// Same "request the max shared-memory carveout once" pattern as q5_rowsplit_gemv's occupancy
// fix -- ncu showed shared memory co-limiting at the same block count as registers once the
// launch_bounds fix above landed, so raising the carveout gives the higher minBlocks target
// room to actually take effect instead of shared memory silently capping it back down.
template <typename KernelPtr>
inline void w8_simt_request_max_shared_carveout(KernelPtr kernel) {
    static bool done = [&] {
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)done;
}
#endif

template <int ColsPerWarp, int ColWarpsPerRow, bool Full>
void launch_tt(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* scales,
               __nv_bfloat16* outp, std::int32_t n, std::int32_t k, std::int32_t t,
               std::int32_t padded_k, std::int32_t full_slabs, cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlockDefault * 32;
    constexpr int kRowsPerCta = kRowsPerBlockDefault / ColWarpsPerRow;
    constexpr int kColsPerCta = ColsPerWarp * ColWarpsPerRow;
    const dim3 grid(static_cast<unsigned>(div_up(n, kRowsPerCta)),
                    static_cast<unsigned>(div_up(t, kColsPerCta)), 1u);
    const W8ContiguousOutput output{outp, n};
#ifdef NINFER_VOLTA_BUILD
    w8_simt_request_max_shared_carveout(
        w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, ColsPerWarp, kRowsPerBlockDefault,
                                     kStages, Full, W8Epilogue::Store, W8ContiguousOutput,
                                     ColWarpsPerRow>);
#endif
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, ColsPerWarp, kRowsPerBlockDefault, kStages,
                                 Full, W8Epilogue::Store, W8ContiguousOutput,
                                 ColWarpsPerRow><<<grid, kBlockThreads, 0, stream>>>(
        xp, codes, scales, output, n, k, t, padded_k, full_slabs);
}

template <int ColsPerWarp, int ColWarpsPerRow, bool Full>
void launch_slice(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const auto* xp       = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (x.ne[0] % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? x.ne[0] / 1024 : 0;

    launch_tt<ColsPerWarp, ColWarpsPerRow, Full>(xp, static_cast<const std::uint8_t*>(w.qdata),
                                 static_cast<const std::uint8_t*>(w.scales),
                                 static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], x.ne[1],
                                 w.padded_shape[1], full_slabs, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <int ColsPerWarp, int ColWarpsPerRow = 1>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kRowsPerCta = kRowsPerBlockDefault / ColWarpsPerRow;
    constexpr int kColsPerCta = ColsPerWarp * ColWarpsPerRow;
    // The exact/full C16 specialization makes nvcc fully unroll both paired column warps on
    // sm_70 and spills badly (T=16 measured 53.0 ms versus 19.0 ms for the otherwise identical
    // predicated body). Keep the established full specialization for the one-warp schedules,
    // but deliberately compile the paired-warp route through the lean predicated body.
    const bool full = ColWarpsPerRow == 1 && (out.ne[0] % kRowsPerCta) == 0 &&
                      (x.ne[1] % kColsPerCta) == 0;
    for_each_token_slice(x.ne[1], kColsPerCta, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_slice<ColsPerWarp, ColWarpsPerRow, true>(x_slice, w, out_slice, stream);
        } else {
            launch_slice<ColsPerWarp, ColWarpsPerRow, false>(x_slice, w, out_slice, stream);
        }
    });
}

} // namespace

void launch_w8_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<4>(x, w, out, stream);
}

void launch_w8_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<8>(x, w, out, stream);
}

void launch_w8_simt_r4_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<8, 2>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
