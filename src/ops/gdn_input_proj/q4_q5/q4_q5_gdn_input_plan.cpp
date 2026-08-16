#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_cutlass_sm70.h"
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5GdnInputScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// GroupedMixedMmaR64C128 needs Ampere+ mma/ldmatrix and is trap-stubbed on sm_70.
// IndependentDirectFixed's underlying kernels (q4_rowsplit_gemm_simt_kernel,
// q5_rowsplit_gemm_simt_kernel) are plain SIMT with cols as a runtime grid parameter, so it
// generalizes past T=16 unchanged and stays the route for small T -- decode and small prefill
// chunks, bandwidth-bound territory where CUTLASS's fixed (T-independent) dequant cost
// wouldn't pay for itself.
//
// For wide T (real prefill), CutlassSm70TensorCore dequantizes both parent weights once each
// (qk_weight is q4, value_z_weight is q5) to scratch FP16 buffers and runs three NVIDIA
// CUTLASS Sm70 (mma.sync.m8n8k4) GEMMs: qk (into qkv[0:4096]), value (into qkv[4096:10240],
// a pointer offset into the same dequant buffer as z, not a separate dequant pass), and z.
// This is the single largest remaining prefill cost this port found by profiling -- GDN layers
// vastly outnumber full-attention layers in this hybrid model, so this op's SIMT fallback
// dominated total prefill GPU time even after the other three CUTLASS conversions landed.
// Same T=64 threshold as the rest of this port -- see docs/volta-port.md.
constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 63}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{64, kAnyCols}, Q4Q5GdnInputScheduleId::CutlassSm70TensorCore},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}
#else
constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 16}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{17, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}
#endif

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    case Q4Q5GdnInputScheduleId::CutlassSm70TensorCore:
        return "gdn_input_proj.q4_q5.cutlass_sm70.tensor_core";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

const char* q4_q5_gdn_input_conv_schedule_name(Q4Q5GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused:
        return "gdn_input_proj_conv.q4_q5.projection_epilogue_fused";
    case Q4Q5GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q4_q5.materialized";
    }
    return "gdn_input_proj_conv.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4Q5GdnInputPlan plan{route.schedule, 0};
        switch (route.schedule) {
        case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
            return plan;
        case Q4Q5GdnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
            plan.workspace_bytes = q4_q5_gdn_input_cutlass_workspace_bytes(problem.cols);
            return plan;
#else
            throw std::logic_error("Q4/Q5 GDN input: CutlassSm70TensorCore is Volta-only");
#endif
        }
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 GDN input: invalid column interval");
    }
    constexpr std::int32_t kInputRows = 5120, kQkRows = 4096, kValueZRows = 12288,
                           kQkvRows = 10240, kZRows = 6144;
    (void)q4_q5_gdn_input_resolve_plan(
        {kInputRows, kQkRows, kValueZRows, kQkvRows, kZRows, kInputRows, min_cols});
    (void)q4_q5_gdn_input_resolve_plan(
        {kInputRows, kQkRows, kValueZRows, kQkvRows, kZRows, kInputRows, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum = std::max(maximum,
                           q4_q5_gdn_input_resolve_plan({kInputRows, kQkRows, kValueZRows,
                                                         kQkvRows, kZRows, kInputRows, endpoint})
                               .workspace_bytes);
    }
    return maximum;
}

Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q5_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q4Q5GdnInputConvScheduleId::Materialized}; }
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
        q4_q5_gdn_input_cutlass_sm70_launch(x, qk_weight, value_z_weight, qkv, z, workspace,
                                            stream);
        return;
#else
        throw std::logic_error("Q4/Q5 GDN input: CutlassSm70TensorCore is Volta-only");
#endif
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
