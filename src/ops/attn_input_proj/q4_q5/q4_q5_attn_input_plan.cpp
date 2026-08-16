#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_cutlass_sm70.h"
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols     = std::numeric_limits<std::int32_t>::max();
constexpr std::int32_t kHiddenShape = 5120;
constexpr std::int32_t kQRows       = 6144;
constexpr std::int32_t kKvRows      = 1024;

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5AttnInputScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// GroupedHomogeneousPairMma* need Ampere+ mma/ldmatrix and are trap-stubbed on sm_70.
// ParentSplitFixed's underlying kernels take cols as a runtime grid parameter (see
// q4_q5_attn_input_small_t.cu), so it generalizes past T=16 unchanged and stays the route for
// small T -- decode and small prefill chunks, bandwidth-bound territory where CUTLASS's fixed
// (T-independent) dequant cost wouldn't pay for itself.
//
// For wide T (real prefill), CutlassSm70TensorCore dequantizes both parent weights once each
// (query_key is q4, gate_value is q5) to scratch FP16 buffers and runs four NVIDIA CUTLASS
// Sm70 (mma.sync.m8n8k4) GEMMs -- one per output (q, k, gate, v), each a pointer offset into
// its shared dequant buffer, not a separate dequant pass. Same T=64 threshold as the other two
// CUTLASS conversions in this port (linear_swiglu, linear_add) -- see docs/volta-port.md.
constexpr std::array<RouteSpec, 2> kRoutes{{
    {{1, 63}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{64, kAnyCols}, Q4Q5AttnInputScheduleId::CutlassSm70TensorCore},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last == kAnyCols;
}
#else
constexpr std::array<RouteSpec, 3> kRoutes{{
    {{1, 16}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{17, 20}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3},
    {{21, kAnyCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last + 1 == kRoutes[2].cols.first && kRoutes[2].cols.last == kAnyCols;
}
#endif

static_assert(catalog_is_closed(), "attention input routes must be exact and closed");

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == kHiddenShape && problem.query_rows == kQRows &&
           problem.kv_rows == kKvRows && problem.padded_k == kHiddenShape;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r16.c64.s3";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::CutlassSm70TensorCore:
        return "attn_input_proj.q4_q5.cutlass_sm70.tensor_core";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4Q5AttnInputPlan plan{route.schedule, 0};
        switch (route.schedule) {
        case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
            return plan;
        case Q4Q5AttnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
            plan.workspace_bytes = q4_q5_attn_input_cutlass_workspace_bytes(problem.cols);
            return plan;
#else
            throw std::logic_error("Q4/Q5 attention input: CutlassSm70TensorCore is Volta-only");
#endif
        }
    }
    throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
}

std::size_t q4_q5_attn_input_capacity_workspace_bytes(std::int32_t min_cols,
                                                       std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 attention input: invalid column interval");
    }
    (void)q4_q5_attn_input_resolve_plan({kHiddenShape, kQRows, kKvRows, kHiddenShape, min_cols});
    (void)q4_q5_attn_input_resolve_plan({kHiddenShape, kQRows, kKvRows, kHiddenShape, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum = std::max(maximum, q4_q5_attn_input_resolve_plan(
                                        {kHiddenShape, kQRows, kKvRows, kHiddenShape, endpoint})
                                        .workspace_bytes);
    }
    return maximum;
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::CutlassSm70TensorCore:
#ifdef NINFER_VOLTA_BUILD
        q4_q5_attn_input_cutlass_sm70_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                             v, workspace, stream);
        return;
#else
        throw std::logic_error("Q4/Q5 attention input: CutlassSm70TensorCore is Volta-only");
#endif
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, WorkspaceArena& workspace, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  workspace, stream);
}

} // namespace ninfer::ops::detail
