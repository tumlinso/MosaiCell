#include <CellShardPreprocess/runtime.hh>

#include <cstdio>

int main() {
    cspre::adapter_source_view source{};
    source.path = "input.h5ad";
    source.format = "h5ad";
    source.matrix_source = "counts";
    source.allow_processed = 0u;

    cspre::cellshard_stage_plan plan{};
    cspre::status status{};
    if (!cspre::plan_cellshard_adapter_stage(&source, &plan, &status)) {
        std::fprintf(stderr, "%s\n", status.message);
        return 1;
    }
    if (plan.layout != cspre::native_sparse_blocked_ell) return 2;
    if (plan.value_type != cspre::value_precision_fp16) return 3;
    if (plan.accumulator_type != cspre::value_precision_fp32_accumulator) return 4;
    if (plan.adapt_to_cellshard_first == 0u || plan.direct_external_kernels != 0u) return 5;
    return 0;
}
