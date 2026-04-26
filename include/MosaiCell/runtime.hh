#pragma once

#include "preprocess.cuh"

#include <cstddef>
#include <cstdint>

namespace mosaicell {

enum status_code : int {
    status_ok = 0,
    status_invalid_argument = 1,
    status_not_raw_counts = 2,
    status_already_preprocessed = 3,
    status_unsupported_layout = 4
};

enum native_sparse_layout : std::uint32_t {
    native_sparse_unknown = 0u,
    native_sparse_blocked_ell = 1u,
    native_sparse_sliced_ell = 2u,
    native_sparse_compressed_fallback = 3u
};

enum value_precision : std::uint32_t {
    value_precision_unknown = 0u,
    value_precision_fp16 = 1u,
    value_precision_fp32_accumulator = 2u
};

struct status {
    int code;
    char message[192];
};

struct preprocess_state_view {
    const char *assay;
    const char *matrix_orientation;
    const char *matrix_state;
    const char *feature_namespace;
    unsigned int preprocess_available;
    unsigned int raw_counts_available;
    unsigned int processed_matrix_available;
};

struct adapter_source_view {
    const char *path;
    const char *format;
    const char *matrix_source;
    unsigned int allow_processed;
};

struct cellshard_stage_plan {
    native_sparse_layout layout;
    value_precision value_type;
    value_precision accumulator_type;
    unsigned int adapt_to_cellshard_first;
    unsigned int direct_external_kernels;
};

const char *version();

void clear_status(status *out);

int validate_raw_count_state(const preprocess_state_view *state, status *out);

int reject_double_preprocess(const preprocess_state_view *state, status *out);

int mark_mito_features_by_prefix(const char * const *feature_names,
                                 std::uint32_t feature_count,
                                 const char *prefix,
                                 unsigned char *gene_flags);

int plan_cellshard_adapter_stage(const adapter_source_view *source,
                                 cellshard_stage_plan *plan,
                                 status *out);

} // namespace mosaicell
