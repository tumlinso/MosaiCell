#include <CellShardPreprocess/runtime.hh>

#include <cstdio>
#include <cstring>

namespace cellshard_preprocess {

namespace {

void set_status(status *out, int code, const char *message) {
    if (out == nullptr) return;
    out->code = code;
    if (message == nullptr) message = "";
    std::snprintf(out->message, sizeof(out->message), "%s", message);
}

int streq(const char *a, const char *b) {
    if (a == nullptr || b == nullptr) return 0;
    return std::strcmp(a, b) == 0;
}

int starts_with(const char *value, const char *prefix) {
    if (value == nullptr || prefix == nullptr) return 0;
    while (*prefix != '\0') {
        if (*value == '\0' || *value != *prefix) return 0;
        ++value;
        ++prefix;
    }
    return 1;
}

int rule_metadata_matches(const qc_feature_annotation_view *features,
                          const qc_group_rule_view *rule,
                          std::uint32_t index) {
    const char *type = features->feature_types != nullptr ? features->feature_types[index] : nullptr;
    const char *modality = features->modalities != nullptr ? features->modalities[index] : nullptr;
    if (rule->feature_type != nullptr && !streq(type, rule->feature_type)) return 0;
    if (rule->modality != nullptr && !streq(modality, rule->modality)) return 0;
    return 1;
}

int rule_name_matches(const qc_feature_annotation_view *features,
                      const qc_group_rule_view *rule,
                      std::uint32_t index) {
    const char *id = features->feature_ids != nullptr ? features->feature_ids[index] : nullptr;
    const char *name = features->feature_names != nullptr ? features->feature_names[index] : nullptr;
    if (rule->prefix != nullptr && (starts_with(name, rule->prefix) || starts_with(id, rule->prefix))) return 1;
    if (rule->exact_feature_id != nullptr && streq(id, rule->exact_feature_id)) return 1;
    if (rule->exact_feature_name != nullptr && streq(name, rule->exact_feature_name)) return 1;
    return rule->prefix == nullptr && rule->exact_feature_id == nullptr && rule->exact_feature_name == nullptr;
}

} // namespace

const char *version() {
    return "0.1.0";
}

void clear_status(status *out) {
    set_status(out, status_ok, "");
}

int validate_raw_count_state(const preprocess_state_view *state, status *out) {
    if (state == nullptr) {
        set_status(out, status_invalid_argument, "missing preprocess state");
        return 0;
    }
    if (state->preprocess_available != 0u) {
        set_status(out, status_already_preprocessed, "dataset already contains persisted preprocess metadata");
        return 0;
    }
    if (state->raw_counts_available == 0u) {
        set_status(out, status_not_raw_counts, "raw counts are not available");
        return 0;
    }
    if (state->processed_matrix_available != 0u) {
        set_status(out, status_not_raw_counts, "active matrix is already processed");
        return 0;
    }
    if (state->matrix_state != nullptr && !streq(state->matrix_state, "raw_counts")) {
        set_status(out, status_not_raw_counts, "matrix_state is not raw_counts");
        return 0;
    }
    set_status(out, status_ok, "");
    return 1;
}

int reject_double_preprocess(const preprocess_state_view *state, status *out) {
    if (state == nullptr) {
        set_status(out, status_invalid_argument, "missing preprocess state");
        return 0;
    }
    if (state->preprocess_available != 0u) {
        set_status(out, status_already_preprocessed, "preprocess metadata already exists");
        return 0;
    }
    set_status(out, status_ok, "");
    return 1;
}

int mark_mito_features_by_prefix(const char * const *feature_names,
                                 std::uint32_t feature_count,
                                 const char *prefix,
                                 unsigned char *gene_flags) {
    if (gene_flags == nullptr) return 0;
    const char *active_prefix = prefix != nullptr ? prefix : "MT-";
    for (std::uint32_t i = 0u; i < feature_count; ++i) {
        const char *name = feature_names != nullptr ? feature_names[i] : nullptr;
        if (starts_with(name, active_prefix)) gene_flags[i] = (unsigned char) (gene_flags[i] | gene_flag_mito);
    }
    return 1;
}

int compile_qc_feature_group_masks(const qc_feature_annotation_view *features,
                                   const qc_group_rule_view *rules,
                                   std::uint32_t rule_count,
                                   const std::uint32_t *explicit_masks,
                                   std::uint32_t *feature_group_masks) {
    if (features == nullptr || feature_group_masks == nullptr) return 0;
    for (std::uint32_t i = 0u; i < features->feature_count; ++i) {
        feature_group_masks[i] = explicit_masks != nullptr ? explicit_masks[i] : 0u;
    }
    if (rules == nullptr || rule_count == 0u) return 1;
    for (std::uint32_t rule_idx = 0u; rule_idx < rule_count; ++rule_idx) {
        const qc_group_rule_view *rule = rules + rule_idx;
        if (rule->group_index >= CELLSHARD_PREPROCESS_MAX_QC_GROUPS) return 0;
        const std::uint32_t bit = qc_group_bit(rule->group_index);
        for (std::uint32_t feature = 0u; feature < features->feature_count; ++feature) {
            if (rule_metadata_matches(features, rule, feature) && rule_name_matches(features, rule, feature)) {
                feature_group_masks[feature] |= bit;
            }
        }
    }
    return 1;
}

int compile_default_qc_feature_group_masks(const qc_feature_annotation_view *features,
                                           const std::uint32_t *explicit_masks,
                                           std::uint32_t *feature_group_masks) {
    const qc_group_rule_view rules[] = {
        {qc_group_mt, "mt", "MT-", nullptr, nullptr, nullptr, nullptr},
        {qc_group_mt, "mt", "mt-", nullptr, nullptr, nullptr, nullptr},
        {qc_group_ribo, "ribo", "RPS", nullptr, nullptr, nullptr, nullptr},
        {qc_group_ribo, "ribo", "RPL", nullptr, nullptr, nullptr, nullptr},
        {qc_group_ribo, "ribo", "rps", nullptr, nullptr, nullptr, nullptr},
        {qc_group_ribo, "ribo", "rpl", nullptr, nullptr, nullptr, nullptr},
        {qc_group_hb, "hb", "HB", nullptr, nullptr, nullptr, nullptr},
        {qc_group_hb, "hb", "Hb", nullptr, nullptr, nullptr, nullptr},
        {qc_group_hb, "hb", "hb", nullptr, nullptr, nullptr, nullptr}
    };
    return compile_qc_feature_group_masks(features,
                                          rules,
                                          (std::uint32_t) (sizeof(rules) / sizeof(rules[0])),
                                          explicit_masks,
                                          feature_group_masks);
}

int plan_cellshard_adapter_stage(const adapter_source_view *source,
                                 cellshard_stage_plan *plan,
                                 status *out) {
    if (source == nullptr || plan == nullptr) {
        set_status(out, status_invalid_argument, "missing adapter source or output plan");
        return 0;
    }
    if (!streq(source->format, "h5ad") && !streq(source->format, "mtx")) {
        set_status(out, status_unsupported_layout, "unsupported adapter source format");
        return 0;
    }
    if (source->allow_processed == 0u && !streq(source->matrix_source, "raw") && !streq(source->matrix_source, "counts")) {
        set_status(out, status_not_raw_counts, "adapter source must identify a raw/count matrix");
        return 0;
    }
    plan->layout = native_sparse_blocked_ell;
    plan->value_type = value_precision_fp16;
    plan->accumulator_type = value_precision_fp32_accumulator;
    plan->adapt_to_cellshard_first = 1u;
    plan->direct_external_kernels = 0u;
    set_status(out, status_ok, "");
    return 1;
}

} // namespace cellshard_preprocess
