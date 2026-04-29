#include <CellShardPreprocess/runtime.hh>

#include <cstdio>

int main() {
    cspre::preprocess_state_view raw{};
    raw.assay = "scrna";
    raw.matrix_orientation = "observations_by_features";
    raw.matrix_state = "raw_counts";
    raw.raw_counts_available = 1u;

    cspre::status status{};
    if (!cspre::validate_raw_count_state(&raw, &status)) {
        std::fprintf(stderr, "%s\n", status.message);
        return 1;
    }

    cspre::preprocess_state_view processed = raw;
    processed.preprocess_available = 1u;
    if (cspre::reject_double_preprocess(&processed, &status)) return 2;
    if (status.code != cspre::status_already_preprocessed) return 3;

    cspre::preprocess_state_view normalized = raw;
    normalized.matrix_state = "normalized_log1p";
    if (cspre::validate_raw_count_state(&normalized, &status)) return 4;
    if (status.code != cspre::status_not_raw_counts) return 5;
    return 0;
}
