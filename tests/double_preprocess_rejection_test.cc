#include <MosaiCell/runtime.hh>

#include <cstdio>

int main() {
    mosaicell::preprocess_state_view raw{};
    raw.assay = "scrna";
    raw.matrix_orientation = "observations_by_features";
    raw.matrix_state = "raw_counts";
    raw.raw_counts_available = 1u;

    mosaicell::status status{};
    if (!mosaicell::validate_raw_count_state(&raw, &status)) {
        std::fprintf(stderr, "%s\n", status.message);
        return 1;
    }

    mosaicell::preprocess_state_view processed = raw;
    processed.preprocess_available = 1u;
    if (mosaicell::reject_double_preprocess(&processed, &status)) return 2;
    if (status.code != mosaicell::status_already_preprocessed) return 3;

    mosaicell::preprocess_state_view normalized = raw;
    normalized.matrix_state = "normalized_log1p";
    if (mosaicell::validate_raw_count_state(&normalized, &status)) return 4;
    if (status.code != mosaicell::status_not_raw_counts) return 5;
    return 0;
}
