#include <CellShardPreprocess/runtime.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>

namespace {

int check_cuda(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

int close_enough(float got, float expected, float tol = 1.0e-3f) {
    return std::fabs(got - expected) <= tol;
}

int require(int condition, const char *label) {
    if (condition) return 1;
    std::fprintf(stderr, "%s\n", label);
    return 0;
}

} // namespace

int main() {
    const char *ids[4] = {"ENSG_MT-ND1", "RPS3", "HBA1", "USER_GENE"};
    const char *names[4] = {"MT-ND1", "RPS3", "HBA1", "USER_GENE"};
    const char *types[4] = {"gene", "gene", "gene", "gene"};
    const char *modalities[4] = {"rna", "rna", "rna", "rna"};
    std::uint32_t masks[4] = {};

    cspre::qc_feature_annotation_view features{ids, names, types, modalities, 4u};
    if (!cspre::compile_default_qc_feature_group_masks(&features, nullptr, masks)) return 1;
    if (!require((masks[0] & cspre::qc_group_bit(cspre::qc_group_mt)) != 0u, "mt default missing")) return 2;
    if (!require((masks[1] & cspre::qc_group_bit(cspre::qc_group_ribo)) != 0u, "ribo default missing")) return 3;
    if (!require((masks[2] & cspre::qc_group_bit(cspre::qc_group_hb)) != 0u, "hb default missing")) return 4;

    cspre::qc_group_rule_view user_rule{3u, "custom", nullptr, "USER_GENE", nullptr, "gene", "rna"};
    if (!cspre::compile_qc_feature_group_masks(&features, &user_rule, 1u, masks, masks)) return 5;
    masks[3] |= cspre::qc_group_bit(cspre::qc_group_mt);

    unsigned int h_cols[4] = {0u, 1u, 0u, 1u};
    const float host_values[16] = {
        5.0f, 3.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 4.0f, 0.0f, 6.0f,
        8.0f, 0.0f, 0.0f, 0.0f
    };
    __half h_values[16];
    for (int i = 0; i < 16; ++i) h_values[i] = __float2half(host_values[i]);

    unsigned int *d_cols = nullptr;
    __half *d_values = nullptr;
    if (!check_cuda(cudaMalloc((void **) &d_cols, sizeof(h_cols)), "cudaMalloc d_cols")) return 6;
    if (!check_cuda(cudaMalloc((void **) &d_values, sizeof(h_values)), "cudaMalloc d_values")) return 7;
    if (!check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice), "copy cols")) return 8;
    if (!check_cuda(cudaMemcpy(d_values, h_values, sizeof(h_values), cudaMemcpyHostToDevice), "copy values")) return 9;

    cellshard::device::blocked_ell_view view{};
    view.rows = 4u;
    view.cols = 4u;
    view.nnz = 7u;
    view.block_size = 2u;
    view.ell_cols = 4u;
    view.blockColIdx = d_cols;
    view.val = d_values;

    cspre::preprocess_workspace workspace;
    cspre::init(&workspace);
    if (!cspre::setup(&workspace, 0)) return 10;

    const char *group_names[4] = {"mt", "ribo", "hb", "custom"};
    cspre::qc_group_config_view groups{4u, group_names, masks, nullptr};
    cspre::cell_qc_filter_params filter{1.0f, 1u, 0.80f, cspre::qc_group_mt};
    cspre::cell_metrics_view metrics{};
    if (!cspre::compute_qc_metrics(&view, &workspace, &groups, &filter, &metrics)) return 11;
    if (!check_cuda(cudaStreamSynchronize(workspace.stream), "sync metrics")) return 12;

    float total[4] = {}, group_counts[16] = {}, group_pct[16] = {};
    unsigned int detected[4] = {};
    unsigned char keep[4] = {};
    if (!check_cuda(cudaMemcpy(total, metrics.total_counts, sizeof(total), cudaMemcpyDeviceToHost), "copy total")) return 13;
    if (!check_cuda(cudaMemcpy(group_counts, metrics.cell_group_counts, sizeof(group_counts), cudaMemcpyDeviceToHost), "copy group counts")) return 14;
    if (!check_cuda(cudaMemcpy(group_pct, metrics.cell_group_pct, sizeof(group_pct), cudaMemcpyDeviceToHost), "copy group pct")) return 15;
    if (!check_cuda(cudaMemcpy(detected, metrics.detected_genes, sizeof(detected), cudaMemcpyDeviceToHost), "copy detected")) return 16;
    if (!check_cuda(cudaMemcpy(keep, metrics.keep_cells, sizeof(keep), cudaMemcpyDeviceToHost), "copy keep")) return 17;

    if (!close_enough(total[0], 10.0f) || !close_enough(total[1], 0.0f)
        || !close_enough(total[2], 11.0f) || !close_enough(total[3], 8.0f)) return 18;
    if (detected[0] != 3u || detected[1] != 0u || detected[2] != 3u || detected[3] != 1u) return 19;
    if (keep[0] == 0u || keep[1] != 0u || keep[2] == 0u || keep[3] != 0u) return 20;
    if (!close_enough(group_counts[0], 5.0f) || !close_enough(group_counts[1], 3.0f)
        || !close_enough(group_counts[2], 2.0f) || !close_enough(group_counts[3], 0.0f)) return 21;
    if (!close_enough(group_counts[8], 7.0f) || !close_enough(group_counts[9], 4.0f)
        || !close_enough(group_counts[10], 0.0f) || !close_enough(group_counts[11], 6.0f)) return 22;
    if (!close_enough(group_pct[0], 50.0f) || !close_enough(group_pct[1], 30.0f)
        || !close_enough(group_pct[2], 20.0f) || !close_enough(group_pct[4], 0.0f)) {
        std::fprintf(stderr,
                     "bad group pct: %.4f %.4f %.4f %.4f %.4f\n",
                     group_pct[0],
                     group_pct[1],
                     group_pct[2],
                     group_pct[3],
                     group_pct[4]);
        return 23;
    }

    cspre::qc_group_config_view no_groups{0u, nullptr, nullptr, nullptr};
    cspre::cell_qc_filter_params no_group_filter{1.0f, 1u, 1.0f, 0u};
    if (!cspre::compute_qc_metrics(&view, &workspace, &no_groups, &no_group_filter, &metrics)) return 24;
    if (!check_cuda(cudaStreamSynchronize(workspace.stream), "sync no-group metrics")) return 25;
    if (!require(metrics.group_count == 0u, "no-group result exposed groups")) return 26;

    cspre::clear(&workspace);
    cudaFree(d_values);
    cudaFree(d_cols);
    return 0;
}
