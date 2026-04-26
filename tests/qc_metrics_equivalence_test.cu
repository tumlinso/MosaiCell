#include <MosaiCell/preprocess.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

int check_cuda(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

int close_enough(float got, float expected) {
    return std::fabs(got - expected) <= 1.0e-3f;
}

} // namespace

int main() {
    unsigned int h_cols[2] = {0u, 1u};
    __half h_values[8];
    const float host_values[8] = {5.0f, 0.0f, 1.0f, 0.0f, 0.0f, 7.0f, 0.0f, 2.0f};
    for (int i = 0; i < 8; ++i) h_values[i] = __float2half(host_values[i]);

    unsigned int *d_cols = nullptr;
    __half *d_values = nullptr;
    if (!check_cuda(cudaMalloc((void **) &d_cols, sizeof(h_cols)), "cudaMalloc d_cols")) return 1;
    if (!check_cuda(cudaMalloc((void **) &d_values, sizeof(h_values)), "cudaMalloc d_values")) return 1;
    if (!check_cuda(cudaMemcpy(d_cols, h_cols, sizeof(h_cols), cudaMemcpyHostToDevice), "copy cols")) return 1;
    if (!check_cuda(cudaMemcpy(d_values, h_values, sizeof(h_values), cudaMemcpyHostToDevice), "copy values")) return 1;

    cellshard::device::blocked_ell_view view{};
    view.rows = 2u;
    view.cols = 4u;
    view.nnz = 4u;
    view.block_size = 2u;
    view.ell_cols = 4u;
    view.blockColIdx = d_cols;
    view.val = d_values;

    mosaicell::preprocess_workspace workspace;
    mosaicell::init(&workspace);
    if (!mosaicell::setup(&workspace, 0)) return 1;
    unsigned char flags[4] = {mosaicell::gene_flag_mito, 0u, 0u, 0u};
    if (!mosaicell::upload_gene_flags(&workspace, 4u, flags)) return 1;

    mosaicell::cell_filter_params filter{1.0f, 1u, 0.90f};
    mosaicell::cell_metrics_view metrics{};
    if (!mosaicell::compute_cell_metrics(&view, &workspace, &filter, &metrics)) return 1;
    if (!check_cuda(cudaStreamSynchronize(workspace.stream), "sync metrics")) return 1;

    float total[2] = {}, mito[2] = {}, vmax[2] = {};
    unsigned int detected[2] = {};
    unsigned char keep[2] = {};
    if (!check_cuda(cudaMemcpy(total, metrics.total_counts, sizeof(total), cudaMemcpyDeviceToHost), "copy total")) return 1;
    if (!check_cuda(cudaMemcpy(mito, metrics.mito_counts, sizeof(mito), cudaMemcpyDeviceToHost), "copy mito")) return 1;
    if (!check_cuda(cudaMemcpy(vmax, metrics.max_counts, sizeof(vmax), cudaMemcpyDeviceToHost), "copy max")) return 1;
    if (!check_cuda(cudaMemcpy(detected, metrics.detected_genes, sizeof(detected), cudaMemcpyDeviceToHost), "copy detected")) return 1;
    if (!check_cuda(cudaMemcpy(keep, metrics.keep_cells, sizeof(keep), cudaMemcpyDeviceToHost), "copy keep")) return 1;

    if (!close_enough(total[0], 6.0f) || !close_enough(total[1], 9.0f)) return 2;
    if (!close_enough(mito[0], 5.0f) || !close_enough(mito[1], 0.0f)) return 3;
    if (!close_enough(vmax[0], 5.0f) || !close_enough(vmax[1], 7.0f)) return 4;
    if (detected[0] != 2u || detected[1] != 2u) return 5;
    if (keep[0] == 0u || keep[1] == 0u) return 6;

    mosaicell::clear(&workspace);
    cudaFree(d_values);
    cudaFree(d_cols);
    return 0;
}
