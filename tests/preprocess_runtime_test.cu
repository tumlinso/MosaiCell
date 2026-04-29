#include <CellShardPreprocess/preprocess.cuh>

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

int close_enough(float got, float expected, float tol = 1.0e-2f) {
    return std::fabs(got - expected) <= tol;
}

} // namespace

int main() {
    unsigned int h_cols[2] = {0u, 1u};
    __half h_values[8];
    h_values[0] = __float2half(5.0f);
    h_values[1] = __float2half(0.0f);
    h_values[2] = __float2half(1.0f);
    h_values[3] = __float2half(0.0f);
    h_values[4] = __float2half(0.0f);
    h_values[5] = __float2half(7.0f);
    h_values[6] = __float2half(0.0f);
    h_values[7] = __float2half(2.0f);

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

    cspre::preprocess_workspace workspace;
    cspre::init(&workspace);
    if (!cspre::setup(&workspace, 0)) return 1;
    unsigned char flags[4] = {cspre::gene_flag_mito, 0u, 0u, 0u};
    if (!cspre::upload_gene_flags(&workspace, 4u, flags)) return 1;

    cspre::cell_filter_params filter{1.0f, 1u, 0.90f};
    cspre::part_preprocess_result result{};
    if (!cspre::preprocess_blocked_ell_inplace(&view, &workspace, &filter, 10000.0f, &result)) return 1;
    if (!check_cuda(cudaStreamSynchronize(workspace.stream), "sync preprocess")) return 1;

    float total[2] = {};
    unsigned int detected[2] = {};
    unsigned char keep[2] = {};
    float gene_sum[4] = {};
    if (!check_cuda(cudaMemcpy(total, result.cell.total_counts, sizeof(total), cudaMemcpyDeviceToHost), "copy total")) return 1;
    if (!check_cuda(cudaMemcpy(detected, result.cell.detected_genes, sizeof(detected), cudaMemcpyDeviceToHost), "copy detected")) return 1;
    if (!check_cuda(cudaMemcpy(keep, result.cell.keep_cells, sizeof(keep), cudaMemcpyDeviceToHost), "copy keep")) return 1;
    if (!check_cuda(cudaMemcpy(gene_sum, result.gene.sum, sizeof(gene_sum), cudaMemcpyDeviceToHost), "copy gene sum")) return 1;

    if (!close_enough(total[0], 6.0f) || !close_enough(total[1], 9.0f)) return 2;
    if (detected[0] != 2u || detected[1] != 2u) return 3;
    if (keep[0] == 0u || keep[1] == 0u) return 4;
    if (!(gene_sum[0] > 0.0f && gene_sum[1] > 0.0f && gene_sum[2] > 0.0f && gene_sum[3] > 0.0f)) return 5;

    cspre::clear(&workspace);
    cudaFree(d_values);
    cudaFree(d_cols);
    return 0;
}
