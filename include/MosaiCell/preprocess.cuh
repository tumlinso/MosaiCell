#pragma once

#include <CellShard/formats/blocked_ell.cuh>
#include <CellShard/runtime/device/sharded_device.cuh>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace mosaicell {

namespace cs_device = ::cellshard::device;

enum gene_flag : unsigned char {
    gene_flag_none = 0u,
    gene_flag_mito = 1u << 0
};

struct alignas(16) cell_filter_params {
    float min_counts;
    unsigned int min_genes;
    float max_mito_fraction;
};

struct alignas(16) gene_filter_params {
    float min_sum;
    float min_detected_cells;
    float min_variance;
};

struct alignas(16) cell_metrics_view {
    unsigned int rows;
    float *total_counts;
    float *mito_counts;
    float *max_counts;
    unsigned int *detected_genes;
    unsigned char *keep_cells;
};

struct alignas(16) gene_metrics_view {
    unsigned int cols;
    float *sum;
    float *sq_sum;
    float *detected_cells;
    unsigned char *keep_genes;
};

struct alignas(16) preprocess_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    unsigned int rows_capacity;
    unsigned int cols_capacity;
    unsigned int values_capacity;

    void *cell_block;
    void *gene_block;

    float *total_counts;
    float *mito_counts;
    float *max_counts;
    unsigned int *detected_genes;
    unsigned char *keep_cells;

    float *gene_sum;
    float *gene_sq_sum;
    float *gene_detected;
    unsigned char *keep_genes;
    unsigned char *gene_flags;
};

struct alignas(16) part_preprocess_result {
    cell_metrics_view cell;
    gene_metrics_view gene;
};

void init(preprocess_workspace *workspace);

void clear(preprocess_workspace *workspace);

int setup(preprocess_workspace *workspace, int device, cudaStream_t stream = (cudaStream_t) 0);

int reserve(preprocess_workspace *workspace,
            unsigned int rows,
            unsigned int cols,
            unsigned int values);

int upload_gene_flags(preprocess_workspace *workspace,
                      unsigned int cols,
                      const unsigned char *host_flags);

int zero_gene_metrics(preprocess_workspace *workspace, unsigned int cols);

int compute_cell_metrics(const cs_device::blocked_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out);

int normalize_log1p_inplace(cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum);

int accumulate_gene_metrics(const cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out);

int preprocess_blocked_ell_inplace(cs_device::blocked_ell_view *src,
                                   preprocess_workspace *workspace,
                                   const cell_filter_params *cell_filter,
                                   float target_sum,
                                   part_preprocess_result *out);

int finalize_gene_keep_mask_host(const float *gene_sum,
                                 const float *gene_sq_sum,
                                 const float *gene_detected,
                                 unsigned int cols,
                                 float kept_cells,
                                 const gene_filter_params *filter,
                                 unsigned char *keep_genes,
                                 unsigned int *kept_genes);

} // namespace mosaicell
