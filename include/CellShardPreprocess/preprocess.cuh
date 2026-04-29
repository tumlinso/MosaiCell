#pragma once

#include <CellShard/formats/blocked_ell.cuh>
#include <CellShard/runtime/device/sharded_device.cuh>
#include <CellShard/runtime/distributed/distributed.cuh>
#include <CellShard/runtime/mask_groups.cuh>
#include <CellShardPreprocess/aliases.hh>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cellshard_preprocess {

namespace cs_device = ::cellshard::device;
namespace cs_dist = ::cellshard::distributed;
namespace cs_runtime = ::cellshard::runtime;

static constexpr unsigned int CELLSHARD_PREPROCESS_MAX_QC_GROUPS = 32u;

enum gene_flag : unsigned char {
    gene_flag_none = 0u,
    gene_flag_mito = 1u << 0
};

enum qc_group_index : unsigned int {
    qc_group_mt = 0u,
    qc_group_ribo = 1u,
    qc_group_hb = 2u
};

constexpr std::uint32_t qc_group_bit(unsigned int group) {
    return group < CELLSHARD_PREPROCESS_MAX_QC_GROUPS ? (1u << group) : 0u;
}

struct alignas(16) cell_filter_params {
    float min_counts;
    unsigned int min_genes;
    float max_mito_fraction;
};

struct alignas(16) cell_qc_filter_params {
    float min_counts;
    unsigned int min_features;
    float max_group_fraction;
    unsigned int fraction_group_index;
};

struct alignas(16) gene_filter_params {
    float min_sum;
    float min_detected_cells;
    float min_variance;
};

struct alignas(16) qc_group_config_view {
    unsigned int group_count;
    const char * const *group_names;
    const std::uint32_t *feature_group_masks;
    const std::uint32_t *explicit_feature_group_masks;
};

struct alignas(16) device_qc_group_config_view {
    unsigned int group_count;
    const std::uint32_t *feature_group_masks;
};

struct alignas(16) cell_metrics_view {
    unsigned int rows;
    unsigned int group_count;
    float *total_counts;
    float *mito_counts;
    float *max_counts;
    unsigned int *detected_genes;
    unsigned char *keep_cells;
    float *cell_group_counts;
    float *cell_group_pct;
};

struct alignas(16) gene_metrics_view {
    unsigned int cols;
    float *sum;
    float *sq_sum;
    float *detected_cells;
    unsigned char *keep_genes;
    std::uint32_t *feature_group_masks;
    unsigned char *gene_flags;
};

struct alignas(16) preprocess_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    unsigned int rows_capacity;
    unsigned int cols_capacity;
    unsigned int values_capacity;
    unsigned int group_capacity;

    void *cell_block;
    void *gene_block;

    float *total_counts;
    float *mito_counts;
    float *max_counts;
    unsigned int *detected_genes;
    unsigned char *keep_cells;
    float *cell_group_counts;
    float *cell_group_pct;

    float *gene_sum;
    float *gene_sq_sum;
    float *gene_detected;
    unsigned char *keep_genes;
    unsigned char *gene_flags;
    std::uint32_t *feature_group_masks;
    float *active_rows;
    cs_runtime::sparse_group_reduce_workspace mask_groups;
};

struct alignas(16) part_preprocess_result {
    cell_metrics_view cell;
    gene_metrics_view gene;
};

struct alignas(16) preprocess_ranked_nccl_config {
    int world_size;
    const int *local_world_ranks;
    const void *unique_id;
    std::size_t unique_id_bytes;
};

struct alignas(16) preprocess_fleet_config {
    const int *device_ids;
    unsigned int device_count;
    unsigned int enable_peer_access;
    unsigned int stream_flags;
    const preprocess_ranked_nccl_config *ranked_nccl;
};

struct alignas(16) preprocess_fleet_result {
    unsigned int slot_count;
    unsigned int leader_index;
    part_preprocess_result *slot_results;
    gene_metrics_view reduced_gene;
};

struct alignas(16) preprocess_fleet_workspace {
    cs_dist::local_context local;
    unsigned int slot_count;
    unsigned int *slots;
    preprocess_workspace *devices;
    part_preprocess_result *results;
    void **reduce_scratch;
    std::size_t *reduce_scratch_bytes;
#if CELLSHARD_HAS_NCCL
    cs_dist::nccl_communicator ranked_nccl;
#endif
};

void init(preprocess_workspace *workspace);

void init(preprocess_fleet_workspace *fleet);

void clear(preprocess_workspace *workspace);

void clear(preprocess_fleet_workspace *fleet);

int setup(preprocess_workspace *workspace, int device, cudaStream_t stream = (cudaStream_t) 0);

int setup_fleet(preprocess_fleet_workspace *fleet, const preprocess_fleet_config *config = nullptr);

int reserve(preprocess_workspace *workspace,
            unsigned int rows,
            unsigned int cols,
            unsigned int values);

int reserve_qc_groups(preprocess_workspace *workspace,
                      unsigned int rows,
                      unsigned int cols,
                      unsigned int values,
                      unsigned int group_count);

int upload_feature_group_masks(preprocess_workspace *workspace,
                               unsigned int cols,
                               const std::uint32_t *host_masks);

int upload_gene_flags(preprocess_workspace *workspace,
                      unsigned int cols,
                      const unsigned char *host_flags);

int zero_gene_metrics(preprocess_workspace *workspace, unsigned int cols);

int compute_cell_metrics(const cs_device::blocked_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out);

int compute_cell_metrics(const cs_device::sliced_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out);

int compute_cell_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                             preprocess_workspace *workspace,
                                             const cell_filter_params *filter,
                                             cell_metrics_view *out);

int compute_qc_metrics(const cs_device::blocked_ell_view *src,
                       preprocess_workspace *workspace,
                       const qc_group_config_view *groups,
                       const cell_qc_filter_params *filter,
                       cell_metrics_view *out);

int compute_qc_metrics(const cs_device::sliced_ell_view *src,
                       preprocess_workspace *workspace,
                       const qc_group_config_view *groups,
                       const cell_qc_filter_params *filter,
                       cell_metrics_view *out);

int compute_qc_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                           preprocess_workspace *workspace,
                                           const qc_group_config_view *groups,
                                           const cell_qc_filter_params *filter,
                                           cell_metrics_view *out);

int compute_qc_metrics_fleet(const cs_device::blocked_ell_view *src_by_slot,
                             preprocess_fleet_workspace *fleet,
                             const qc_group_config_view *groups,
                             const cell_qc_filter_params *filter,
                             preprocess_fleet_result *out);

int compute_qc_metrics_fleet(const cs_device::sliced_ell_view *src_by_slot,
                             preprocess_fleet_workspace *fleet,
                             const qc_group_config_view *groups,
                             const cell_qc_filter_params *filter,
                             preprocess_fleet_result *out);

int normalize_log1p_inplace(cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum);

int normalize_log1p_inplace(cs_device::sliced_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum);

int normalize_log1p_compressed_fallback_inplace(cs_device::compressed_view *src,
                                                preprocess_workspace *workspace,
                                                const float *device_total_counts,
                                                const unsigned char *device_keep_cells,
                                                float target_sum);

int accumulate_gene_metrics(const cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out);

int accumulate_gene_metrics(const cs_device::sliced_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out);

int accumulate_gene_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                                preprocess_workspace *workspace,
                                                const unsigned char *device_keep_cells,
                                                gene_metrics_view *out);

int preprocess_blocked_ell_inplace(cs_device::blocked_ell_view *src,
                                   preprocess_workspace *workspace,
                                   const cell_filter_params *cell_filter,
                                   float target_sum,
                                   part_preprocess_result *out);

int preprocess_sliced_ell_inplace(cs_device::sliced_ell_view *src,
                                  preprocess_workspace *workspace,
                                  const cell_filter_params *cell_filter,
                                  float target_sum,
                                  part_preprocess_result *out);

int preprocess_compressed_fallback_inplace(cs_device::compressed_view *src,
                                           preprocess_workspace *workspace,
                                           const cell_filter_params *cell_filter,
                                           float target_sum,
                                           part_preprocess_result *out);

int preprocess_blocked_ell_qc_groups_inplace(cs_device::blocked_ell_view *src,
                                             preprocess_workspace *workspace,
                                             const qc_group_config_view *groups,
                                             const cell_qc_filter_params *cell_filter,
                                             float target_sum,
                                             part_preprocess_result *out);

int preprocess_sliced_ell_qc_groups_inplace(cs_device::sliced_ell_view *src,
                                            preprocess_workspace *workspace,
                                            const qc_group_config_view *groups,
                                            const cell_qc_filter_params *cell_filter,
                                            float target_sum,
                                            part_preprocess_result *out);

int preprocess_compressed_fallback_qc_groups_inplace(cs_device::compressed_view *src,
                                                     preprocess_workspace *workspace,
                                                     const qc_group_config_view *groups,
                                                     const cell_qc_filter_params *cell_filter,
                                                     float target_sum,
                                                     part_preprocess_result *out);

int preprocess_blocked_ell_qc_groups_fleet_inplace(cs_device::blocked_ell_view *src_by_slot,
                                                   preprocess_fleet_workspace *fleet,
                                                   const qc_group_config_view *groups,
                                                   const cell_qc_filter_params *cell_filter,
                                                   float target_sum,
                                                   preprocess_fleet_result *out);

int preprocess_sliced_ell_qc_groups_fleet_inplace(cs_device::sliced_ell_view *src_by_slot,
                                                  preprocess_fleet_workspace *fleet,
                                                  const qc_group_config_view *groups,
                                                  const cell_qc_filter_params *cell_filter,
                                                  float target_sum,
                                                  preprocess_fleet_result *out);

int build_gene_filter_mask(preprocess_workspace *workspace,
                           unsigned int cols,
                           const gene_filter_params *filter,
                           gene_metrics_view *out);

int finalize_gene_keep_mask_host(const float *gene_sum,
                                 const float *gene_sq_sum,
                                 const float *gene_detected,
                                 unsigned int cols,
                                 float kept_cells,
                                 const gene_filter_params *filter,
                                 unsigned char *keep_genes,
                                 unsigned int *kept_genes);

} // namespace cellshard_preprocess
