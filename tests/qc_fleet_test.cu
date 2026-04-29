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

int close_enough(float got, float expected, float tol = 1.0e-2f) {
    return std::fabs(got - expected) <= tol;
}

template<typename T>
T *upload_on_device(int device, const T *host, std::size_t count, const char *label) {
    T *ptr = nullptr;
    if (!check_cuda(cudaSetDevice(device), "cudaSetDevice upload")) return nullptr;
    if (!check_cuda(cudaMalloc((void **) &ptr, count * sizeof(T)), label)) return nullptr;
    if (!check_cuda(cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy upload")) return nullptr;
    return ptr;
}

int sync_fleet(cspre::preprocess_fleet_workspace *fleet) {
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        const unsigned int slot = fleet->slots[i];
        if (!check_cuda(cudaSetDevice(fleet->local.device_ids[slot]), "cudaSetDevice sync fleet")) return 0;
        if (!check_cuda(cudaStreamSynchronize(fleet->local.streams[slot]), "cudaStreamSynchronize fleet")) return 0;
    }
    return 1;
}

int compare_array(const float *got, const float *expected, unsigned int count, const char *label) {
    for (unsigned int i = 0u; i < count; ++i) {
        if (!close_enough(got[i], expected[i])) {
            std::fprintf(stderr, "%s[%u] got %.6f expected %.6f\n", label, i, got[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

} // namespace

int main() {
    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) return 1;
    if (device_count <= 0) return 0;

    const std::uint32_t masks[4] = {
        cspre::qc_group_bit(cspre::qc_group_mt),
        cspre::qc_group_bit(cspre::qc_group_ribo),
        cspre::qc_group_bit(cspre::qc_group_hb),
        cspre::qc_group_bit(cspre::qc_group_mt) | cspre::qc_group_bit(3u)
    };
    const char *group_names[4] = {"mt", "ribo", "hb", "custom"};
    cspre::qc_group_config_view groups{4u, group_names, masks, nullptr};
    cspre::cell_qc_filter_params filter{1.0f, 1u, 0.80f, cspre::qc_group_mt};

    const unsigned int full_cols[4] = {0u, 1u, 0u, 1u};
    const unsigned int part_cols[2] = {0u, 1u};
    const float full_values_f32[16] = {
        5.0f, 3.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 4.0f, 0.0f, 6.0f,
        8.0f, 0.0f, 0.0f, 0.0f
    };
    __half full_values[16], part0_values[8], part1_values[8];
    for (int i = 0; i < 16; ++i) full_values[i] = __float2half(full_values_f32[i]);
    for (int i = 0; i < 8; ++i) {
        part0_values[i] = __float2half(full_values_f32[i]);
        part1_values[i] = __float2half(full_values_f32[i + 8]);
    }

    unsigned int *full_d_cols = upload_on_device(0, full_cols, 4u, "cudaMalloc full cols");
    __half *full_d_values = upload_on_device(0, full_values, 16u, "cudaMalloc full values");
    if (full_d_cols == nullptr || full_d_values == nullptr) return 2;

    cellshard::device::blocked_ell_view full_view{};
    full_view.rows = 4u;
    full_view.cols = 4u;
    full_view.nnz = 7u;
    full_view.block_size = 2u;
    full_view.ell_cols = 4u;
    full_view.blockColIdx = full_d_cols;
    full_view.val = full_d_values;

    cspre::preprocess_workspace single_ws;
    cspre::init(&single_ws);
    if (!cspre::setup(&single_ws, 0)) return 3;
    cspre::part_preprocess_result single{};
    if (!cspre::preprocess_blocked_ell_qc_groups_inplace(&full_view, &single_ws, &groups, &filter, 10000.0f, &single)) return 4;
    if (!check_cuda(cudaStreamSynchronize(single_ws.stream), "sync single")) return 5;

    float single_total[4] = {}, single_gene_sum[4] = {};
    if (!check_cuda(cudaMemcpy(single_total, single.cell.total_counts, sizeof(single_total), cudaMemcpyDeviceToHost), "copy single total")) return 6;
    if (!check_cuda(cudaMemcpy(single_gene_sum, single.gene.sum, sizeof(single_gene_sum), cudaMemcpyDeviceToHost), "copy single gene")) return 7;

    {
        unsigned int *fleet_cols = upload_on_device(0, full_cols, 4u, "cudaMalloc fleet one cols");
        __half *fleet_values = upload_on_device(0, full_values, 16u, "cudaMalloc fleet one values");
        if (fleet_cols == nullptr || fleet_values == nullptr) return 8;
        cellshard::device::blocked_ell_view fleet_view = full_view;
        fleet_view.blockColIdx = fleet_cols;
        fleet_view.val = fleet_values;
        int device_ids[1] = {0};
        cspre::preprocess_fleet_config config{device_ids, 1u, 1u, cudaStreamNonBlocking, nullptr};
        cspre::preprocess_fleet_workspace fleet;
        cspre::init(&fleet);
        if (!cspre::setup_fleet(&fleet, &config)) return 9;
        cspre::preprocess_fleet_result result{};
        if (!cspre::preprocess_blocked_ell_qc_groups_fleet_inplace(&fleet_view, &fleet, &groups, &filter, 10000.0f, &result)) return 10;
        if (!sync_fleet(&fleet)) return 11;
        float total[4] = {}, gene_sum[4] = {};
        if (!check_cuda(cudaMemcpy(total, result.slot_results[0].cell.total_counts, sizeof(total), cudaMemcpyDeviceToHost), "copy one total")) return 12;
        if (!check_cuda(cudaMemcpy(gene_sum, result.reduced_gene.sum, sizeof(gene_sum), cudaMemcpyDeviceToHost), "copy one gene")) return 13;
        if (!compare_array(total, single_total, 4u, "one-slot total")) return 14;
        if (!compare_array(gene_sum, single_gene_sum, 4u, "one-slot gene")) return 15;
        cspre::clear(&fleet);
        cudaFree(fleet_values);
        cudaFree(fleet_cols);
    }

    if (device_count >= 2) {
        unsigned int *part0_cols = upload_on_device(0, part_cols, 2u, "cudaMalloc part0 cols");
        __half *part0_vals = upload_on_device(0, part0_values, 8u, "cudaMalloc part0 vals");
        unsigned int *part1_cols = upload_on_device(1, part_cols, 2u, "cudaMalloc part1 cols");
        __half *part1_vals = upload_on_device(1, part1_values, 8u, "cudaMalloc part1 vals");
        if (part0_cols == nullptr || part0_vals == nullptr || part1_cols == nullptr || part1_vals == nullptr) return 16;

        cellshard::device::blocked_ell_view parts[2] = {};
        for (int i = 0; i < 2; ++i) {
            parts[i].rows = 2u;
            parts[i].cols = 4u;
            parts[i].block_size = 2u;
            parts[i].ell_cols = 4u;
        }
        parts[0].nnz = 3u;
        parts[0].blockColIdx = part0_cols;
        parts[0].val = part0_vals;
        parts[1].nnz = 4u;
        parts[1].blockColIdx = part1_cols;
        parts[1].val = part1_vals;

        int device_ids[2] = {0, 1};
        cspre::preprocess_fleet_config config{device_ids, 2u, 1u, cudaStreamNonBlocking, nullptr};
        cspre::preprocess_fleet_workspace fleet;
        cspre::init(&fleet);
        if (!cspre::setup_fleet(&fleet, &config)) return 17;
        cspre::preprocess_fleet_result result{};
        if (!cspre::preprocess_blocked_ell_qc_groups_fleet_inplace(parts, &fleet, &groups, &filter, 10000.0f, &result)) return 18;
        if (!sync_fleet(&fleet)) return 19;

        float total0[2] = {}, total1[2] = {}, gene_sum[4] = {};
        if (!check_cuda(cudaMemcpy(total0, result.slot_results[0].cell.total_counts, sizeof(total0), cudaMemcpyDeviceToHost), "copy total0")) return 20;
        if (!check_cuda(cudaMemcpy(total1, result.slot_results[1].cell.total_counts, sizeof(total1), cudaMemcpyDeviceToHost), "copy total1")) return 21;
        if (!check_cuda(cudaMemcpy(gene_sum, result.reduced_gene.sum, sizeof(gene_sum), cudaMemcpyDeviceToHost), "copy fleet gene")) return 22;
        if (!compare_array(total0, single_total, 2u, "fleet total0")) return 23;
        if (!compare_array(total1, single_total + 2, 2u, "fleet total1")) return 24;
        if (!compare_array(gene_sum, single_gene_sum, 4u, "fleet gene")) return 25;

        cspre::clear(&fleet);
        cudaSetDevice(0);
        cudaFree(part0_vals);
        cudaFree(part0_cols);
        cudaSetDevice(1);
        cudaFree(part1_vals);
        cudaFree(part1_cols);
    }

    cspre::clear(&single_ws);
    cudaSetDevice(0);
    cudaFree(full_d_values);
    cudaFree(full_d_cols);
    return 0;
}
