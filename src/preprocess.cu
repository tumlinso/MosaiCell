#include <MosaiCell/preprocess.cuh>

#include <cstdio>
#include <cstring>

#include <cuda_fp16.h>

namespace mosaicell {

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

std::size_t align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

int cuda_ok(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "MosaiCell CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

void bind_cell_metrics(preprocess_workspace *workspace, unsigned int rows, cell_metrics_view *out) {
    if (out == nullptr) return;
    out->rows = rows;
    out->total_counts = workspace->total_counts;
    out->mito_counts = workspace->mito_counts;
    out->max_counts = workspace->max_counts;
    out->detected_genes = workspace->detected_genes;
    out->keep_cells = workspace->keep_cells;
}

void bind_gene_metrics(preprocess_workspace *workspace, unsigned int cols, gene_metrics_view *out) {
    if (out == nullptr) return;
    out->cols = cols;
    out->sum = workspace->gene_sum;
    out->sq_sum = workspace->gene_sq_sum;
    out->detected_cells = workspace->gene_detected;
    out->keep_genes = workspace->keep_genes;
}

__global__ void compute_cell_metrics_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const unsigned char *__restrict__ gene_flags,
    cell_filter_params filter,
    float *__restrict__ total_counts,
    float *__restrict__ mito_counts,
    float *__restrict__ max_counts,
    unsigned int *__restrict__ detected_genes,
    unsigned char *__restrict__ keep_cells
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned int row = warp_global;

    while (row < src.rows) {
        float sum = 0.0f, mito = 0.0f, vmax = 0.0f, detected = 0.0f;
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        unsigned int ell_col = lane;

        while (ell_col < src.ell_cols) {
            const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
            const unsigned int lane_in_block = block_size != 0u ? ell_col % block_size : 0u;
            const unsigned int block_col = ell_width_blocks != 0u
                ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                : cellshard::sparse::blocked_ell_invalid_col;
            const unsigned int gene = block_col != cellshard::sparse::blocked_ell_invalid_col
                ? block_col * block_size + lane_in_block
                : src.cols;
            const float value = __half2float(src.val[(unsigned long) row * src.ell_cols + ell_col]);
            if (gene < src.cols && value != 0.0f) {
                sum += value;
                if (gene_flags != nullptr && (gene_flags[gene] & gene_flag_mito) != 0u) mito += value;
                vmax = fmaxf(vmax, value);
                detected += 1.0f;
            }
            ell_col += 32u;
        }

        sum = warp_sum(sum);
        mito = warp_sum(mito);
        vmax = warp_max(vmax);
        detected = warp_sum(detected);

        if (lane == 0u) {
            total_counts[row] = sum;
            mito_counts[row] = mito;
            max_counts[row] = vmax;
            detected_genes[row] = (unsigned int) detected;
            const float mito_fraction = sum > 0.0f ? mito / sum : 0.0f;
            keep_cells[row] = (unsigned char) (sum >= filter.min_counts
                                               && (unsigned int) detected >= filter.min_genes
                                               && mito_fraction <= filter.max_mito_fraction);
        }

        row += warp_stride;
    }
}

__global__ void normalize_log1p_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == nullptr ? 1 : keep_cells[row] != 0u;
        unsigned int ell_col = lane;

        if (keep) {
            while (ell_col < src.ell_cols) {
                const unsigned long offset = (unsigned long) row * src.ell_cols + ell_col;
                const float value = __half2float(src.val[offset]);
                src.val[offset] = value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
                ell_col += 32u;
            }
        }
        row += warp_stride;
    }
}

__global__ void accumulate_gene_metrics_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const unsigned char *__restrict__ keep_cells,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum
) {
    const unsigned long tid = (unsigned long) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned long stride = (unsigned long) (gridDim.x * blockDim.x);
    const unsigned long total = (unsigned long) src.rows * (unsigned long) src.ell_cols;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned long linear = tid;

    while (linear < total) {
        const unsigned int row = (unsigned int) (linear / src.ell_cols);
        const unsigned int ell_col = (unsigned int) (linear % src.ell_cols);
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
        const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
        const unsigned int block_col = ell_width_blocks != 0u
            ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
            : cellshard::sparse::blocked_ell_invalid_col;
        const unsigned int col = block_col != cellshard::sparse::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if ((keep_cells == nullptr || keep_cells[row] != 0u) && col < src.cols) {
            const float value = __half2float(src.val[linear]);
            if (value != 0.0f) {
                atomicAdd(gene_sum + col, value);
                atomicAdd(gene_detected + col, 1.0f);
                atomicAdd(gene_sq_sum + col, value * value);
            }
        }
        linear += stride;
    }
}

} // namespace

void init(preprocess_workspace *workspace) {
    if (workspace == nullptr) return;
    std::memset(workspace, 0, sizeof(*workspace));
    workspace->device = -1;
}

void clear(preprocess_workspace *workspace) {
    if (workspace == nullptr) return;
    if (workspace->device >= 0) (void) cudaSetDevice(workspace->device);
    if (workspace->owns_stream != 0 && workspace->stream != (cudaStream_t) 0) (void) cudaStreamDestroy(workspace->stream);
    if (workspace->gene_block != nullptr) (void) cudaFree(workspace->gene_block);
    if (workspace->cell_block != nullptr) (void) cudaFree(workspace->cell_block);
    init(workspace);
}

int setup(preprocess_workspace *workspace, int device, cudaStream_t stream) {
    if (workspace == nullptr) return 0;
    clear(workspace);
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice preprocess setup")) return 0;
    workspace->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&workspace->stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags preprocess")) return 0;
        workspace->owns_stream = 1;
    } else {
        workspace->stream = stream;
        workspace->owns_stream = 0;
    }
    return 1;
}

int reserve(preprocess_workspace *workspace, unsigned int rows, unsigned int cols, unsigned int values) {
    if (workspace == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess reserve")) return 0;

    if (rows > workspace->rows_capacity) {
        std::size_t bytes = 0u;
        char *base = nullptr;
        if (workspace->cell_block != nullptr) (void) cudaFree(workspace->cell_block);
        workspace->cell_block = nullptr;

        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned int));
        bytes += (std::size_t) rows * sizeof(unsigned int);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        bytes += (std::size_t) rows * sizeof(unsigned char);

        if (bytes != 0u && !cuda_ok(cudaMalloc(&workspace->cell_block, bytes), "cudaMalloc preprocess cell block")) return 0;
        base = (char *) workspace->cell_block;
        bytes = 0u;
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->total_counts = (float *) (base + bytes);
        bytes += (std::size_t) rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->mito_counts = (float *) (base + bytes);
        bytes += (std::size_t) rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->max_counts = (float *) (base + bytes);
        bytes += (std::size_t) rows * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned int));
        workspace->detected_genes = (unsigned int *) (base + bytes);
        bytes += (std::size_t) rows * sizeof(unsigned int);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        workspace->keep_cells = (unsigned char *) (base + bytes);
        workspace->rows_capacity = rows;
    }

    if (cols > workspace->cols_capacity) {
        std::size_t bytes = 0u;
        char *base = nullptr;
        if (workspace->gene_block != nullptr) (void) cudaFree(workspace->gene_block);
        workspace->gene_block = nullptr;

        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        bytes += (std::size_t) cols * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        bytes += (std::size_t) cols * sizeof(unsigned char);

        if (bytes != 0u && !cuda_ok(cudaMalloc(&workspace->gene_block, bytes), "cudaMalloc preprocess gene block")) return 0;
        base = (char *) workspace->gene_block;
        bytes = 0u;
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->gene_sum = (float *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->gene_sq_sum = (float *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(float));
        workspace->gene_detected = (float *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(float);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        workspace->keep_genes = (unsigned char *) (base + bytes);
        bytes += (std::size_t) cols * sizeof(unsigned char);
        bytes = align_up_bytes(bytes, alignof(unsigned char));
        workspace->gene_flags = (unsigned char *) (base + bytes);
        workspace->cols_capacity = cols;
    }

    if (values > workspace->values_capacity) workspace->values_capacity = values;
    return 1;
}

int upload_gene_flags(preprocess_workspace *workspace,
                      unsigned int cols,
                      const unsigned char *host_flags) {
    if (workspace == nullptr) return 0;
    if (!reserve(workspace, workspace->rows_capacity, cols, workspace->values_capacity)) return 0;
    if (cols == 0u) return 1;
    if (host_flags == nullptr) {
        return cuda_ok(cudaMemsetAsync(workspace->gene_flags, 0, (std::size_t) cols, workspace->stream),
                       "cudaMemsetAsync gene flags");
    }
    return cuda_ok(cudaMemcpyAsync(workspace->gene_flags,
                                   host_flags,
                                   (std::size_t) cols,
                                   cudaMemcpyHostToDevice,
                                   workspace->stream),
                   "cudaMemcpyAsync gene flags");
}

int zero_gene_metrics(preprocess_workspace *workspace, unsigned int cols) {
    if (workspace == nullptr) return 0;
    if (!reserve(workspace, workspace->rows_capacity, cols, workspace->values_capacity)) return 0;
    if (!cuda_ok(cudaMemsetAsync(workspace->gene_sum, 0, (std::size_t) cols * sizeof(float), workspace->stream),
                 "cudaMemsetAsync gene sum")) return 0;
    if (!cuda_ok(cudaMemsetAsync(workspace->gene_sq_sum, 0, (std::size_t) cols * sizeof(float), workspace->stream),
                 "cudaMemsetAsync gene sq sum")) return 0;
    if (!cuda_ok(cudaMemsetAsync(workspace->gene_detected, 0, (std::size_t) cols * sizeof(float), workspace->stream),
                 "cudaMemsetAsync gene detected")) return 0;
    return cuda_ok(cudaMemsetAsync(workspace->keep_genes, 0, (std::size_t) cols, workspace->stream),
                   "cudaMemsetAsync keep genes");
}

int compute_cell_metrics(const cs_device::blocked_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->rows * src->ell_cols)) return 0;
    if (!cuda_ok(cudaMemsetAsync(workspace->keep_cells, 0, (std::size_t) src->rows, workspace->stream),
                 "cudaMemsetAsync keep cells")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    compute_cell_metrics_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        workspace->gene_flags,
        *filter,
        workspace->total_counts,
        workspace->mito_counts,
        workspace->max_counts,
        workspace->detected_genes,
        workspace->keep_cells);
    if (!cuda_ok(cudaGetLastError(), "compute_cell_metrics_blocked_ell_kernel")) return 0;
    bind_cell_metrics(workspace, src->rows, out);
    return 1;
}

int normalize_log1p_inplace(cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    normalize_log1p_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum);
    return cuda_ok(cudaGetLastError(), "normalize_log1p_blocked_ell_kernel");
}

int accumulate_gene_metrics(const cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->rows * src->ell_cols)) return 0;
    unsigned int blocks = (unsigned int) ((((unsigned long) src->rows * (unsigned long) src->ell_cols) + 255ul) >> 8);
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    accumulate_gene_metrics_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_keep_cells,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum);
    if (!cuda_ok(cudaGetLastError(), "accumulate_gene_metrics_blocked_ell_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int preprocess_blocked_ell_inplace(cs_device::blocked_ell_view *src,
                                   preprocess_workspace *workspace,
                                   const cell_filter_params *cell_filter,
                                   float target_sum,
                                   part_preprocess_result *out) {
    cell_metrics_view cell{};
    gene_metrics_view gene{};
    if (!compute_cell_metrics(src, workspace, cell_filter, &cell)) return 0;
    if (!normalize_log1p_inplace(src, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
    if (!zero_gene_metrics(workspace, src->cols)) return 0;
    if (!accumulate_gene_metrics(src, workspace, cell.keep_cells, &gene)) return 0;
    if (out != nullptr) {
        out->cell = cell;
        out->gene = gene;
    }
    return 1;
}

int finalize_gene_keep_mask_host(const float *gene_sum,
                                 const float *gene_sq_sum,
                                 const float *gene_detected,
                                 unsigned int cols,
                                 float kept_cells,
                                 const gene_filter_params *filter,
                                 unsigned char *keep_genes,
                                 unsigned int *kept_genes) {
    if (gene_sum == nullptr || gene_sq_sum == nullptr || gene_detected == nullptr || filter == nullptr || keep_genes == nullptr) return 0;
    const float inv_cells = kept_cells > 0.0f ? 1.0f / kept_cells : 0.0f;
    unsigned int kept = 0u;
    for (unsigned int gene = 0u; gene < cols; ++gene) {
        const float mean = gene_sum[gene] * inv_cells;
        float var = gene_sq_sum[gene] * inv_cells - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const unsigned char keep = (unsigned char) (gene_sum[gene] >= filter->min_sum
                                                    && gene_detected[gene] >= filter->min_detected_cells
                                                    && var >= filter->min_variance);
        keep_genes[gene] = keep;
        kept += keep != 0u ? 1u : 0u;
    }
    if (kept_genes != nullptr) *kept_genes = kept;
    return 1;
}

} // namespace mosaicell
