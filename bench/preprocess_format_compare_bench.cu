#include "benchmark_mutex.hh"
#include <CellShard/CellShard.hh>
#include <CellShardPreprocess/preprocess.cuh>
#include "../../CellShard/src/convert/blocked_ell_from_compressed.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace cs = ::cellshard;
namespace csv = ::cellshard::device;
namespace csc = ::cellshard::convert;
namespace cspre = ::cellshard_preprocess;

enum class format_mode {
    compressed,
    blocked_ell,
    sliced_ell,
    both,
    all
};

struct config {
    format_mode mode = format_mode::both;
    unsigned int device = 0u;
    unsigned int parts = 16u;
    unsigned int rows_per_part = 8192u;
    unsigned int cols = 16384u;
    unsigned int avg_nnz_per_row = 64u;
    unsigned int shards = 4u;
    unsigned int repeats = 3u;
    unsigned int seed = 7u;
    float target_sum = 10000.0f;
    float min_counts = 500.0f;
    unsigned int min_genes = 200u;
    float max_mito_fraction = 0.2f;
    float min_gene_sum = 1.0f;
    float min_gene_detected = 5.0f;
    float min_gene_variance = 0.01f;
};

struct result {
    const char *label = "";
    double convert_ms = 0.0;
    double part_loop_ms = 0.0;
    double gene_mask_ms = 0.0;
    unsigned int tuned_block_size = 0u;
    double tuned_fill_ratio = 0.0;
    std::size_t tuned_padded_bytes = 0u;
    unsigned long rows = 0ul;
    unsigned int cols = 0u;
    unsigned long nnz = 0ul;
    unsigned long partitions = 0ul;
    unsigned long shards = 0ul;
    unsigned long kept_genes = 0ul;
    double kept_cells = 0.0;
    double gene_sum_checksum = 0.0;
};

static int check_cuda(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --mode {compressed|blocked_ell|sliced_ell|both|all}  Format mode. Default: both\n"
                 "  --device N                            CUDA device. Default: 0\n"
                 "  --parts N                             Partition count. Default: 16\n"
                 "  --rows-per-partition N                Rows per partition. Default: 8192\n"
                 "  --cols N                              Feature count. Default: 16384\n"
                 "  --avg-nnz-row N                       Average nnz per row. Default: 64\n"
                 "  --shards N                            Shard count. Default: 4\n"
                 "  --repeats N                           Repeat count. Default: 3\n"
                 "  --seed N                              RNG seed. Default: 7\n"
                 "  --target-sum F                        Normalize target. Default: 10000\n"
                 "  --min-counts F                        Cell min counts. Default: 500\n"
                 "  --min-genes N                         Cell min genes. Default: 200\n"
                 "  --max-mito-fraction F                 Cell max mito fraction. Default: 0.2\n"
                 "  --min-gene-sum F                      Gene min sum. Default: 1\n"
                 "  --min-gene-detected F                 Gene min detected cells. Default: 5\n"
                 "  --min-gene-variance F                 Gene min variance. Default: 0.01\n",
                 argv0);
}

static int parse_u32(const char *text, unsigned int *value) {
    char *end = 0;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != 0 || parsed > 0xfffffffful) return 0;
    *value = (unsigned int) parsed;
    return 1;
}

static int parse_f32(const char *text, float *value) {
    char *end = 0;
    const float parsed = std::strtof(text, &end);
    if (text == end || *end != 0) return 0;
    *value = parsed;
    return 1;
}

static int parse_mode(const char *text, format_mode *value) {
    if (std::strcmp(text, "compressed") == 0) {
        *value = format_mode::compressed;
        return 1;
    }
    if (std::strcmp(text, "blocked_ell") == 0) {
        *value = format_mode::blocked_ell;
        return 1;
    }
    if (std::strcmp(text, "sliced_ell") == 0) {
        *value = format_mode::sliced_ell;
        return 1;
    }
    if (std::strcmp(text, "both") == 0) {
        *value = format_mode::both;
        return 1;
    }
    if (std::strcmp(text, "all") == 0) {
        *value = format_mode::all;
        return 1;
    }
    return 0;
}

static int parse_args(int argc, char **argv, config *cfg) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (!parse_mode(argv[++i], &cfg->mode)) return 0;
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->device)) return 0;
        } else if (std::strcmp(argv[i], "--parts") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->parts)) return 0;
        } else if (std::strcmp(argv[i], "--rows-per-partition") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->rows_per_part)) return 0;
        } else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->cols)) return 0;
        } else if (std::strcmp(argv[i], "--avg-nnz-row") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->avg_nnz_per_row)) return 0;
        } else if (std::strcmp(argv[i], "--shards") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->shards)) return 0;
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->repeats)) return 0;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->seed)) return 0;
        } else if (std::strcmp(argv[i], "--target-sum") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->target_sum)) return 0;
        } else if (std::strcmp(argv[i], "--min-counts") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_counts)) return 0;
        } else if (std::strcmp(argv[i], "--min-genes") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->min_genes)) return 0;
        } else if (std::strcmp(argv[i], "--max-mito-fraction") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->max_mito_fraction)) return 0;
        } else if (std::strcmp(argv[i], "--min-gene-sum") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_gene_sum)) return 0;
        } else if (std::strcmp(argv[i], "--min-gene-detected") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_gene_detected)) return 0;
        } else if (std::strcmp(argv[i], "--min-gene-variance") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_gene_variance)) return 0;
        } else {
            return 0;
        }
    }

    return cfg->parts != 0u
        && cfg->rows_per_part != 0u
        && cfg->cols != 0u
        && cfg->shards != 0u
        && cfg->repeats != 0u;
}

static void build_row_counts(std::vector<unsigned int> *counts,
                             unsigned int rows,
                             unsigned int cols,
                             unsigned int avg_nnz_per_row,
                             std::mt19937 *rng) {
    std::vector<float> weights(rows, 0.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    unsigned long long assigned = 0ull;
    const unsigned long long target_nnz = (unsigned long long) rows * (unsigned long long) avg_nnz_per_row;

    counts->assign(rows, 0u);
    for (unsigned int i = 0; i < rows; ++i) {
        const float heavy = unit(*rng) < 0.15f ? 6.0f + unit(*rng) * 4.0f : 0.5f + unit(*rng) * 1.5f;
        weights[i] = heavy;
    }

    float weight_sum = 0.0f;
    for (float w : weights) weight_sum += w;
    for (unsigned int i = 0; i < rows; ++i) {
        unsigned int count = (unsigned int) ((target_nnz * (unsigned long long) (weights[i] * 1024.0f / weight_sum)) >> 10);
        if (count > cols) count = cols;
        (*counts)[i] = count;
        assigned += count;
    }

    while (assigned < target_nnz) {
        const unsigned int row = (*rng)() % rows;
        if ((*counts)[row] < cols) {
            ++(*counts)[row];
            ++assigned;
        }
    }
    while (assigned > target_nnz) {
        const unsigned int row = (*rng)() % rows;
        if ((*counts)[row] != 0u) {
            --(*counts)[row];
            --assigned;
        }
    }
}

static cs::sparse::compressed *make_compressed_part(unsigned int rows,
                                                    unsigned int cols,
                                                    unsigned int avg_nnz_per_row,
                                                    unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<unsigned int> row_counts;
    unsigned long long nnz = 0ull;
    auto *part = new cs::sparse::compressed;

    build_row_counts(&row_counts, rows, cols, avg_nnz_per_row, &rng);
    for (unsigned int row = 0; row < rows; ++row) nnz += row_counts[row];

    cs::sparse::init(part, rows, cols, (cs::types::nnz_t) nnz, cs::sparse::compressed_by_row);
    if (!cs::sparse::allocate(part)) {
        delete part;
        return nullptr;
    }

    part->majorPtr[0] = 0u;
    for (unsigned int row = 0; row < rows; ++row) {
        const unsigned int count = row_counts[row];
        const unsigned int begin = part->majorPtr[row];
        const unsigned int max_start = cols > count ? cols - count : 0u;
        const unsigned int start = max_start != 0u ? (rng() % (max_start + 1u)) : 0u;
        part->majorPtr[row + 1u] = begin + count;
        for (unsigned int j = 0; j < count; ++j) {
            part->minorIdx[begin + j] = start + j;
            part->val[begin + j] = __float2half((float) (1u + ((row + j) % 23u)));
        }
    }
    return part;
}

static int build_compressed_matrix(const config &cfg, cs::sharded<cs::sparse::compressed> *out) {
    cs::init(out);
    for (unsigned int part = 0; part < cfg.parts; ++part) {
        cs::sparse::compressed *matrix_part = make_compressed_part(cfg.rows_per_part,
                                                                   cfg.cols,
                                                                   cfg.avg_nnz_per_row,
                                                                   cfg.seed + 17u * part);
        if (matrix_part == nullptr) return 0;
        if (!cs::append_partition(out, matrix_part)) return 0;
    }
    return cs::set_equal_shards(out, cfg.shards);
}

static int build_whole_compressed(const cs::sharded<cs::sparse::compressed> *src,
                                  cs::sparse::compressed *whole,
                                  std::vector<cs::types::ptr_t> *row_ptr,
                                  std::vector<cs::types::idx_t> *col_idx,
                                  std::vector<::real::storage_t> *values) {
    cs::sparse::init(whole, (cs::types::dim_t) src->rows, (cs::types::dim_t) src->cols, (cs::types::nnz_t) src->nnz, cs::sparse::compressed_by_row);
    row_ptr->assign((std::size_t) src->rows + 1u, 0u);
    col_idx->assign((std::size_t) src->nnz, 0u);
    values->assign((std::size_t) src->nnz, (__half) 0);

    cs::types::nnz_t cursor = 0u;
    (*row_ptr)[0] = 0u;
    for (unsigned long part = 0; part < src->num_partitions; ++part) {
        const cs::sparse::compressed *p = src->parts[part];
        if (p == nullptr || p->axis != cs::sparse::compressed_by_row) return 0;
        for (cs::types::u32 row = 0u; row < p->rows; ++row) {
            const cs::types::ptr_t begin = p->majorPtr[row];
            const cs::types::ptr_t end = p->majorPtr[row + 1u];
            const cs::types::ptr_t count = end - begin;
            if (count != 0u) {
                std::memcpy(col_idx->data() + cursor, p->minorIdx + begin, (std::size_t) count * sizeof(cs::types::idx_t));
                std::memcpy(values->data() + cursor, p->val + begin, (std::size_t) count * sizeof(::real::storage_t));
            }
            cursor += count;
            (*row_ptr)[src->partition_offsets[part] + row + 1u] = cursor;
        }
    }

    whole->majorPtr = row_ptr->data();
    whole->minorIdx = col_idx->data();
    whole->val = values->data();
    return 1;
}

static int build_blocked_ell_matrix(const config &cfg,
                                    const cs::sharded<cs::sparse::compressed> *src,
                                    cs::sharded<cs::sparse::blocked_ell> *dst,
                                    result *meta) {
    const unsigned int candidates[3] = {8u, 16u, 32u};
    cs::sparse::compressed whole;
    std::vector<cs::types::ptr_t> row_ptr;
    std::vector<cs::types::idx_t> col_idx;
    std::vector<::real::storage_t> values;
    csc::blocked_ell_tune_result tune = {0u, 0.0, 0u};
    const auto convert_start = std::chrono::steady_clock::now();

    cs::init(dst);
    cs::sparse::init(&whole);
    if (!build_whole_compressed(src, &whole, &row_ptr, &col_idx, &values)) return 0;
    if (!csc::choose_blocked_ell_block_size(&whole, candidates, 3u, &tune)) return 0;

    const unsigned long target_block_rows = (cfg.rows_per_part + tune.block_size - 1u) / tune.block_size;
    if (!csc::repack_sharded_compressed_to_blocked_ell(src, tune.block_size, target_block_rows, dst)) return 0;
    if (!cs::set_equal_shards(dst, cfg.shards)) return 0;

    meta->convert_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - convert_start).count();
    meta->tuned_block_size = tune.block_size;
    meta->tuned_fill_ratio = tune.fill_ratio;
    meta->tuned_padded_bytes = tune.padded_bytes;
    return 1;
}

static int build_sliced_ell_part(const cs::sparse::blocked_ell *src,
                                 unsigned int slice_rows,
                                 cs::sparse::sliced_ell *out) {
    std::vector<unsigned int> row_offsets;
    std::vector<unsigned int> slice_widths;
    const unsigned int rows = src != nullptr ? src->rows : 0u;
    const unsigned int block_size = src != nullptr ? src->block_size : 0u;
    const unsigned int ell_width_blocks = block_size != 0u ? cs::sparse::ell_width_blocks(src) : 0u;
    unsigned int row = 0u;

    if (src == nullptr || out == nullptr || block_size == 0u || slice_rows == 0u) return 0;
    cs::sparse::clear(out);
    cs::sparse::init(out, src->rows, src->cols, src->nnz);

    row_offsets.push_back(0u);
    while (row < rows) {
        const unsigned int row_end = std::min<unsigned int>(rows, row + slice_rows);
        unsigned int max_width = 0u;
        for (unsigned int local_row = row; local_row < row_end; ++local_row) {
            const unsigned int row_block = local_row / block_size;
            unsigned int row_nnz = 0u;
            for (unsigned int ell_col = 0u; ell_col < src->ell_cols; ++ell_col) {
                const unsigned int slot = ell_col / block_size;
                const unsigned int lane_in_block = ell_col % block_size;
                const unsigned int block_col = src->blockColIdx[(unsigned long) row_block * ell_width_blocks + slot];
                const unsigned int col = block_col != cs::sparse::blocked_ell_invalid_col
                    ? block_col * block_size + lane_in_block
                    : src->cols;
                const float value = __half2float(src->val[(unsigned long) local_row * src->ell_cols + ell_col]);
                if (col < src->cols && value != 0.0f) ++row_nnz;
            }
            max_width = std::max(max_width, row_nnz);
        }
        slice_widths.push_back(max_width);
        row_offsets.push_back(row_end);
        row = row_end;
    }

    if (!cs::sparse::allocate(out,
                              (unsigned int) slice_widths.size(),
                              row_offsets.data(),
                              slice_widths.data())) {
        cs::sparse::clear(out);
        return 0;
    }

    for (unsigned int slice = 0u; slice < out->slice_count; ++slice) {
        const unsigned int row_begin = out->slice_row_offsets[slice];
        const unsigned int row_end = out->slice_row_offsets[slice + 1u];
        const unsigned int width = out->slice_widths[slice];
        const std::size_t slice_base = cs::sparse::slice_slot_base(out, slice);
        for (unsigned int local_row = row_begin; local_row < row_end; ++local_row) {
            const unsigned int row_block = local_row / block_size;
            const std::size_t dst_row_base = slice_base + (std::size_t) (local_row - row_begin) * (std::size_t) width;
            unsigned int dst_slot = 0u;
            for (unsigned int ell_col = 0u; ell_col < src->ell_cols; ++ell_col) {
                const unsigned int slot = ell_col / block_size;
                const unsigned int lane_in_block = ell_col % block_size;
                const unsigned int block_col = src->blockColIdx[(unsigned long) row_block * ell_width_blocks + slot];
                const unsigned int col = block_col != cs::sparse::blocked_ell_invalid_col
                    ? block_col * block_size + lane_in_block
                    : src->cols;
                const __half value = src->val[(unsigned long) local_row * src->ell_cols + ell_col];
                if (col >= src->cols || __half2float(value) == 0.0f) continue;
                out->col_idx[dst_row_base + dst_slot] = col;
                out->val[dst_row_base + dst_slot] = value;
                ++dst_slot;
            }
        }
    }

    return 1;
}

static int build_sliced_ell_matrix(const config &cfg,
                                   const cs::sharded<cs::sparse::compressed> *src,
                                   cs::sharded<cs::sparse::sliced_ell> *dst,
                                   result *meta) {
    cs::sharded<cs::sparse::blocked_ell> blocked;
    const auto convert_start = std::chrono::steady_clock::now();

    cs::init(dst);
    cs::init(&blocked);
    if (!build_blocked_ell_matrix(cfg, src, &blocked, meta)) {
        cs::clear(&blocked);
        return 0;
    }

    for (unsigned long part = 0; part < blocked.num_partitions; ++part) {
        auto *sliced = new cs::sparse::sliced_ell;
        cs::sparse::init(sliced);
        if (!build_sliced_ell_part(blocked.parts[part], 32u, sliced)) {
            cs::sparse::clear(sliced);
            delete sliced;
            cs::clear(&blocked);
            cs::clear(dst);
            return 0;
        }
        if (!cs::append_partition(dst, sliced)) {
            cs::sparse::clear(sliced);
            delete sliced;
            cs::clear(&blocked);
            cs::clear(dst);
            return 0;
        }
    }
    if (!cs::set_equal_shards(dst, cfg.shards)) {
        cs::clear(&blocked);
        cs::clear(dst);
        return 0;
    }

    meta->convert_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - convert_start).count();
    cs::clear(&blocked);
    return 1;
}

template<typename MatrixT>
static unsigned int max_part_rows(const cs::sharded<MatrixT> *view) {
    unsigned long best = 0ul;
    for (unsigned long part = 0; part < view->num_partitions; ++part) best = std::max(best, view->partition_rows[part]);
    return (unsigned int) best;
}

template<typename MatrixT>
static unsigned int max_part_nnz(const cs::sharded<MatrixT> *view) {
    unsigned long best = 0ul;
    for (unsigned long part = 0; part < view->num_partitions; ++part) best = std::max(best, view->partition_nnz[part]);
    return (unsigned int) best;
}

static int bind_uploaded_part_view(csv::compressed_view *out,
                                   const cs::sharded<cs::sparse::compressed> *host,
                                   const csv::partition_record<cs::sparse::compressed> *record,
                                   unsigned long part_id) {
    if (out == nullptr || host == nullptr || record == nullptr || part_id >= host->num_partitions) return 0;
    out->rows = (unsigned int) host->partition_rows[part_id];
    out->cols = (unsigned int) host->cols;
    out->nnz = (unsigned int) host->partition_nnz[part_id];
    out->axis = (unsigned int) host->partition_aux[part_id];
    out->majorPtr = (unsigned int *) record->a0;
    out->minorIdx = (unsigned int *) record->a1;
    out->val = (__half *) record->a2;
    return 1;
}

static int bind_uploaded_part_view(csv::blocked_ell_view *out,
                                   const cs::sharded<cs::sparse::blocked_ell> *host,
                                   const csv::partition_record<cs::sparse::blocked_ell> *record,
                                   unsigned long part_id) {
    if (out == nullptr || host == nullptr || record == nullptr || part_id >= host->num_partitions) return 0;
    out->rows = (unsigned int) host->partition_rows[part_id];
    out->cols = (unsigned int) host->cols;
    out->nnz = (unsigned int) host->partition_nnz[part_id];
    out->block_size = cs::sparse::unpack_blocked_ell_block_size(host->partition_aux[part_id]);
    out->ell_cols = cs::sparse::unpack_blocked_ell_cols(host->partition_aux[part_id]);
    out->blockColIdx = (unsigned int *) record->a0;
    out->val = (__half *) record->a1;
    return 1;
}

static int bind_uploaded_part_view(csv::sliced_ell_view *out,
                                   const cs::sparse::sliced_ell *host,
                                   const csv::partition_record<cs::sparse::sliced_ell> *record) {
    const std::size_t offsets_bytes =
        host != nullptr ? (std::size_t) (host->slice_count + 1u) * sizeof(unsigned int) : 0u;
    const std::size_t widths_offset = csv::align_up_bytes(offsets_bytes, alignof(unsigned int));
    const std::size_t widths_bytes =
        host != nullptr ? (std::size_t) host->slice_count * sizeof(unsigned int) : 0u;
    const std::size_t slot_offsets_offset = csv::align_up_bytes(widths_offset + widths_bytes, alignof(unsigned int));
    if (out == nullptr || host == nullptr || record == nullptr) return 0;
    out->rows = host->rows;
    out->cols = host->cols;
    out->nnz = host->nnz;
    out->slice_count = host->slice_count;
    out->slice_rows = cs::sparse::uniform_slice_rows(host);
    out->slice_row_offsets = (unsigned int *) record->a0;
    out->slice_widths = (unsigned int *) record->a1;
    out->slice_slot_offsets = host->slice_count != 0u
        ? (unsigned int *) ((char *) record->storage + slot_offsets_offset)
        : nullptr;
    out->col_idx = (unsigned int *) record->a2;
    out->val = (__half *) record->a3;
    return 1;
}

static int bind_uploaded_part_view(csv::sliced_ell_view *out,
                                   const cs::sharded<cs::sparse::sliced_ell> *host,
                                   const csv::partition_record<cs::sparse::sliced_ell> *record,
                                   unsigned long part_id) {
    if (out == nullptr || host == nullptr || part_id >= host->num_partitions) return 0;
    return bind_uploaded_part_view(out, host->parts[part_id], record);
}

static int preprocess_part_accumulating(csv::blocked_ell_view *part,
                                        cspre::preprocess_workspace *workspace,
                                        const cspre::cell_filter_params &filter,
                                        float target_sum) {
    cspre::cell_metrics_view cell{};
    if (!cspre::compute_cell_metrics(part, workspace, &filter, &cell)) return 0;
    if (!cspre::normalize_log1p_inplace(part, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
    return cspre::accumulate_gene_metrics(part, workspace, cell.keep_cells, nullptr);
}

static int preprocess_part_accumulating(csv::sliced_ell_view *part,
                                        cspre::preprocess_workspace *workspace,
                                        const cspre::cell_filter_params &filter,
                                        float target_sum) {
    cspre::cell_metrics_view cell{};
    if (!cspre::compute_cell_metrics(part, workspace, &filter, &cell)) return 0;
    if (!cspre::normalize_log1p_inplace(part, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
    return cspre::accumulate_gene_metrics(part, workspace, cell.keep_cells, nullptr);
}

static int preprocess_part_accumulating(csv::compressed_view *part,
                                        cspre::preprocess_workspace *workspace,
                                        const cspre::cell_filter_params &filter,
                                        float target_sum) {
    cspre::cell_metrics_view cell{};
    if (!cspre::compute_cell_metrics_compressed_fallback(part, workspace, &filter, &cell)) return 0;
    if (!cspre::normalize_log1p_compressed_fallback_inplace(part, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
    return cspre::accumulate_gene_metrics_compressed_fallback(part, workspace, cell.keep_cells, nullptr);
}

template<typename MatrixT, typename ViewT>
static int run_preprocess_for_matrix(const config &cfg,
                                     const cs::sharded<MatrixT> *matrix,
                                     result *out) {
    csv::sharded_device<MatrixT> device_state;
    cspre::preprocess_workspace workspace;
    std::vector<unsigned char> gene_flags(cfg.cols, 0u);
    std::vector<float> host_gene_sum(cfg.cols, 0.0f);
    std::vector<unsigned char> host_keep_genes(cfg.cols, 0u);
    float kept_cells = 0.0f;
    const cspre::cell_filter_params cell_filter = {
        cfg.min_counts,
        cfg.min_genes,
        cfg.max_mito_fraction
    };
    const cspre::gene_filter_params gene_filter = {
        cfg.min_gene_sum,
        cfg.min_gene_detected,
        cfg.min_gene_variance
    };

    csv::init(&device_state);
    cspre::init(&workspace);

    if (!csv::reserve(&device_state, matrix->num_partitions)) goto done;
    if (!cspre::setup(&workspace, (int) cfg.device, (cudaStream_t) 0)) goto done;
    if (!cspre::reserve(&workspace, max_part_rows(matrix), (unsigned int) matrix->cols, max_part_nnz(matrix))) goto done;

    for (unsigned int gene = 0; gene < cfg.cols; ++gene) {
        if (gene < 64u || (gene % 97u) == 0u) gene_flags[gene] = (unsigned char) cspre::gene_flag_mito;
    }
    if (!cspre::upload_gene_flags(&workspace, cfg.cols, gene_flags.data())) goto done;
    if (!check_cuda(cudaStreamSynchronize(workspace.stream), "cudaStreamSynchronize(upload_gene_flags)")) goto done;

    for (unsigned int iter = 0; iter < cfg.repeats; ++iter) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!cspre::zero_gene_metrics(&workspace, cfg.cols)) goto done;
        for (unsigned long part = 0; part < matrix->num_partitions; ++part) {
            ViewT part_view;
            if (!check_cuda(csv::upload_partition(&device_state, matrix, part, (int) cfg.device), "upload_partition")) goto done;
            if (!bind_uploaded_part_view(&part_view, matrix, device_state.parts + part, part)) goto done;
            if (!preprocess_part_accumulating(&part_view, &workspace, cell_filter, cfg.target_sum)) goto done;
            if (!check_cuda(cudaStreamSynchronize(workspace.stream), "cudaStreamSynchronize(part)")) goto done;
            if (!check_cuda(csv::release_partition(&device_state, part), "release_partition")) goto done;
        }
        out->part_loop_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const auto t1 = std::chrono::steady_clock::now();
        if (!cspre::build_gene_filter_mask(&workspace, (unsigned int) matrix->cols, &gene_filter, nullptr)) goto done;
        if (!check_cuda(cudaStreamSynchronize(workspace.stream), "cudaStreamSynchronize(gene_mask)")) goto done;
        out->gene_mask_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    }

    if (!check_cuda(cudaMemcpy(host_gene_sum.data(),
                               workspace.gene_sum,
                               (std::size_t) cfg.cols * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy(host_gene_sum)")) goto done;
    if (!check_cuda(cudaMemcpy(host_keep_genes.data(),
                               workspace.keep_genes,
                               (std::size_t) cfg.cols * sizeof(unsigned char),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy(host_keep_genes)")) goto done;
    if (!check_cuda(cudaMemcpy(&kept_cells,
                               workspace.active_rows,
                               sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy(kept_cells)")) goto done;

    out->rows = matrix->rows;
    out->cols = (unsigned int) matrix->cols;
    out->nnz = matrix->nnz;
    out->partitions = matrix->num_partitions;
    out->shards = matrix->num_shards;
    out->kept_cells = kept_cells;
    out->kept_genes = 0ul;
    out->gene_sum_checksum = 0.0;
    for (unsigned int gene = 0; gene < cfg.cols; ++gene) {
        out->kept_genes += host_keep_genes[gene] != 0u;
        out->gene_sum_checksum += (double) host_gene_sum[gene];
    }

    out->part_loop_ms /= (double) cfg.repeats;
    out->gene_mask_ms /= (double) cfg.repeats;
    cspre::clear(&workspace);
    csv::clear(&device_state);
    return 1;

done:
    cspre::clear(&workspace);
    csv::clear(&device_state);
    return 0;
}

static void print_result(const result &entry) {
    std::printf("preprocess_compare: format=%s rows=%lu cols=%u nnz=%lu parts=%lu shards=%lu convert_ms=%.3f part_loop_ms=%.3f gene_mask_ms=%.3f kept_cells=%.1f kept_genes=%lu checksum=%.6f",
                entry.label,
                entry.rows,
                entry.cols,
                entry.nnz,
                entry.partitions,
                entry.shards,
                entry.convert_ms,
                entry.part_loop_ms,
                entry.gene_mask_ms,
                entry.kept_cells,
                entry.kept_genes,
                entry.gene_sum_checksum);
    if (std::strcmp(entry.label, "blocked_ell") == 0 || std::strcmp(entry.label, "sliced_ell") == 0) {
        std::printf(" block_size=%u fill_ratio=%.6f padded_bytes=%zu",
                    entry.tuned_block_size,
                    entry.tuned_fill_ratio,
                    entry.tuned_padded_bytes);
    }
    std::printf("\n");
}

static void print_delta(const result &compressed, const result &blocked_ell) {
    const double compressed_total = compressed.part_loop_ms + compressed.gene_mask_ms;
    const double blocked_total = blocked_ell.part_loop_ms + blocked_ell.gene_mask_ms;
    std::printf("preprocess_compare_delta: blocked_minus_compressed_runtime_ms=%.3f blocked_minus_compressed_total_ms=%.3f checksum_delta=%.6f kept_genes_delta=%ld kept_cells_delta=%.1f\n",
                blocked_total - compressed_total,
                (blocked_ell.convert_ms + blocked_total) - (compressed.convert_ms + compressed_total),
                blocked_ell.gene_sum_checksum - compressed.gene_sum_checksum,
                (long) blocked_ell.kept_genes - (long) compressed.kept_genes,
                blocked_ell.kept_cells - compressed.kept_cells);
}

static void print_delta(const result &baseline, const result &candidate, const char *label) {
    const double baseline_total = baseline.part_loop_ms + baseline.gene_mask_ms;
    const double candidate_total = candidate.part_loop_ms + candidate.gene_mask_ms;
    std::printf("preprocess_compare_delta: label=%s runtime_ms=%.3f total_ms=%.3f checksum_delta=%.6f kept_genes_delta=%ld kept_cells_delta=%.1f\n",
                label,
                candidate_total - baseline_total,
                (candidate.convert_ms + candidate_total) - (baseline.convert_ms + baseline_total),
                candidate.gene_sum_checksum - baseline.gene_sum_checksum,
                (long) candidate.kept_genes - (long) baseline.kept_genes,
                candidate.kept_cells - baseline.kept_cells);
}

} // namespace

int main(int argc, char **argv) {
    cspre::bench::benchmark_mutex_guard benchmark_mutex("cellShardPreprocessFormatCompareBench");
    config cfg;
    int device_count = 0;
    cs::sharded<cs::sparse::compressed> compressed;
    cs::sharded<cs::sparse::blocked_ell> blocked_ell;
    cs::sharded<cs::sparse::sliced_ell> sliced_ell;
    result compressed_result = {"compressed"};
    result blocked_ell_result = {"blocked_ell"};
    result sliced_ell_result = {"sliced_ell"};

    cs::init(&compressed);
    cs::init(&blocked_ell);
    cs::init(&sliced_ell);

    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 2;
    }
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) return 1;
    if (device_count <= 0 || cfg.device >= (unsigned int) device_count) {
        std::fprintf(stderr, "requested CUDA device %u is unavailable\n", cfg.device);
        return 1;
    }

    std::printf("config: mode=%s device=%u parts=%u rows_per_part=%u cols=%u avg_nnz_row=%u shards=%u repeats=%u target_sum=%.1f min_counts=%.1f min_genes=%u max_mito_fraction=%.3f\n",
                cfg.mode == format_mode::compressed ? "compressed"
                    : (cfg.mode == format_mode::blocked_ell ? "blocked_ell"
                    : (cfg.mode == format_mode::sliced_ell ? "sliced_ell"
                    : (cfg.mode == format_mode::both ? "both" : "all"))),
                cfg.device,
                cfg.parts,
                cfg.rows_per_part,
                cfg.cols,
                cfg.avg_nnz_per_row,
                cfg.shards,
                cfg.repeats,
                cfg.target_sum,
                cfg.min_counts,
                cfg.min_genes,
                cfg.max_mito_fraction);

    if (!build_compressed_matrix(cfg, &compressed)) {
        cs::clear(&compressed);
        return 1;
    }

    if (cfg.mode == format_mode::compressed || cfg.mode == format_mode::both || cfg.mode == format_mode::all) {
        if (!run_preprocess_for_matrix<cs::sparse::compressed, csv::compressed_view>(cfg, &compressed, &compressed_result)) {
            cs::clear(&compressed);
            return 1;
        }
        print_result(compressed_result);
    }

    if (cfg.mode == format_mode::blocked_ell || cfg.mode == format_mode::both || cfg.mode == format_mode::all) {
        if (!build_blocked_ell_matrix(cfg, &compressed, &blocked_ell, &blocked_ell_result)) {
            cs::clear(&blocked_ell);
            cs::clear(&compressed);
            return 1;
        }
        if (!run_preprocess_for_matrix<cs::sparse::blocked_ell, csv::blocked_ell_view>(cfg, &blocked_ell, &blocked_ell_result)) {
            cs::clear(&blocked_ell);
            cs::clear(&compressed);
            return 1;
        }
        print_result(blocked_ell_result);
    }

    if (cfg.mode == format_mode::sliced_ell || cfg.mode == format_mode::all) {
        if (!build_sliced_ell_matrix(cfg, &compressed, &sliced_ell, &sliced_ell_result)) {
            cs::clear(&sliced_ell);
            cs::clear(&blocked_ell);
            cs::clear(&compressed);
            return 1;
        }
        if (!run_preprocess_for_matrix<cs::sparse::sliced_ell, csv::sliced_ell_view>(cfg, &sliced_ell, &sliced_ell_result)) {
            cs::clear(&sliced_ell);
            cs::clear(&blocked_ell);
            cs::clear(&compressed);
            return 1;
        }
        print_result(sliced_ell_result);
    }

    if (cfg.mode == format_mode::both || cfg.mode == format_mode::all) {
        print_delta(compressed_result, blocked_ell_result);
    }
    if (cfg.mode == format_mode::all) {
        print_delta(blocked_ell_result, sliced_ell_result, "sliced_minus_blocked");
    }

    cs::clear(&sliced_ell);
    cs::clear(&blocked_ell);
    cs::clear(&compressed);
    return 0;
}
