#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "benchmark_mutex.hh"
#include <CellShard/CellShard.hh>
#include <CellShardPreprocess/preprocess.cuh>

namespace {

namespace cs = ::cellshard;
namespace csv = ::cellshard::device;
namespace cspre = ::cellshard_preprocess;

struct config {
    unsigned int parts = 32;
    unsigned int rows_per_partition = 32768;
    unsigned int cols = 32768;
    unsigned int avg_nnz_per_row = 128;
    unsigned int shards = 8;
    unsigned int repeats = 1;
    unsigned int seed = 7;
    float target_sum = 10000.0f;
    float min_counts = 500.0f;
    unsigned int min_genes = 200;
    float max_mito_fraction = 0.2f;
    float min_gene_sum = 1.0f;
    float min_gene_detected = 5.0f;
    float min_gene_variance = 0.01f;
};

struct scoped_nvtx_range {
    explicit scoped_nvtx_range(const char *label) { nvtxRangePushA(label); }
    ~scoped_nvtx_range() { nvtxRangePop(); }
};

static int check_cuda(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --parts N                Physical parts. Default: 32\n"
                 "  --rows-per-partition N   Rows per partition. Default: 32768\n"
                 "  --cols N                 Number of genes. Default: 32768\n"
                 "  --avg-nnz-row N          Average nnz per row. Default: 128\n"
                 "  --shards N               Logical shard count. Default: 8\n"
                 "  --repeats N              Full preprocess repeats. Default: 1\n"
                 "  --seed N                 RNG seed. Default: 7\n"
                 "  --target-sum F           Library size target. Default: 10000\n"
                 "  --min-counts F           Cell filter minimum counts. Default: 500\n"
                 "  --min-genes N            Cell filter minimum genes. Default: 200\n"
                 "  --max-mito-fraction F    Cell filter max mito fraction. Default: 0.2\n"
                 "  --min-gene-sum F         Gene filter minimum sum. Default: 1\n"
                 "  --min-gene-detected F    Gene filter minimum detected cells. Default: 5\n"
                 "  --min-gene-variance F    Gene filter minimum variance. Default: 0.01\n",
                 argv0);
}

static int parse_u32(const char *text, unsigned int *value) {
    char *end = 0;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != 0 || parsed > 0xfffffffful) return 0;
    *value = (unsigned int) parsed;
    return 1;
}

static int parse_f32(const char *text, float *value) {
    char *end = 0;
    float parsed = std::strtof(text, &end);
    if (text == end || *end != 0) return 0;
    *value = parsed;
    return 1;
}

static int parse_args(int argc, char **argv, config *cfg) {
    int i = 1;
    while (i < argc) {
        if (std::strcmp(argv[i], "--parts") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->parts)) return 0;
        } else if (std::strcmp(argv[i], "--rows-per-partition") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->rows_per_partition)) return 0;
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
        ++i;
    }
    return cfg->parts != 0 && cfg->rows_per_partition != 0 && cfg->cols != 0 && cfg->shards != 0;
}

static void build_row_counts(std::vector<unsigned int> *counts,
                             unsigned int rows,
                             unsigned int cols,
                             unsigned int avg_nnz_per_row,
                             std::mt19937 *rng) {
    std::vector<float> weights(rows, 0.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    unsigned long long assigned = 0;
    const unsigned long long target_nnz = (unsigned long long) rows * (unsigned long long) avg_nnz_per_row;
    unsigned int i = 0;

    counts->assign(rows, 0u);
    for (i = 0; i < rows; ++i) {
        const float heavy = unit(*rng) < 0.15f ? 6.0f + unit(*rng) * 4.0f : 0.5f + unit(*rng) * 1.5f;
        weights[i] = heavy;
    }

    {
        float weight_sum = 0.0f;
        for (float w : weights) weight_sum += w;
        for (i = 0; i < rows; ++i) {
            unsigned int count = (unsigned int) ((target_nnz * (unsigned long long) (weights[i] * 1024.0f / weight_sum)) >> 10);
            if (count > cols) count = cols;
            (*counts)[i] = count;
            assigned += count;
        }
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
        if ((*counts)[row] != 0) {
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
    unsigned long long nnz = 0;
    unsigned int row = 0;
    auto *part = new cs::sparse::compressed;

    build_row_counts(&row_counts, rows, cols, avg_nnz_per_row, &rng);
    for (row = 0; row < rows; ++row) nnz += row_counts[row];

    cs::sparse::init(part, rows, cols, (cs::types::nnz_t) nnz, cs::sparse::compressed_by_row);
    if (!cs::sparse::allocate(part)) {
        delete part;
        return 0;
    }

    part->majorPtr[0] = 0;
    for (row = 0; row < rows; ++row) {
        const unsigned int count = row_counts[row];
        const unsigned int begin = part->majorPtr[row];
        const unsigned int max_start = cols > count ? cols - count : 0;
        const unsigned int start = max_start != 0 ? (rng() % (max_start + 1u)) : 0u;
        unsigned int j = 0;

        part->majorPtr[row + 1] = begin + count;
        for (j = 0; j < count; ++j) {
            part->minorIdx[begin + j] = start + j;
            part->val[begin + j] = __float2half((float) (1u + ((row + j) % 23u)));
        }
    }
    return part;
}

static int pin_loaded_parts(cs::sharded<cs::sparse::compressed> *view) {
    unsigned long part = 0;
    for (part = 0; part < view->num_partitions; ++part) {
        if (view->parts[part] == 0) return 0;
        if (!cs::sparse::pin(view->parts[part])) return 0;
    }
    return 1;
}

static void unpin_loaded_parts(cs::sharded<cs::sparse::compressed> *view) {
    for (unsigned long part = 0; part < view->num_partitions; ++part) {
        if (view->parts[part] != 0) cs::sparse::unpin(view->parts[part]);
    }
}

static int build_host_matrix(const config &cfg, cs::sharded<cs::sparse::compressed> *built) {
    cs::init(built);
    for (unsigned int part = 0; part < cfg.parts; ++part) {
        cs::sparse::compressed *matrix_part = make_compressed_part(cfg.rows_per_partition,
                                                                   cfg.cols,
                                                                   cfg.avg_nnz_per_row,
                                                                   cfg.seed + 17u * part);
        if (matrix_part == 0) return 0;
        if (!cs::append_partition(built, matrix_part)) return 0;
    }
    if (!cs::set_equal_shards(built, cfg.shards)) return 0;
    return pin_loaded_parts(built);
}

static int max_part_rows(const cs::sharded<cs::sparse::compressed> *view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view->num_partitions; ++part) best = std::max(best, view->partition_rows[part]);
    return (int) best;
}

static int max_part_nnz(const cs::sharded<cs::sparse::compressed> *view) {
    unsigned long best = 0;
    for (unsigned long part = 0; part < view->num_partitions; ++part) best = std::max(best, view->partition_nnz[part]);
    return (int) best;
}

static int run_preprocess_benchmark(const config &cfg) {
    cs::sharded<cs::sparse::compressed> built;
    csv::sharded_device<cs::sparse::compressed> device_state;
    cspre::preprocess_workspace workspace;
    std::vector<unsigned char> gene_flags;
    std::vector<float> host_gene_sum;
    std::vector<unsigned char> host_keep_genes;
    double qc_norm_ms = 0.0;
    double gene_ms = 0.0;
    int kept_genes = 0;
    double gene_sum_checksum = 0.0;
    int ok = 0;

    cs::init(&built);
    csv::init(&device_state);
    cspre::init(&workspace);

    if (!build_host_matrix(cfg, &built)) goto done;
    if (!csv::reserve(&device_state, built.num_partitions)) goto done;
    if (!cspre::setup(&workspace, 0, (cudaStream_t) 0)) goto done;
    if (!cspre::reserve(&workspace, (unsigned int) max_part_rows(&built), cfg.cols, (unsigned int) max_part_nnz(&built))) goto done;

    gene_flags.assign(cfg.cols, 0u);
    for (unsigned int gene = 0; gene < cfg.cols; ++gene) {
        if (gene < 64u || (gene % 97u) == 0u) gene_flags[gene] = (unsigned char) cspre::gene_flag_mito;
    }
    if (!cspre::upload_gene_flags(&workspace, cfg.cols, gene_flags.data())) goto done;
    if (!check_cuda(cudaStreamSynchronize(workspace.stream), "sync upload_gene_flags")) goto done;

    {
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

        for (unsigned int iter = 0; iter < cfg.repeats; ++iter) {
            auto t_qc0 = std::chrono::steady_clock::now();
            if (!cspre::zero_gene_metrics(&workspace, cfg.cols)) goto done;

            {
                scoped_nvtx_range range("scrna_qc_normalize_loop");
                for (unsigned long part = 0; part < built.num_partitions; ++part) {
                    csv::compressed_view part_view{};
                    cspre::cell_metrics_view cell{};
                    if (!check_cuda(csv::upload_partition(&device_state, &built, part, 0), "upload_partition")) goto done;
                    part_view.rows = (unsigned int) built.partition_rows[part];
                    part_view.cols = (unsigned int) built.cols;
                    part_view.nnz = (unsigned int) built.partition_nnz[part];
                    part_view.axis = (unsigned int) built.partition_aux[part];
                    part_view.majorPtr = (unsigned int *) device_state.parts[part].a0;
                    part_view.minorIdx = (unsigned int *) device_state.parts[part].a1;
                    part_view.val = (__half *) device_state.parts[part].a2;
                    if (!cspre::compute_cell_metrics_compressed_fallback(&part_view, &workspace, &cell_filter, &cell)) goto done;
                    if (!cspre::normalize_log1p_compressed_fallback_inplace(&part_view, &workspace, cell.total_counts, cell.keep_cells, cfg.target_sum)) goto done;
                    if (!cspre::accumulate_gene_metrics_compressed_fallback(&part_view, &workspace, cell.keep_cells, nullptr)) goto done;
                    if (!check_cuda(cudaStreamSynchronize(workspace.stream), "sync part preprocess")) goto done;
                    if (!check_cuda(csv::release_partition(&device_state, part), "release_partition")) goto done;
                }
            }

            {
                auto t_qc1 = std::chrono::steady_clock::now();
                qc_norm_ms += std::chrono::duration<double, std::milli>(t_qc1 - t_qc0).count();
            }

            {
                auto t_gene0 = std::chrono::steady_clock::now();
                scoped_nvtx_range range("scrna_gene_reduce_loop");
                if (!cspre::build_gene_filter_mask(&workspace, cfg.cols, &gene_filter, nullptr)) goto done;
                if (!check_cuda(cudaStreamSynchronize(workspace.stream), "sync gene mask")) goto done;
                {
                    auto t_gene1 = std::chrono::steady_clock::now();
                    gene_ms += std::chrono::duration<double, std::milli>(t_gene1 - t_gene0).count();
                }
            }
        }
    }

    host_gene_sum.resize(cfg.cols);
    host_keep_genes.resize(cfg.cols);
    if (!check_cuda(cudaMemcpy(host_gene_sum.data(),
                               workspace.gene_sum,
                               (std::size_t) cfg.cols * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy host gene sum")) goto done;
    if (!check_cuda(cudaMemcpy(host_keep_genes.data(),
                               workspace.keep_genes,
                               (std::size_t) cfg.cols * sizeof(unsigned char),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy host keep genes")) goto done;

    kept_genes = 0;
    gene_sum_checksum = 0.0;
    for (unsigned int gene = 0; gene < cfg.cols; ++gene) {
        kept_genes += host_keep_genes[gene] != 0;
        gene_sum_checksum += (double) host_gene_sum[gene];
    }

    std::printf("scrna_preprocess: devices=%u parts=%u rows=%lu cols=%u nnz=%lu repeats=%u qc_norm_ms=%.3f gene_ms=%.3f kept_genes=%d gene_sum_checksum=%.6f\n",
                1u,
                cfg.parts,
                built.rows,
                cfg.cols,
                built.nnz,
                cfg.repeats,
                qc_norm_ms / (double) cfg.repeats,
                gene_ms / (double) cfg.repeats,
                kept_genes,
                gene_sum_checksum);

    ok = 1;

done:
    cspre::clear(&workspace);
    csv::clear(&device_state);
    unpin_loaded_parts(&built);
    cs::clear(&built);
    return ok;
}

} // namespace

int main(int argc, char **argv) {
    cspre::bench::benchmark_mutex_guard benchmark_mutex("cellShardPreprocessScrnaBench");
    config cfg;

    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 2;
    }

    std::printf("config: parts=%u rows_per_partition=%u cols=%u avg_nnz_row=%u shards=%u repeats=%u target_sum=%.1f min_counts=%.1f min_genes=%u max_mito_fraction=%.3f\n",
                cfg.parts,
                cfg.rows_per_partition,
                cfg.cols,
                cfg.avg_nnz_per_row,
                cfg.shards,
                cfg.repeats,
                cfg.target_sum,
                cfg.min_counts,
                cfg.min_genes,
                cfg.max_mito_fraction);

    return run_preprocess_benchmark(cfg) ? 0 : 1;
}
