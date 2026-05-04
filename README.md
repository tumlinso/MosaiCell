# CellShardPreprocess

CellShardPreprocess is the biology-facing single-cell preprocessing backbone split out of
Cellerator. Its native surface is CellShard-layout-first and pointer-first:
Blocked-ELL and Sliced-ELL are peer first-class preprocessing layouts, while
compressed / CSR is kept as an explicit fallback and validation path.
Numerical math over CellShard matrices belongs in Cellerator compute; this
project owns the preprocessing policy, runtime API, workbench, raw-count state,
and biological feature semantics that call that math layer.

QC feature masks are modality-agnostic. Each feature carries a `uint32_t`
`feature_group_mask` with up to 32 groups. CellShardPreprocess owns the
biological rule compilation and naming policy, including the built-in `mt`,
`ribo`, and `hb` defaults plus user-defined prefix, exact-id/name,
feature-type, modality, or explicit-mask rules. The sparse row traversal,
row/feature mask application, and row-major group count/percentage reductions
are delegated to CellShard's generic `runtime/mask_groups.cuh` primitive.

CellShardPreprocess's native C++ contract does not add vector-based GPU or
public preprocessing APIs. Generic sparse masking belongs in CellShard runtime;
this package keeps preprocessing thresholds, raw-count validation,
normalization/log1p, gene metrics, and persisted QC naming.

The native preprocessing ABI also has a fleet workspace for row-sharded
Blocked-ELL and Sliced-ELL assays. Local multi-GPU execution keeps row-owned cell QC outputs on
their owning devices, reduces feature summaries onto a leader device, and uses
CelleratorDist for peer access, NCCL-backed local reductions when available,
and an optional ranked NCCL communicator for extension beyond a single host.

## Build

CellShardPreprocess builds as a standalone CUDA/C++ project. It resolves CellShard in this
order:

1. `-DCELLSHARD_PREPROCESS_CELLSHARD_SOURCE_DIR=/path/to/CellShard`
2. a sibling checkout at `../CellShard`
3. a vendored checkout at `extern/CellShard`
4. `FetchContent` from `git@github.com:tumlinso/CellShard.git`

```bash
cmake -S . -B build
cmake --build build -j 4
```

The small native preprocessing target is `CellShardPreprocess::preprocess`. The composed
runtime target is `CellShardPreprocess::runtime`, which Cellerator consumes.
The native C++ namespace is `cellshard_preprocess`; callers may include
`CellShardPreprocess/aliases.hh` or any public runtime/preprocess header and use
`namespace cspre = ::cellshard_preprocess;` for shorter call sites.

Primary source layout:

- `include/CellShardPreprocess/preprocess.cuh`: pointer-first preprocessing ABI for Blocked-ELL, Sliced-ELL, and CSR fallback
- `src/preprocess.cu`: preprocessing normalization/log1p, feature metrics, and fleet reduction glue; QC group row reductions call CellShard runtime masking
- `include/CellShardPreprocess/runtime.hh`: validation, adapter staging, and runtime ABI
- `src/runtime.cc`: raw-count policy, QC mask rule compilation, and staging policy
- `src/apps/workbench_main.cc`: optional `cellShardPreprocessWorkbench` entrypoint
- `bench/`: native preprocessing benchmarks with serialized GPU benchmark locking
