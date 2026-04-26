# MosaiCell

MosaiCell is the accelerated single-cell preprocessing backbone split out of
Cellerator. Its native surface is CellShard-layout-first and pointer-first:
steady-state matrix values are fp16, QC accumulators are fp32, and Blocked-ELL is
the preferred persisted and hot-path sparse layout.

The compatibility-facing Cellerator workbench API may still expose legacy C++
containers, but MosaiCell's native C++ contract does not add vector-based GPU or
public preprocessing APIs.

## Build

MosaiCell builds as a standalone CUDA/C++ project. It resolves CellShard in this
order:

1. `-DMOSAICELL_CELLSHARD_SOURCE_DIR=/path/to/CellShard`
2. a sibling checkout at `../CellShard`
3. a vendored checkout at `extern/CellShard`
4. `FetchContent` from `git@github.com:tumlinso/CellShard.git`

```bash
cmake -S . -B build
cmake --build build -j 4
```

The small native preprocessing target is `MosaiCell::preprocess`. The composed
runtime target is `MosaiCell::runtime`, which Cellerator consumes.

Primary source layout:

- `include/MosaiCell/preprocess.cuh`: pointer-first Blocked-ELL preprocessing ABI
- `src/preprocess.cu`: CUDA QC, normalize-log1p, and gene metric kernels
- `include/MosaiCell/runtime.hh`: validation, adapter staging, and runtime ABI
- `src/runtime.cc`: raw-count policy, mito feature marking, and staging policy
- `src/apps/workbench_main.cc`: optional `mosaiCellWorkbench` entrypoint

The legacy Cellerator compatibility layer still lives in Cellerator while the
native backbone moves into these smaller MosaiCell targets first.
