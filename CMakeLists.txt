cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(EXISTS "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc")
        set(CMAKE_CUDA_COMPILER
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc"
            CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    if(DEFINED ENV{CUDAHOSTCXX} AND NOT "$ENV{CUDAHOSTCXX}" STREQUAL "")
        set(CMAKE_CUDA_HOST_COMPILER "$ENV{CUDAHOSTCXX}" CACHE FILEPATH "CUDA host compiler" FORCE)
    elseif(EXISTS "/usr/bin/g++-12")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "CUDA host compiler" FORCE)
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(MosaiCell LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig QUIET)

option(MOSAICELL_BUILD_WORKBENCH "Build the optional MosaiCell ncurses workbench stub." ON)
option(MOSAICELL_BUILD_TESTS "Build MosaiCell native tests." ON)
option(MOSAICELL_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for MosaiCell kernels." OFF)
set(MOSAICELL_CELLSHARD_SOURCE_DIR "" CACHE PATH
    "Optional path to a CellShard source checkout. If unset, MosaiCell checks sibling/vendor paths, then FetchContent.")

if(NOT TARGET CellShard::runtime)
    set(CELLSHARD_ENABLE_CUDA ON CACHE BOOL "MosaiCell requires the CellShard CUDA runtime." FORCE)
    set(CELLSHARD_BUILD_EXPORT ON CACHE BOOL "Build CellShard export helpers for MosaiCell." FORCE)
    set(CELLSHARD_BUILD_APPS OFF CACHE BOOL "Do not build CellShard apps from standalone MosaiCell." FORCE)
    set(CELLSHARD_BUILD_TESTS OFF CACHE BOOL "Do not build nested CellShard tests from MosaiCell." FORCE)
    if(MOSAICELL_CELLSHARD_SOURCE_DIR AND EXISTS "${MOSAICELL_CELLSHARD_SOURCE_DIR}/CMakeLists.txt")
        add_subdirectory("${MOSAICELL_CELLSHARD_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/CellShard" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../CellShard/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../CellShard" "${CMAKE_CURRENT_BINARY_DIR}/CellShard" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/extern/CellShard/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/extern/CellShard" "${CMAKE_CURRENT_BINARY_DIR}/CellShard" EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        FetchContent_Declare(
            CellShard
            GIT_REPOSITORY git@github.com:tumlinso/CellShard.git
            GIT_TAG main
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(CellShard)
    endif()
endif()

add_library(mosaicell_preprocess STATIC
    src/preprocess.cu
)
add_library(MosaiCell::preprocess ALIAS mosaicell_preprocess)
target_include_directories(mosaicell_preprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mosaicell_preprocess PUBLIC
    CellShard::runtime
    CUDA::cudart
)
target_compile_features(mosaicell_preprocess PUBLIC cxx_std_17)
set_target_properties(mosaicell_preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(mosaicell_preprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -Xptxas=-O3,-dlcm=ca>
)
if(MOSAICELL_ENABLE_CUDA_LINEINFO)
    target_compile_options(mosaicell_preprocess PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

add_library(mosaicell_runtime STATIC
    src/runtime.cc
)
add_library(MosaiCell::runtime ALIAS mosaicell_runtime)
target_include_directories(mosaicell_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mosaicell_runtime PUBLIC
    MosaiCell::preprocess
)
target_compile_features(mosaicell_runtime PUBLIC cxx_std_17)
set_target_properties(mosaicell_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(mosaicell_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
)

if(MOSAICELL_BUILD_WORKBENCH AND PkgConfig_FOUND)
    pkg_check_modules(MOSAICELL_NCURSESW QUIET ncursesw)
    if(MOSAICELL_NCURSESW_FOUND)
        add_executable(mosaiCellWorkbench
            src/apps/workbench_main.cc
        )
        target_include_directories(mosaiCellWorkbench PRIVATE
            ${MOSAICELL_NCURSESW_INCLUDE_DIRS}
        )
        target_compile_options(mosaiCellWorkbench PRIVATE
            ${MOSAICELL_NCURSESW_CFLAGS_OTHER}
        )
        target_link_directories(mosaiCellWorkbench PRIVATE
            ${MOSAICELL_NCURSESW_LIBRARY_DIRS}
        )
        target_link_libraries(mosaiCellWorkbench PRIVATE
            MosaiCell::runtime
            ${MOSAICELL_NCURSESW_LIBRARIES}
        )
        set_target_properties(mosaiCellWorkbench PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
        )
    endif()
endif()

if(MOSAICELL_BUILD_TESTS)
    add_executable(mosaicellPreprocessRuntimeTest
        tests/preprocess_runtime_test.cu
    )
    target_link_libraries(mosaicellPreprocessRuntimeTest PRIVATE MosaiCell::runtime CUDA::cudart)
    set_target_properties(mosaicellPreprocessRuntimeTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_executable(mosaicellQCMetricsEquivalenceTest
        tests/qc_metrics_equivalence_test.cu
    )
    target_link_libraries(mosaicellQCMetricsEquivalenceTest PRIVATE MosaiCell::runtime CUDA::cudart)
    set_target_properties(mosaicellQCMetricsEquivalenceTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_executable(mosaicellAdapterStagingTest
        tests/adapter_staging_test.cc
    )
    target_link_libraries(mosaicellAdapterStagingTest PRIVATE MosaiCell::runtime)
    set_target_properties(mosaicellAdapterStagingTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )

    add_executable(mosaicellDoublePreprocessRejectionTest
        tests/double_preprocess_rejection_test.cc
    )
    target_link_libraries(mosaicellDoublePreprocessRejectionTest PRIVATE MosaiCell::runtime)
    set_target_properties(mosaicellDoublePreprocessRejectionTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
endif()
