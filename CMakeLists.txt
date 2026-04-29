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

project(CellShardPreprocess LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CELLSHARD_PREPROCESS_VERSION 0.1.0)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig QUIET)

option(CELLSHARD_PREPROCESS_BUILD_WORKBENCH "Build the optional CellShardPreprocess ncurses workbench stub." ON)
option(CELLSHARD_PREPROCESS_BUILD_TESTS "Build CellShardPreprocess native tests." ON)
option(CELLSHARD_PREPROCESS_BUILD_BENCHMARKS "Build CellShardPreprocess native benchmarks." ON)
option(CELLSHARD_PREPROCESS_INSTALL_RULES "Install CellShardPreprocess targets and package config." ON)
option(CELLSHARD_PREPROCESS_ENABLE_CUDA_LINEINFO "Emit CUDA line mapping for CellShardPreprocess kernels." OFF)
set(CELLSHARD_PREPROCESS_CELLSHARD_SOURCE_DIR "" CACHE PATH
    "Optional path to a CellShard source checkout. If unset, CellShardPreprocess checks sibling/vendor paths, then FetchContent.")

if(NOT TARGET CellShard::runtime)
    set(CELLSHARD_ENABLE_CUDA ON CACHE BOOL "CellShardPreprocess requires the CellShard CUDA runtime." FORCE)
    set(CELLSHARD_BUILD_EXPORT ON CACHE BOOL "Build CellShard export helpers for CellShardPreprocess." FORCE)
    set(CELLSHARD_BUILD_APPS OFF CACHE BOOL "Do not build CellShard apps from standalone CellShardPreprocess." FORCE)
    set(CELLSHARD_BUILD_TESTS OFF CACHE BOOL "Do not build nested CellShard tests from CellShardPreprocess." FORCE)
    if(CELLSHARD_PREPROCESS_CELLSHARD_SOURCE_DIR AND EXISTS "${CELLSHARD_PREPROCESS_CELLSHARD_SOURCE_DIR}/CMakeLists.txt")
        add_subdirectory("${CELLSHARD_PREPROCESS_CELLSHARD_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/CellShard" EXCLUDE_FROM_ALL)
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

add_library(cellshard_preprocess_preprocess STATIC
    src/preprocess.cu
)
add_library(CellShardPreprocess::preprocess ALIAS cellshard_preprocess_preprocess)
set_target_properties(cellshard_preprocess_preprocess PROPERTIES EXPORT_NAME preprocess)
target_include_directories(cellshard_preprocess_preprocess PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cellshard_preprocess_preprocess PUBLIC
    CellShard::runtime
    CUDA::cudart
)
target_compile_features(cellshard_preprocess_preprocess PUBLIC cxx_std_17)
set_target_properties(cellshard_preprocess_preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cellshard_preprocess_preprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -Xptxas=-O3,-dlcm=ca>
)
if(CELLSHARD_PREPROCESS_ENABLE_CUDA_LINEINFO)
    target_compile_options(cellshard_preprocess_preprocess PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

add_library(cellshard_preprocess_runtime STATIC
    src/runtime.cc
)
add_library(CellShardPreprocess::runtime ALIAS cellshard_preprocess_runtime)
set_target_properties(cellshard_preprocess_runtime PROPERTIES EXPORT_NAME runtime)
target_include_directories(cellshard_preprocess_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cellshard_preprocess_runtime PUBLIC
    CellShardPreprocess::preprocess
)
target_compile_features(cellshard_preprocess_runtime PUBLIC cxx_std_17)
set_target_properties(cellshard_preprocess_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cellshard_preprocess_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -mtune=native>
)

if(CELLSHARD_PREPROCESS_BUILD_WORKBENCH AND PkgConfig_FOUND)
    pkg_check_modules(CELLSHARD_PREPROCESS_NCURSESW QUIET ncursesw)
    if(CELLSHARD_PREPROCESS_NCURSESW_FOUND)
        add_executable(cellShardPreprocessWorkbench
            src/apps/workbench_main.cc
        )
        target_include_directories(cellShardPreprocessWorkbench PRIVATE
            ${CELLSHARD_PREPROCESS_NCURSESW_INCLUDE_DIRS}
        )
        target_compile_options(cellShardPreprocessWorkbench PRIVATE
            ${CELLSHARD_PREPROCESS_NCURSESW_CFLAGS_OTHER}
        )
        target_link_directories(cellShardPreprocessWorkbench PRIVATE
            ${CELLSHARD_PREPROCESS_NCURSESW_LIBRARY_DIRS}
        )
        target_link_libraries(cellShardPreprocessWorkbench PRIVATE
            CellShardPreprocess::runtime
            ${CELLSHARD_PREPROCESS_NCURSESW_LIBRARIES}
        )
        set_target_properties(cellShardPreprocessWorkbench PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
        )
    endif()
endif()

if(CELLSHARD_PREPROCESS_INSTALL_RULES)
    install(TARGETS
        cellshard_preprocess_preprocess
        cellshard_preprocess_runtime
        EXPORT CellShardPreprocessTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
            PATTERN "*.hh"
            PATTERN "*.cuh"
    )

    install(EXPORT CellShardPreprocessTargets
        FILE CellShardPreprocessTargets.cmake
        NAMESPACE CellShardPreprocess::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellShardPreprocess
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/CellShardPreprocessConfigVersion.cmake
        VERSION ${CELLSHARD_PREPROCESS_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CellShardPreprocessConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CellShardPreprocessConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellShardPreprocess
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CellShardPreprocessConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/CellShardPreprocessConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellShardPreprocess
    )
endif()

if(CELLSHARD_PREPROCESS_BUILD_BENCHMARKS)
    add_executable(cellShardPreprocessScrnaBench
        bench/scrna_preprocess_bench.cu
    )
    target_link_libraries(cellShardPreprocessScrnaBench PRIVATE CellShardPreprocess::runtime CUDA::cudart)
    set_target_properties(cellShardPreprocessScrnaBench PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_executable(cellShardPreprocessFormatCompareBench
        bench/preprocess_format_compare_bench.cu
    )
    target_link_libraries(cellShardPreprocessFormatCompareBench PRIVATE CellShardPreprocess::runtime CUDA::cudart)
    set_target_properties(cellShardPreprocessFormatCompareBench PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
endif()

if(CELLSHARD_PREPROCESS_BUILD_TESTS)
    add_executable(cellShardPreprocessRuntimeTest
        tests/preprocess_runtime_test.cu
    )
    target_link_libraries(cellShardPreprocessRuntimeTest PRIVATE CellShardPreprocess::runtime CUDA::cudart)
    set_target_properties(cellShardPreprocessRuntimeTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_executable(cellShardPreprocessQCMetricsEquivalenceTest
        tests/qc_metrics_equivalence_test.cu
    )
    target_link_libraries(cellShardPreprocessQCMetricsEquivalenceTest PRIVATE CellShardPreprocess::runtime CUDA::cudart)
    set_target_properties(cellShardPreprocessQCMetricsEquivalenceTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_executable(cellShardPreprocessQCMaskGroupsTest
        tests/qc_mask_groups_test.cu
    )
    target_link_libraries(cellShardPreprocessQCMaskGroupsTest PRIVATE CellShardPreprocess::runtime CUDA::cudart)
    set_target_properties(cellShardPreprocessQCMaskGroupsTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_executable(cellShardPreprocessQCFleetTest
        tests/qc_fleet_test.cu
    )
    target_link_libraries(cellShardPreprocessQCFleetTest PRIVATE CellShardPreprocess::runtime CUDA::cudart)
    set_target_properties(cellShardPreprocessQCFleetTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_executable(cellShardPreprocessAdapterStagingTest
        tests/adapter_staging_test.cc
    )
    target_link_libraries(cellShardPreprocessAdapterStagingTest PRIVATE CellShardPreprocess::runtime)
    set_target_properties(cellShardPreprocessAdapterStagingTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )

    add_executable(cellShardPreprocessDoublePreprocessRejectionTest
        tests/double_preprocess_rejection_test.cc
    )
    target_link_libraries(cellShardPreprocessDoublePreprocessRejectionTest PRIVATE CellShardPreprocess::runtime)
    set_target_properties(cellShardPreprocessDoublePreprocessRejectionTest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
endif()
