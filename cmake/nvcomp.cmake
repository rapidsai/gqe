# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Configure nvcomp for GQE.

include_guard(GLOBAL)

if(GQE_NVCOMP_LIB_DIR)
  message(STATUS "Using prebuilt nvCOMP from: ${GQE_NVCOMP_LIB_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${GQE_NVCOMP_LIB_DIR}")
  find_package(nvcomp CONFIG REQUIRED)
  list(REMOVE_AT CMAKE_PREFIX_PATH -1)
  install(IMPORTED_RUNTIME_ARTIFACTS nvcomp::nvcomp nvcomp::nvcomp_cpu LIBRARY
          DESTINATION ${CMAKE_INSTALL_LIBDIR})
elseif(GQE_ENABLE_PUBLIC_BUILD)
  # Public consumers can't reach the gqe-nvcomp.git fork. Fetch the
  # NVIDIA-published nvcomp redistrib from the public CDN; the manifest resolves
  # point revisions (5.2.0.10 → 5.2.0.11) automatically, so only release-line
  # bumps (5.2 → 5.3) need a GQE_NVCOMP_PUBLIC_VERSION change.
  set(GQE_NVCOMP_PUBLIC_VERSION
      "5.2.0"
      CACHE STRING "nvcomp redistrib release label (3-part, e.g. 5.2.0)")

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_nvcomp_arch "linux-sbsa")
  else()
    set(_nvcomp_arch "linux-x86_64")
  endif()
  string(REGEX MATCH "^[0-9]+" _cuda_major "${CMAKE_CUDA_COMPILER_VERSION}")
  if(NOT _cuda_major)
    message(
      FATAL_ERROR
        "Cannot resolve CUDA major version (CMAKE_CUDA_COMPILER_VERSION='${CMAKE_CUDA_COMPILER_VERSION}'); enable_language(CUDA) must run before nvcomp setup."
    )
  endif()
  set(_nvcomp_cdn_root
      "https://developer.download.nvidia.com/compute/nvcomp/redist")
  set(_nvcomp_manifest_url
      "${_nvcomp_cdn_root}/redistrib_${GQE_NVCOMP_PUBLIC_VERSION}.json")
  set(_nvcomp_manifest_file "${CMAKE_BINARY_DIR}/nvcomp_redistrib.json")
  file(
    DOWNLOAD "${_nvcomp_manifest_url}" "${_nvcomp_manifest_file}"
    STATUS _nvcomp_dl_status
    TLS_VERIFY ON)
  list(GET _nvcomp_dl_status 0 _nvcomp_dl_rc)
  if(NOT _nvcomp_dl_rc EQUAL 0)
    message(
      FATAL_ERROR
        "Failed to fetch nvcomp manifest from ${_nvcomp_manifest_url}: ${_nvcomp_dl_status}"
    )
  endif()
  file(READ "${_nvcomp_manifest_file}" _nvcomp_manifest_text)
  string(JSON _nvcomp_resolved_version GET "${_nvcomp_manifest_text}" "nvcomp"
         "version")
  string(
    JSON
    _nvcomp_entry
    GET
    "${_nvcomp_manifest_text}"
    "nvcomp"
    "${_nvcomp_arch}"
    "cuda${_cuda_major}")
  string(JSON _nvcomp_relpath GET "${_nvcomp_entry}" "relative_path")
  string(JSON _nvcomp_sha256 GET "${_nvcomp_entry}" "sha256")
  message(
    STATUS
      "Using public nvcomp ${_nvcomp_resolved_version} (${_nvcomp_arch}/cuda${_cuda_major})"
  )

  cpmaddpackage(
    NAME
    nvcomp
    URL
    "${_nvcomp_cdn_root}/${_nvcomp_relpath}"
    URL_HASH
    "SHA256=${_nvcomp_sha256}"
    DOWNLOAD_ONLY
    YES)

  list(APPEND CMAKE_PREFIX_PATH "${nvcomp_SOURCE_DIR}")
  # FetchContent_Populate leaks nvcomp_BINARY_DIR (and friends) into our scope
  # and short-circuits nvcomp-config.cmake's find_path. Clear both scopes.
  # Mirrors rapids-cmake's both-unset pattern for nvcomp_DIR (file removed
  # upstream for their use case — we still need it):
  # https://github.com/rapidsai/rapids-cmake/blob/branch-25.10/rapids-cmake/cpm/nvcomp.cmake
  foreach(_var nvcomp_BINARY_DIR nvcomp_INCLUDE_DIR nvcomp_LIBRARY_DIR)
    unset(${_var} CACHE)
    unset(${_var})
  endforeach()
  find_package(nvcomp CONFIG REQUIRED)
  list(REMOVE_AT CMAKE_PREFIX_PATH -1)
  # Install libnvcomp.so.5 + libnvcomp_cpu.so.5 (with SONAME symlinks) to the
  # project LIBDIR. gqe binaries' INSTALL_RPATH
  # ($ORIGIN/../${GQE_INSTALL_LIBDIR}) finds them, and libcudf.so's DT_NEEDED
  # libnvcomp.so.5 resolves through the same RPATH via DT_RPATH propagation.
  install(IMPORTED_RUNTIME_ARTIFACTS nvcomp::nvcomp nvcomp::nvcomp_cpu LIBRARY
          DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
  # Internal build: fetch the gqe-nvcomp fork (custom nvcomp build).
  set(GQE_NVCOMP_TAG
      "4f625bf09309d15bde961f716a42b23afeee2a81"
      CACHE STRING "Git tag for gqe-nvcomp")
  set(GQE_NVCOMP_SOURCE_DIR
      ""
      CACHE PATH
            "Local source directory for gqe-nvcomp (overrides GIT_TAG if set)")

  # Fetch nvcomp from source using CPM
  if(GQE_NVCOMP_SOURCE_DIR)
    message(
      STATUS
        "Using nvcomp from local source directory: ${GQE_NVCOMP_SOURCE_DIR}")
    cpmaddpackage(
      NAME
      nvcomp
      SOURCE_DIR
      ${GQE_NVCOMP_SOURCE_DIR}
      OPTIONS
      "BUILD_TESTS OFF"
      "BUILD_BENCHMARKS OFF"
      "BUILD_EXAMPLES OFF"
      "BUILD_NVCOMP ON"
      "BUILD_NVCOMPDX OFF"
      "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}"
      "BUILD_CPU_COMPRESSORS ON")
  else()
    message(STATUS "Using nvcomp from git with tag: ${GQE_NVCOMP_TAG}")
    cpmaddpackage(
      NAME
      nvcomp
      GIT_REPOSITORY
      https://gitlab-master.nvidia.com/Devtech-Compute/gqe-nvcomp.git
      GIT_TAG
      ${GQE_NVCOMP_TAG}
      GIT_SUBMODULES_RECURSE
      TRUE
      OPTIONS
      "BUILD_TESTS OFF"
      "BUILD_BENCHMARKS OFF"
      "BUILD_EXAMPLES OFF"
      "BUILD_NVCOMP ON"
      "BUILD_NVCOMPDX OFF"
      "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}"
      "BUILD_CPU_COMPRESSORS ON")
  endif()
  # Fork target names are unprefixed; nvcomp::nvcomp is an ALIAS for nvcomp.
  # Install the SHARED libs into the project LIBDIR so gqe binaries' RPATH
  # resolves them at runtime.
  install(TARGETS nvcomp nvcomp_cpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
