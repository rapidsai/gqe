/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gqe/device_properties.hpp>

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#include <nvml.h>

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {

namespace {

// Ref:
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#query-for-support
constexpr int k_min_driver_version_for_async_pools       = 11020;
constexpr int k_min_driver_version_for_pool_handle_types = 11030;

std::vector<rmm::cuda_device_id> get_all_devices()
{
  int deviceCount;
  GQE_CUDA_TRY(cudaGetDeviceCount(&deviceCount));
  std::vector<rmm::cuda_device_id> devices;
  devices.reserve(deviceCount);
  for (auto i = 0; i < deviceCount; i++) {
    devices.emplace_back(rmm::cuda_device_id{i});
  }
  return devices;
}

/**
 * @brief Query CPU affinity of a GPU device within socket scope.
 *
 * The intended purpose is to set worker threads to run on CPUs that are closest to the device.
 * Therefore, this function returns NVML_AFFINITY_SCOPE_NODE.
 *
 * @return cpu_set A set of CPUs on the NUMA node closest to the device.
 */
cpu_set query_cpu_affinity(nvmlDevice_t nvml_device)
{
  cpu_set result;
  GQE_NVML_TRY(nvmlDeviceGetCpuAffinityWithinScope(
    nvml_device, cpu_set::qword_count, result.bits(), NVML_AFFINITY_SCOPE_NODE));
  return result;
}

/**
 * @brief Query NUMA node affinity of a device.
 *
 * The intended purpose is to allocate memory close to the device. On some systems such as AMD Rome,
 * the CPU socket may have multiple NUMA nodes that are physically close to the device. Therefore,
 * this function returns NVML_AFFINITY_SCOPE_SOCKET.
 *
 * @return cpu_set A set of NUMA nodes on the CPU socket closest to the device.
 */
cpu_set query_memory_affinity(nvmlDevice_t nvml_device)
{
  cpu_set result;
  GQE_NVML_TRY(nvmlDeviceGetMemoryAffinity(
    nvml_device, cpu_set::qword_count, result.bits(), NVML_AFFINITY_SCOPE_SOCKET));
  return result;
}

/**
 * @brief Query the host NUMA ID for the device.
 *
 * @return int The host NUMA node ID closest to the device.
 */
int query_host_numa_id(rmm::cuda_device_id device)
{
  int host_numa_id = 0;
  CUresult result =
    cuDeviceGetAttribute(&host_numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device.value());
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to get device attribute HOST_NUMA_ID");
  }
  return host_numa_id;
}

/**
 * @brief Query the memory decompression support.
 *
 * @return bool True if the device supports memory decompression.
 */
bool query_mem_decompress_support(rmm::cuda_device_id device)
{
  int mem_decompress_support = 0;
  CUresult result            = cuDeviceGetAttribute(
    &mem_decompress_support, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device.value());
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to get device attribute MEM_DECOMPRESS_ALGORITHM_MASK");
  }
  return static_cast<bool>(mem_decompress_support);
}
}  // namespace

device_properties const& device_properties::instance()
{
  static device_properties instance;
  return instance;
}

device_properties::device_properties()
{
  // Initialize NVML.
  GQE_NVML_TRY(nvmlInit_v2());

  // Get driver version once for all devices.
  int driver_version = 0;
  GQE_CUDA_TRY(cudaDriverGetVersion(&driver_version));

  for (auto const& device : get_all_devices()) {
    // Cache CUDA runtime properties.
    cudaDeviceProp cuda_properties;
    GQE_CUDA_TRY(cudaGetDeviceProperties(&cuda_properties, device.value()));

    // Query memory pools support with driver version check.
    int memory_pools_supported = 0;
    if (driver_version >= k_min_driver_version_for_async_pools) {
      GQE_CUDA_TRY(cudaDeviceGetAttribute(
        &memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device.value()));
    }

    // Query memory pool supported handle types with driver version check.
    int memory_pool_supported_handle_types = 0;
    if (driver_version >= k_min_driver_version_for_pool_handle_types) {
      GQE_CUDA_TRY(cudaDeviceGetAttribute(&memory_pool_supported_handle_types,
                                          cudaDevAttrMemoryPoolSupportedHandleTypes,
                                          device.value()));
    }

    runtime_properties runtime_props{
      .managed_memory                     = static_cast<bool>(cuda_properties.managedMemory),
      .memory_pools_supported             = static_cast<bool>(memory_pools_supported),
      .memory_pool_supported_handle_types = memory_pool_supported_handle_types,
      .multi_processor_count              = cuda_properties.multiProcessorCount,
      .pageable_memory_access             = static_cast<bool>(cuda_properties.pageableMemoryAccess),
      .unified_addressing                 = static_cast<bool>(cuda_properties.unifiedAddressing)};
    _runtime_properties_cache.insert(std::make_pair(device.value(), std::move(runtime_props)));

    // Cache CUDA driver properties.
    driver_properties driver_props{.host_numa_id           = query_host_numa_id(device),
                                   .mem_decompress_support = query_mem_decompress_support(device)};
    _driver_properties_cache.insert(std::make_pair(device.value(), std::move(driver_props)));

    // Get NVML device handle by PCI bus ID to ensure that NVML gets the same device as CUDA. NVML
    // does not guarantee that the device ordinals are the same.
    //
    // References:
    // https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1gab1895084b6b423475665063c63763410
    char pci_bus_id[32];
    GQE_CUDA_TRY(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device.value()));
    nvmlDevice_t nvml_device;
    GQE_NVML_TRY(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &nvml_device));

    // Cache NVML properties.
    nvml_properties nvml_props{.cpu_affinity    = query_cpu_affinity(nvml_device),
                               .memory_affinity = query_memory_affinity(nvml_device)};
    _nvml_properties_cache.insert(std::make_pair(device.value(), std::move(nvml_props)));
  }

  // Shutdown NVML after caching all properties.
  nvmlShutdown();
}

device_properties::driver_properties const& device_properties::get_driver_properties(
  rmm::cuda_device_id device) const
{
  auto match = _driver_properties_cache.find(device.value());
  if (match == _driver_properties_cache.end()) {
    throw std::runtime_error("CUDA driver properties not found for device");
  }
  return match->second;
}

device_properties::nvml_properties const& device_properties::get_nvml_properties(
  rmm::cuda_device_id device) const
{
  auto match = _nvml_properties_cache.find(device.value());
  if (match == _nvml_properties_cache.end()) {
    throw std::runtime_error("NVML properties not found for device");
  }
  return match->second;
}

device_properties::runtime_properties const& device_properties::get_runtime_properties(
  rmm::cuda_device_id device) const
{
  auto match = _runtime_properties_cache.find(device.value());
  if (match == _runtime_properties_cache.end()) {
    throw std::runtime_error("CUDA runtime properties not found for device");
  }
  return match->second;
}

}  // namespace gqe
