/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once
#include <rmm/mr/device/device_memory_resource.hpp>

namespace gqe {
/**
 * @brief PGAS memory resource allows suballocations within this resource to be accessed by RMA with
 * NVSHMEM. NVSHMEM will have undefined behavior if API calls to RMA routines use suballocations in
 * different pgas_memory_resource objects.
 *
 * NVSHMEM requires initialization before creating resource of this type.
 */
class pgas_memory_resource : public rmm::mr::device_memory_resource {
 public:
  pgas_memory_resource(std::size_t bytes);
  pgas_memory_resource(pgas_memory_resource const&)            = delete;
  pgas_memory_resource& operator=(pgas_memory_resource const&) = delete;
  void* get_local_base_ptr() const { return _local_base_ptr; }
  std::size_t get_bytes() const { return _bytes; }
  /**
   * @brief Finalize the PGAS memory resource.
   *
   * This function will free the symmetric memory allocated by this resource. Function ensures that
   * all processes have finished RMA operations before freeing the memory.
   */
  void finalize();

 private:
  /**
   * @brief Allocates memory of size at least \p bytes as a NVSHMEM symmetric object.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @param[in] bytes The size of the allocation
   * @param[in] stream Stream parameter is ignored
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override;

  /**
   * @brief Deallocated the NVSHMEM symmetric object. The input parameter should match the pointer
   * and size used during allocation.
   *
   * @param[in] ptr   Pointer to memory to be deallocated
   * @param[in] bytes The size of the allocation
   * @param[in] stream Stream parameter is ignored
   */
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view) override;

  void* _local_base_ptr;  // Pointer to the local base of the symmetric memory
  bool _allocated;        // Whether the memory has been allocated
  std::size_t _bytes;     // Total size of the symmetric memory
};

}  // namespace gqe
