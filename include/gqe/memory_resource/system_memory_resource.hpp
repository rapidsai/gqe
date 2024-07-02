/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief System memory resource for pageable memory.
 *
 * Calls `new` and `delete` to allocate and deallocate memory resources.
 */
class system_memory_resource : public rmm::mr::device_memory_resource {
 public:
  system_memory_resource()           = default;
  ~system_memory_resource() override = default;

 private:
  static constexpr auto _allocation_alignment =
    std::align_val_t{256}; /**< `rmm::mr::device_memory_resource` specifies that all allocations are
                            256-byte aligned. */

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override;

  void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view) override;
};

}  // namespace memory_resource

}  // namespace gqe
