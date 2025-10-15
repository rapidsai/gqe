/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/interprocess/managed_shared_memory.hpp>
#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief Boost shared memory resource for inter-process CPU shared memory.
 *
 */
class boost_shared_memory_resource : public rmm::mr::device_memory_resource {
 public:
  boost_shared_memory_resource();
  ~boost_shared_memory_resource();

  boost::interprocess::managed_shared_memory& segment() { return _segment; }

 private:
  boost::interprocess::managed_shared_memory _segment;

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override;
  void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view) override;
};

}  // namespace memory_resource

}  // namespace gqe
