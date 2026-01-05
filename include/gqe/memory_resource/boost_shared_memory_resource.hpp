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

#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

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
