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

#include <gqe/utility/helpers.hpp>
#include <memory>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace gqe {

namespace memory_resource {

/**
 * @brief Create a shared, statically allocated memory pool.
 *
 * Test classes should use this memory resource when they create a `gqe::task_manager_context`
 * to speed up test execution.
 */
std::unique_ptr<rmm::mr::device_memory_resource> create_static_memory_pool();

}  // namespace memory_resource

}  // namespace gqe
