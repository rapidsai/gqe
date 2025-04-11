/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/device_properties.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <memory>
#include <mutex>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gqe {

/**
 * @brief Implements an wrapper around rmm::cuda_stream to allow for cooperative stream sharing
 * among multiple threads.
 */
struct shared_stream {
  std::mutex mtx;
  rmm::cuda_stream stream;
};

/**
 * @brief Task manager context for query execution
 *
 * The task manager context centralizes all important resources and parameters that
 * are relevant for execution across queries on a node.
 *
 */
struct task_manager_context {
  explicit task_manager_context(device_properties device_prop = device_properties())
    : _device_properties{device_prop} {};
  task_manager_context(const task_manager_context&) = delete;
  task_manager_context(task_manager_context&&)      = default;
  task_manager_context& operator=(const task_manager_context&) = delete;
  task_manager_context& operator=(task_manager_context&&) = default;

  device_properties _device_properties;
  shared_stream copy_engine_stream;
};

}  // namespace gqe