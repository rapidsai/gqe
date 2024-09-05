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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/optimizer/optimization_configuration.hpp>
#include <gqe/utility/timer.hpp>

#include <rmm/mr/device/fixed_size_memory_resource.hpp>

#include <cstdint>
#include <memory>

namespace gqe {

/**
 * @brief Query context of the current query.
 *
 * The query context centralizes all important resources and parameters that
 * are relevant for query execution. This enables optimizations that rely on
 * context, as well as logging and debugging.
 *
 * Implementation note: Add a comment for each struct member documenting its purpose.
 * Each member must have a default setting.
 */
struct query_context {
  query_context(query_context&& other) = default;
  explicit query_context(optimization_parameters parameters,
                         optimizer::optimization_configuration logical_rule_config =
                           optimizer::optimization_configuration());

  query_context()                     = delete;
  query_context(query_context const&) = delete;
  query_context& operator=(query_context const&) = delete;

  gqe::optimization_parameters parameters;

  gqe::optimizer::optimization_configuration logical_rule_config;

  // Memory resource used to allocate host-accessible bounce buffers for the customized Parquet
  // reader.
  std::unique_ptr<rmm::mr::fixed_size_memory_resource<gqe::memory_resource::pinned_memory_resource>>
    io_bounce_buffer_mr = nullptr;

  utility::bandwidth_timer disk_timer{"disk_read"}, h2d_timer{"h2d"}, decomp_timer{"decompress"},
    decode_timer{"decode"};

 private:
  gqe::memory_resource::pinned_memory_resource _pinned_mr;
};

}  // namespace gqe
