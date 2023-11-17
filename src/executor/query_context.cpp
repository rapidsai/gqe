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

#include <gqe/executor/query_context.hpp>
#include <gqe/utility/error.hpp>

#include <cstdlib>
#include <stdexcept>

namespace gqe {

query_context::query_context(optimization_parameters const* parameters) : parameters(parameters)
{
  if (parameters->use_customized_io) {
    // Allocate page-locked CPU bounce buffers
    io_bounce_buffer_mr = std::make_unique<
      rmm::mr::fixed_size_memory_resource<gqe::memory_resource::pinned_memory_resource>>(
      &_pinned_mr,
      static_cast<int64_t>(parameters->io_bounce_buffer_size) * 1024 * 1024 * 1024,
      parameters->max_num_workers);
  }
}

}  // namespace gqe
