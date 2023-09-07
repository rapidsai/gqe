/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task.hpp>

#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>

namespace gqe {

namespace test {

/**
 * @brief A task that has a valid result during construction time. Used for testing.
 */
class executed_task : public task {
 public:
  executed_task(gqe::query_context* query_context,
                int32_t task_id,
                int32_t stage_id,
                std::unique_ptr<cudf::table> result)
    : task(query_context, task_id, stage_id, {}, {})
  {
    emit_result(std::move(result));
  }

  void execute() override {}
};

}  // namespace test

}  // namespace gqe
