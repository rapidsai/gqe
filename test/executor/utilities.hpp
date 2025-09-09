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

#include <gqe/context_reference.hpp>
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
  executed_task(gqe::context_reference ctx_ref,
                int32_t task_id,
                int32_t stage_id,
                std::unique_ptr<cudf::table> result,
                std::vector<std::shared_ptr<task>> dependencies   = {},
                std::vector<std::shared_ptr<task>> subquery_tasks = {})
    : task(ctx_ref, task_id, stage_id, dependencies, subquery_tasks)
  {
    emit_result(std::move(result));
  }

  void execute() override {}
};

class no_op_task : public task {
 public:
  no_op_task(gqe::context_reference ctx_ref,
             int32_t task_id,
             int32_t stage_id,
             std::vector<std::shared_ptr<task>> dependencies   = {},
             std::vector<std::shared_ptr<task>> subquery_tasks = {})
    : task(ctx_ref, task_id, stage_id, dependencies, subquery_tasks)
  {
  }

  void execute() override {}
};

}  // namespace test

}  // namespace gqe
