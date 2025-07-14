/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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
#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>

#include <memory>
#include <vector>

namespace gqe {

/**
 * @brief Task that reads data from a base table.
 *
 * This abstract class serves as a base class for all read tasks.
 */
class read_task_base : public task {
 public:
  /**
   * @copydoc gqe::task::task()
   */
  read_task_base(context_reference ctx_ref,
                 int32_t task_id,
                 int32_t stage_id,
                 std::vector<std::shared_ptr<task>> subquery_tasks);

  read_task_base(const read_task_base&)            = delete;
  read_task_base& operator=(const read_task_base&) = delete;
};

}  // namespace gqe
