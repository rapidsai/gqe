/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
