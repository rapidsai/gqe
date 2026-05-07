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

#include <vector>

namespace gqe {

class partition_merge_task : public task {
 public:
  /**
   * @brief A task that merges the input tables into a single table based on the partition index.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] inputs Input tables to be merged.
   * @param[in] partition_idx Partition index.
   */
  partition_merge_task(context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::vector<std::shared_ptr<task>> inputs,
                       int32_t partition_idx);
  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  int32_t _partition_idx;
};

}  // namespace gqe
