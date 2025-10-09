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
