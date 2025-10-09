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
#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>

#include <vector>

namespace gqe {

class partition_task : public task {
 public:
  /**
   * @brief A task that partitions the input table into the given number of partitions.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be partitioned.
   * @param[in] partition_cols Partition columns.
   * @param[in] num_partitions Number of partitions.
   *
   */
  partition_task(context_reference ctx_ref,
                 int32_t task_id,
                 int32_t stage_id,
                 std::shared_ptr<task> input,
                 std::vector<std::unique_ptr<expression>> partition_cols,
                 int32_t num_partitions);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

  /**
   * @brief Return the offset of the partition with the given index.
   *
   * @note This function can only be called after the task has been executed.
   * otherwise, it hits the assertion.
   *
   * @param[in] partition_idx The index of the partition.
   * @return The offset of the partition.
   */
  cudf::size_type partition_offset(int32_t partition_idx)
  {
    assert(partition_idx >= 0 && partition_idx <= _num_partitions);
    return _partition_offsets[partition_idx];
  }

 private:
  std::vector<std::unique_ptr<expression>> _partition_cols;
  int32_t _num_partitions;
  std::vector<cudf::size_type> _partition_offsets;
};

}  // namespace gqe
