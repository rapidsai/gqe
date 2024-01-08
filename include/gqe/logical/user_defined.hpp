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

#include <gqe/logical/relation.hpp>
#include <gqe/types.hpp>

namespace gqe {
namespace logical {

class user_defined_relation : public relation {
 public:
  /**
   * @brief Construct a new user-defined relation.
   *
   * During task graph generation, each child relation of the user-defined relation will first be
   * translated into tasks. These tasks serve as the input and will be passed to the *task_functor*
   * to produce the output tasks.
   *
   * Note: all children relations except the last one must be a pipeline breaker to make sure the
   * subsequent children have the correct stage ID. The pipeline breakers will be inserted
   * automatically by the task graph generator.
   *
   * @param[in] children Input relations.
   * @param[in] task_functor Functor which takes the input tasks and returns the output tasks.
   * @param[in] data_types Output column types.
   * @param[in] last_child_break_pipeline Whether the last child relation is a pipeline breaker. If
   * `false`, *task_functor* starts at the same stage as the tasks of the last child relation. If
   * `true`, *task_functor* starts at a new stage.
   */
  user_defined_relation(std::vector<std::shared_ptr<relation>> children,
                        user_defined_task_functor task_functor,
                        std::vector<cudf::data_type> data_types,
                        bool last_child_break_pipeline = true)
    : relation(std::move(children), {}),
      _task_functor(std::move(task_functor)),
      _data_types(std::move(data_types)),
      _last_child_break_pipeline(last_child_break_pipeline)
  {
  }

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept final { return relation_type::user_defined; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override { return _data_types; };

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the functor used to generate output tasks from input tasks.
   */
  [[nodiscard]] user_defined_task_functor task_functor() const noexcept { return _task_functor; }

  /**
   * @brief Return whether the last child is a pipeline breaker.
   */
  [[nodiscard]] bool last_child_break_pipeline() const noexcept
  {
    return _last_child_break_pipeline;
  }

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override { return false; }

 private:
  user_defined_task_functor _task_functor;
  std::vector<cudf::data_type> _data_types;
  bool _last_child_break_pipeline;
};

}  // namespace logical
}  // namespace gqe
