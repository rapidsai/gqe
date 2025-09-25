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

#include <gqe/physical/relation.hpp>
#include <gqe/types.hpp>

namespace gqe {
namespace physical {

class user_defined_relation : public relation {
 public:
  /**
   * @brief Construct a new physical user-defined relation.
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
   * @param[in] last_child_break_pipeline Whether the last child relation is a pipeline breaker. If
   * `false`, *task_functor* starts at the same stage as the tasks of the last child relation. If
   * `true`, *task_functor* starts at a new stage.
   */
  user_defined_relation(std::vector<std::shared_ptr<relation>> children,
                        user_defined_task_functor task_functor,
                        bool last_child_break_pipeline)
    : relation(std::move(children), {}),
      _task_functor(std::move(task_functor)),
      _last_child_break_pipeline(last_child_break_pipeline)
  {
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the functor used for generating output tasks from input tasks.
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
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override { return {}; }

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

 private:
  user_defined_task_functor _task_functor;
  bool _last_child_break_pipeline;
};

}  // namespace physical
}  // namespace gqe
