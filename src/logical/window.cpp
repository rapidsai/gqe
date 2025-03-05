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

#include <gqe/logical/utility.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/types.hpp>

#include <vector>

namespace gqe {
namespace logical {

window_relation::window_relation(std::shared_ptr<relation> input_relation,
                                 std::vector<std::shared_ptr<relation>> subquery_relations,
                                 cudf::aggregation::Kind aggr_func,
                                 std::vector<std::unique_ptr<expression>> arguments,
                                 std::vector<std::unique_ptr<expression>> order_by,
                                 std::vector<std::unique_ptr<expression>> partition_by,
                                 std::vector<cudf::order> order_dirs,
                                 window_frame_bound::type window_lower_bound,
                                 window_frame_bound::type window_upper_bound)
  : relation({std::move(input_relation)}, std::move(subquery_relations)),
    _aggr_func(aggr_func),
    _arguments(std::move(arguments)),
    _order_by(std::move(order_by)),
    _partition_by(std::move(partition_by)),
    _order_dirs(std::move(order_dirs)),
    _window_lower_bound(window_lower_bound),
    _window_upper_bound(window_upper_bound)
{
}

std::string window_relation::to_string() const
{
  std::string window_relation_string = "{\"Window\" : {\n";
  window_relation_string +=
    "\t\"cudf::aggregation::Kind\" : " +
    std::to_string(static_cast<std::underlying_type<cudf::aggregation::Kind>::type>(aggr_func())) +
    ",\n";
  window_relation_string +=
    "\t\"arguments\" : " + utility::list_to_string(arguments_unsafe()) + ",\n";
  window_relation_string +=
    "\t\"order_by\" : " + utility::list_to_string(order_by_unsafe()) + ",\n";
  window_relation_string +=
    "\t\"partition_by\" : " + utility::list_to_string(partition_by_unsafe()) + ",\n";
  window_relation_string += "\t\"data types\" : " + utility::list_to_string(data_types()) + ",\n";
  // Children
  window_relation_string += "\t\"children\" : " + utility::list_to_string(children_unsafe()) + "\n";
  window_relation_string += "}}";
  return window_relation_string;
}

cudf::aggregation::Kind window_relation::aggr_func() const noexcept { return _aggr_func; }

std::vector<expression*> window_relation::arguments_unsafe() const noexcept
{
  return gqe::utility::to_raw_ptrs(_arguments);
}

std::vector<expression*> window_relation::order_by_unsafe() const noexcept
{
  return gqe::utility::to_raw_ptrs(_order_by);
}

std::vector<expression*> window_relation::partition_by_unsafe() const noexcept
{
  return gqe::utility::to_raw_ptrs(_partition_by);
}

std::vector<cudf::order> window_relation::order_dirs() const noexcept { return _order_dirs; }

window_frame_bound::type window_relation::window_lower_bound() const noexcept
{
  return _window_lower_bound;
}
window_frame_bound::type window_relation::window_upper_bound() const noexcept
{
  return _window_upper_bound;
}

bool window_relation::operator==(const relation& other) const
{
  auto this_type = this->type();
  if (this_type != other.type()) {
    utility::log_relation_comparison_message(
      this_type,
      "operator==() relation type mismatch with " + utility::relation_type_str(other.type()));
    return false;
  }
  auto other_window_relation = dynamic_cast<const window_relation*>(&other);
  // Compare attributes
  if (this->aggr_func() != other_window_relation->aggr_func()) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): aggregate function kind mismatch");
    return false;
  }
  if (this->window_lower_bound() != other_window_relation->window_lower_bound()) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): window lower bound mismatch");
    return false;
  }
  if (this->window_upper_bound() != other_window_relation->window_upper_bound()) {
    utility::log_relation_comparison_message(this_type,
                                             "operator==(): window upper bound mismatch");
    return false;
  }
  if (!gqe::utility::compare_pointer_vectors(this->arguments_unsafe(),
                                             other_window_relation->arguments_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): arguments mismatch");
    return false;
  }
  if (!gqe::utility::compare_pointer_vectors(this->order_by_unsafe(),
                                             other_window_relation->order_by_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): order keys mismatch");
    return false;
  }
  if (!gqe::utility::compare_pointer_vectors(this->partition_by_unsafe(),
                                             other_window_relation->partition_by_unsafe())) {
    utility::log_relation_comparison_message(this_type, "operator==(): partition keys mismatch");
    return false;
  }
  if (this->order_dirs() != other_window_relation->order_dirs()) {
    utility::log_relation_comparison_message(this_type, "operator==(): order directions mismatch");
    return false;
  }
  if (this->data_types() != other_window_relation->data_types()) {
    utility::log_relation_comparison_message(this_type, "operator==(): data types mismatch");
    return false;
  }
  // Compare members defined in base class
  return relation::compare_relation_members(other);
}

std::vector<cudf::data_type> window_relation::data_types() const
{
  auto data_types = this->children_unsafe()[0]->data_types();
  auto args       = arguments_unsafe();
  if (args.size() > 0) {
    cudf::data_type output_type =
      cudf::detail::target_type(args[0]->data_type(data_types), _aggr_func);
    data_types.push_back(output_type);
  } else {
    // If no arguments, we can assume the aggregation function is RANK
    if (_aggr_func != cudf::aggregation::RANK) {
      throw std::runtime_error("Only RANK is supported for window relations with no arguments.");
    }
    data_types.emplace_back(cudf::data_type(cudf::type_to_id<cudf::size_type>()));
  }
  return data_types;
}

}  // namespace logical
}  // namespace gqe
