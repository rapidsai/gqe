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

#include <gqe/logical/relation.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/types.hpp>

#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gqe {
namespace logical {
namespace utility {

/**
 * @brief Return string representation of the list of output data types
 *
 * @param types Vector of cuDF data types
 * @return Output data types string repressentation
 */
inline std::string list_to_string(std::vector<cudf::data_type> const& types)
{
  std::string data_type_string = "[";
  bool first                   = true;
  for (auto dt : types) {
    if (!first) data_type_string += ", ";
    data_type_string += "\"" + cudf::type_to_name(dt) + "\"";
    first = false;
  }
  return data_type_string + "]";
}

/**
 * @brief Return string representation of relation list
 *
 * @param relation_list The list of relations to convert to string
 * @return Relation list string representation
 */
inline std::string list_to_string(std::vector<logical::relation*> relation_list)
{
  std::string relation_list_string = "[";
  bool first                       = true;
  for (auto relation : relation_list) {
    if (!first) relation_list_string += ", ";
    relation_list_string += relation->to_string();
    first = false;
  }
  relation_list_string += "]";
  return relation_list_string;
}

/**
 * @brief Return string representation of expression list
 *
 * @param expression_list The list of relations to convert to string
 * @return Expression list string representation
 */
inline std::string list_to_string(std::vector<expression*> expression_list)
{
  std::string expression_list_string = "[";
  bool first                         = true;
  for (auto expr : expression_list) {
    if (!first) expression_list_string += ", ";
    expression_list_string += "\"" + expr->to_string() + "\"";
    first = false;
  }
  expression_list_string += "]";
  return expression_list_string;
}

inline std::ostream& operator<<(std::ostream& os, cudf::order order)
{
  switch (order) {
    case cudf::order::ASCENDING: os << "ASC"; break;
    case cudf::order::DESCENDING: os << "DESC"; break;
    default:
      throw std::runtime_error("Invalid sort order enum: " +
                               std::to_string(static_cast<int>(order)));
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, cudf::null_order prec)
{
  switch (prec) {
    case cudf::null_order::BEFORE: os << "NULLS FIRST"; break;
    case cudf::null_order::AFTER: os << "NULLS LAST"; break;
    default:
      throw std::runtime_error("Invalid null order enum: " +
                               std::to_string(static_cast<int>(prec)));
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, cudf::aggregation::Kind k)
{
  switch (k) {
    case cudf::aggregation::COUNT_ALL: os << "COUNT_ALL"; break;
    case cudf::aggregation::COUNT_VALID: os << "COUNT_VALID"; break;
    case cudf::aggregation::MAX: os << "MAX"; break;
    case cudf::aggregation::MEAN: os << "MEAN"; break;
    case cudf::aggregation::MEDIAN: os << "MEDIAN"; break;
    case cudf::aggregation::MIN: os << "MIN"; break;
    case cudf::aggregation::STD: os << "STDDEV"; break;
    case cudf::aggregation::SUM: os << "SUM"; break;
    case cudf::aggregation::VARIANCE: os << "VARIANCE"; break;
    default: os << "Unsupported aggregation kind to string: " + std::to_string(k);
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                std::pair<cudf::aggregation::Kind, gqe::expression*> measure)
{
  os << measure.first << "(" << measure.second->to_string() << ")";
  return os;
}

/**
 * @brief Return string representation of list of elements in input vector
 *
 * @tparam T Type of values in input vector
 * @param begin Input vector begin iterator
 * @param end Input vector end iterator
 * @return Input vector string representation
 */
template <typename T>
inline std::string list_to_string(T begin, T end)
{
  std::stringstream ss;
  ss << "[";
  bool first = true;
  for (; begin != end; begin++) {
    if (!first) ss << ", ";
    ss << "\"";
    ss << *begin;
    ss << "\"";
    first = false;
  }
  ss << "]";
  return ss.str();
}

/**
 * @brief Convert `join_type` to its string representation
 *
 * @param join_type Join type to return string for
 * @return String representation on `join_type`
 */
inline std::string join_type_str(join_type_type join_type)
{
  switch (join_type) {
    case join_type_type::inner: return "inner"; break;
    case join_type_type::left: return "left"; break;
    case join_type_type::left_semi: return "left semi"; break;
    case join_type_type::full: return "full"; break;
    case join_type_type::left_anti: return "left anti"; break;
    case join_type_type::single: return "single"; break;
    default:
      throw std::runtime_error("Join type enum " + std::to_string(static_cast<int>(join_type)) +
                               " not supported");
  }
}

/**
 * @brief Turn relation::relation_type into string for logging
 */
inline std::string relation_type_str(relation::relation_type relation_type)
{
  std::string prefix = "logical ";
  switch (relation_type) {
    case relation::relation_type::fetch: return prefix + "fetch";
    case relation::relation_type::sort: return prefix + "sort";
    case relation::relation_type::project: return prefix + "project";
    case relation::relation_type::aggregate: return prefix + "aggregate";
    case relation::relation_type::join: return prefix + "join";
    case relation::relation_type::window: return prefix + "window";
    case relation::relation_type::read: return prefix + "read";
    case relation::relation_type::write: return prefix + "write";
    case relation::relation_type::filter: return prefix + "filter";
    case relation::relation_type::set: return prefix + "set";
    case relation::relation_type::user_defined: return prefix + "user defined";
    default:
      throw std::runtime_error("Logical relation type enum " +
                               std::to_string(static_cast<int>(relation_type)) + " not supported");
  }
}

/**
 * @brief Log the message with the input relation type information
 */
inline void log_relation_comparison_message(relation::relation_type relation_type,
                                            std::string message)
{
  GQE_LOG_TRACE(relation_type_str(relation_type) + ": " + message);
}

}  // namespace utility
}  // namespace logical
}  // namespace gqe
