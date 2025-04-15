/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/relation_properties.hpp>

#include <memory>
#include <utility>

namespace gqe {
namespace optimizer {
class logical_optimizer;
class optimization_rule;

/**
 * @brief Class maintaining stats and properties of the owner relation
 *
 */
class relation_traits {
  friend class gqe::optimizer::logical_optimizer;
  friend class gqe::optimizer::optimization_rule;

 public:
  /**
   * @brief Construct a new relation traits object
   *
   */
  relation_traits() {}

  relation_traits(relation_properties properties) : _properties(std::move(properties)) {}

  /**
   * @brief Return relation properties
   */
  relation_properties const& properties() const { return _properties; }

  bool operator==(const relation_traits& other) const { return _properties == other._properties; }

  std::string to_string() const { return "properties:\n" + _properties.to_string(); }

 private:
  relation_properties _properties;
  // TODO: add relation stats
};
}  // namespace optimizer
}  // namespace gqe