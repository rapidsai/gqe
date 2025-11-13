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

  std::string to_string() const { return "\"properties\":\n" + _properties.to_string(); }

 private:
  relation_properties _properties;
  // TODO: add relation stats
};
}  // namespace optimizer
}  // namespace gqe