/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/utility.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace gqe {

namespace physical {

class read_relation;
class broadcast_join_relation;
class project_relation;

/**
 * @brief Base interface for a physical relation visitor.
 *
 * A concrete visitor needs to override these methods to customize the behavior.
 */
struct relation_visitor {
  virtual void visit(read_relation* relation)
  {
    throw std::logic_error("Visiting read_relation is not implemented");
  }
  virtual void visit(broadcast_join_relation* relation)
  {
    throw std::logic_error("Visiting broadcast_join_relation is not implemented");
  }
  virtual void visit(project_relation* relation)
  {
    throw std::logic_error("Visiting project_relation is not implemented");
  }
};

/**
 * @brief Abstract base class for all physical relations.
 *
 * Compared to a logical relation, a physical relation encodes information on how to execute the
 * operation. For example, a logical join relation could correspond to either BroadcastJoin or
 * RepartitionedJoin physical relation.
 */
class relation {
 public:
  /**
   * @brief Construct a new physical relation.
   *
   * @param[in] children Child nodes of the new relation.
   */
  relation(std::vector<std::shared_ptr<relation>> children) : _children(std::move(children)) {}

  virtual ~relation()       = default;
  relation(const relation&) = delete;
  relation& operator=(const relation&) = delete;

  /**
   * @brief Return the child nodes of the current relation.
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  [[nodiscard]] std::vector<relation*> children_unsafe() const noexcept
  {
    return utility::to_raw_ptrs(_children);
  }

  /**
   * @brief Accept a visitor.
   *
   * Implement the visitor pattern (https://en.wikipedia.org/wiki/Visitor_pattern) through double
   * dispatch.
   */
  virtual void accept(relation_visitor& visitor) = 0;

 private:
  std::vector<std::shared_ptr<relation>> _children;
};

}  // namespace physical
}  // namespace gqe
