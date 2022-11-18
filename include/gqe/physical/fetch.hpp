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

#include <gqe/physical/relation.hpp>

#include <cstdint>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for getting a consecutive subset of rows based on `count` and `offset`.
 */
class fetch_relation : public relation {
 public:
  /**
   * @brief Construct a physical fetch relation.
   *
   * @param[in] input Input table.
   * @param[in] offset Row index from which the fetch starts.
   * @param[in] count Number of rows to retrieve starting from `offset`.
   */
  fetch_relation(std::shared_ptr<relation> input, int64_t offset, int64_t count)
    : relation({std::move(input)}, {}), _offset(offset), _count(count)
  {
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the row index from which the fetch starts.
   */
  [[nodiscard]] int64_t offset() const noexcept { return _offset; }

  /**
   * @brief Return the number of rows to retrieve.
   */
  [[nodiscard]] int64_t count() const noexcept { return _count; }

 private:
  int64_t _offset;
  int64_t _count;
};

}  // namespace physical
}  // namespace gqe
