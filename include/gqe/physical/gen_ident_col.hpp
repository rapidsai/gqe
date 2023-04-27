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

#include <memory>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for generating a unique row identifier column.
 */
class gen_ident_col_relation : public relation {
 public:
  /**
   * @brief Construct a physical gen_ident_col relation.
   *
   * @param[in] input Input table to which the identifier col will be appended
   */
  gen_ident_col_relation(std::shared_ptr<relation> input) : relation({std::move(input)}, {}) {}

  void accept(relation_visitor& visitor) override { visitor.visit(this); }
};

}  // namespace physical
}  // namespace gqe