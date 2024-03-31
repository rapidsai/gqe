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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace logical {

/**
 * @brief The fetch relation is used for limiting the number of rows returned.
 *
 * If the offset is specified, it will also return the row starting from the spcified
 * offset.
 */
class fetch_relation : public relation {
 public:
  /**
   * @brief Construct a new fetch relation object
   *
   * @param input_relation Input to the fetch relation
   * @param offset The offset experssed in number of records
   * @param count The number of records to return
   */
  fetch_relation(std::shared_ptr<relation> input_relation, int64_t offset, int64_t count);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::fetch; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the offset for the retrieval records
   *
   * @return The index of the starting row to be returned
   */
  [[nodiscard]] int64_t offset() const noexcept { return _offset; };

  /**
   * @brief Return the count for the retrieval records
   *
   * @return The number of rows to be returned
   */
  [[nodiscard]] int64_t count() const noexcept { return _count; };

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  int64_t _offset;
  int64_t _count;
};

}  // namespace logical
}  // namespace gqe