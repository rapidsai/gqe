/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cxx_gqe/logical.hpp>
#include <cxx_gqe/types.hpp>

// Include wrappers of Rust std library types.
#include "rust/cxx.h"

#include <memory>

namespace cxx_gqe {

/*
 * @brief Returns a new read relation wrapper.
 */
std::shared_ptr<logical_relation> new_read_relation(
  rust::Slice<const std::shared_ptr<logical_relation>> subquery_relations,
  rust::Slice<const rust::String> column_names,
  rust::Slice<const type_id> column_types,
  const rust::Str table_name);

/*
 * @brief Returns a new write relation wrapper.
 */
std::shared_ptr<logical_relation> new_write_relation(
  std::shared_ptr<logical_relation> input_relation,
  rust::Slice<const rust::String> column_names,
  rust::Slice<const type_id> column_types,
  const rust::Str table_name);

}  // namespace cxx_gqe
