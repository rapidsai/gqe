/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/qep/shapes/masked_table.hpp>

#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/traits.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

std::optional<masked_table_view> masked_table_view::try_from(state_container_view container)
{
  // Convention: [valid_mask, (cudf_column | cudf_column_view)+]
  if (container.size() == 0) { return std::nullopt; }

  auto const* mk = std::get_if<state_kind::valid_mask>(container[0].get());
  if (mk == nullptr) { return std::nullopt; }

  // The leading slot is a valid_mask, so the container claims masked-table shape. Anything
  // structurally inconsistent from here on is a caller bug, not a shape mismatch -- including a
  // lone valid_mask with no data columns to apply it to.
  GQE_EXPECTS(container.size() >= 2,
              "masked_table_view::try_from: valid_mask must be followed by at least one column");
  GQE_EXPECTS(mk->column != nullptr,
              "masked_table_view::try_from: valid_mask carries a null column");

  std::vector<cudf::column_view> columns;
  columns.reserve(container.size() - 1);
  for (std::size_t i = 1; i < container.size(); ++i) {
    auto const& slot = *container[i];
    if (auto const* owned = std::get_if<state_kind::cudf_column>(&slot)) {
      columns.push_back(owned->column->view());
    } else if (auto const* borrowed = std::get_if<state_kind::cudf_column_view>(&slot)) {
      columns.push_back(borrowed->column);
    } else {
      GQE_EXPECTS(false, "masked_table_view::try_from: non-column slot after leading valid_mask");
    }
  }

  return masked_table_view{
    mk->column->size(), mk->column->view(), cudf::table_view(std::move(columns))};
}

cudf::table_view null_mask_injection::as_table_view() const { return cudf::table_view(columns); }

namespace {

// Return a view over `col`'s data that shares its buffers but has `offset() == 0`.
//
// We need offset 0 because the merged null mask built by `bitmask_and` (below) is zero-based --
// row `i` at bit `i` -- while a `column_view`'s single `offset` is applied to BOTH its data and
// its null mask. Carrying `orig.offset()` onto the rebuilt view would re-apply the offset to the
// already-zero-based mask, reading the wrong (and out-of-bounds) bits.
//
// The fold is pure pointer arithmetic over the shared buffers; no data is copied:
//   * fixed-width: advance the data head by `offset` elements.
//   * strings:     the chars live in the parent data buffer and are addressed by absolute offset
//                  values, so the parent head is unchanged. Advance the offsets child's head
//                  instead -- this mirrors how `column_device_view::element<string_view>` (the AST
//                  evaluator's string accessor) indexes: `offsets.head()[parent.offset() + i]`.
//
// The returned view carries no null mask; the caller attaches the merged mask.
[[nodiscard]] cudf::column_view fold_offset_to_zero(cudf::column_view col)
{
  if (col.offset() == 0) { return col; }

  auto const type = col.type();
  GQE_EXPECTS(cudf::is_fixed_width(type) || type.id() == cudf::type_id::STRING,
              "inject_valid_mask_as_nulls: a non-zero column offset is only supported for "
              "fixed-width and string columns");

  if (cudf::is_fixed_width(type)) {
    auto const* head = static_cast<std::byte const*>(col.head()) +
                       static_cast<std::ptrdiff_t>(col.offset()) * cudf::size_of(type);
    return cudf::column_view(type, col.size(), head, nullptr, 0, /*offset=*/0, {});
  }

  // STRING: leave the parent (chars) head untouched and fold the offset into the offsets child.
  auto const offsets = col.child(cudf::strings_column_view::offsets_column_index);
  auto const* offsets_head =
    static_cast<std::byte const*>(offsets.head()) +
    static_cast<std::ptrdiff_t>(col.offset()) * cudf::size_of(offsets.type());
  // The offsets child spans the logical rows plus the trailing end offset: `size + 1` values.
  auto const folded_offsets =
    cudf::column_view(offsets.type(), col.size() + 1, offsets_head, nullptr, 0, /*offset=*/0, {});
  return cudf::column_view(
    type, col.size(), col.head(), nullptr, 0, /*offset=*/0, {folded_offsets});
}

}  // namespace

null_mask_injection inject_valid_mask_as_nulls(cudf::column_view valid_mask,
                                               cudf::table_view columns,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  GQE_EXPECTS(valid_mask.type().id() == cudf::type_id::BOOL8,
              "inject_valid_mask_as_nulls: valid_mask must be BOOL8");
  GQE_EXPECTS(valid_mask.size() == columns.num_rows(),
              "inject_valid_mask_as_nulls: valid_mask row count must match columns");

  // Convert the BOOL8 valid_mask to a bitmask. `false` and null both produce a clear bit, so
  // AND-merging this into each column's null mask makes downstream null-propagating compute
  // (e.g. the cuDF AST evaluator) emit null at masked-out rows.
  auto [mask_bitmask, valid_mask_null_count] = cudf::bools_to_mask(valid_mask, stream, mr);
  auto const* bitmask_ptr       = static_cast<cudf::bitmask_type const*>(mask_bitmask->data());
  auto const valid_mask_for_and = cudf::column_view(valid_mask.type(),
                                                    valid_mask.size(),
                                                    valid_mask.head(),
                                                    bitmask_ptr,
                                                    valid_mask_null_count,
                                                    /*offset=*/0,
                                                    {});

  null_mask_injection result;
  result.mask_bitmask = std::move(*mask_bitmask);
  result.per_column_null_masks.reserve(columns.num_columns());
  result.columns.reserve(columns.num_columns());

  for (cudf::size_type c = 0; c < columns.num_columns(); ++c) {
    auto const orig = columns.column(c);

    // `bitmask_and` reads each input at its `begin_bit` (consuming `orig.offset()` here) and writes
    // a fresh, zero-based mask. We therefore rebuild the view at offset 0, folding the original
    // offset into the data/child pointers so the data still resolves correctly. See
    // `fold_offset_to_zero`.
    std::vector<cudf::column_view> columns_to_and{orig, valid_mask_for_and};
    auto [combined, null_count] = cudf::bitmask_and(cudf::table_view(columns_to_and), stream, mr);
    result.per_column_null_masks.push_back(std::move(combined));

    auto const zeroed = fold_offset_to_zero(orig);
    std::vector<cudf::column_view> children(zeroed.child_begin(), zeroed.child_end());
    result.columns.emplace_back(
      zeroed.type(),
      zeroed.size(),
      zeroed.head(),
      static_cast<cudf::bitmask_type const*>(result.per_column_null_masks.back().data()),
      null_count,
      /*offset=*/0,
      children);
  }

  return result;
}

}  // namespace qep
}  // namespace gqe
