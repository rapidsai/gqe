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

#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

using gqe::qep::inject_valid_mask_as_nulls;
using gqe::qep::make_shared_state;
using gqe::qep::masked_table_view;
using gqe::qep::state_container;
using gqe::qep::state_container_view;
namespace state_kind = gqe::qep::state_kind;

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

// ---------------------------------------------------------------------------
// masked_table_view::try_from
// ---------------------------------------------------------------------------

/**
 * @brief An empty container has no valid_mask slot and cannot match the convention.
 */
TEST(MaskedTableViewTryFrom, EmptyContainerReturnsNullopt)
{
  state_container empty;
  EXPECT_FALSE(masked_table_view::try_from(state_container_view(empty)).has_value());
}

/**
 * @brief A lone `valid_mask` slot has already claimed the masked-table shape (slot 0 is a
 *        valid_mask), so having no data column to apply it to is a caller bug: throw rather than
 *        silently reject. Mirrors the strict-shape handling of `NonColumnDataSlotThrows`.
 */
TEST(MaskedTableViewTryFrom, LoneValidMaskSlotThrows)
{
  state_container c;
  c.push_back(
    make_shared_state(state_kind::valid_mask{bool_column_wrapper{true, false}.release()}));
  EXPECT_THROW(std::ignore = masked_table_view::try_from(state_container_view(c)),
               std::logic_error);
}

/**
 * @brief When slot 0 is not a `valid_mask`, the container is a plain table and `try_from` must
 *        reject it so the caller can take the non-masked path.
 */
TEST(MaskedTableViewTryFrom, FirstSlotNotValidMaskReturnsNullopt)
{
  state_container c;
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()}));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{4, 5, 6}.release()}));
  EXPECT_FALSE(masked_table_view::try_from(state_container_view(c)).has_value());
}

/**
 * @brief A `valid_mask` slot carrying a null inner column is a caller bug: the leading slot
 *        committed to the masked-table shape, so a malformed inner column must throw rather
 *        than be silently rejected.
 */
TEST(MaskedTableViewTryFrom, NullValidMaskColumnThrows)
{
  state_container c;
  c.push_back(make_shared_state(state_kind::valid_mask{nullptr}));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()}));
  EXPECT_THROW(std::ignore = masked_table_view::try_from(state_container_view(c)),
               std::logic_error);
}

/**
 * @brief A non-column slot (e.g., `row_count`) after the leading `valid_mask` is a caller
 *        bug: once the masked-table shape is claimed by slot 0, the rest of the container
 *        must be columns. Throw rather than silently reject.
 */
TEST(MaskedTableViewTryFrom, NonColumnDataSlotThrows)
{
  state_container c;
  c.push_back(
    make_shared_state(state_kind::valid_mask{bool_column_wrapper{true, true, false}.release()}));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()}));
  c.push_back(make_shared_state(state_kind::row_count{3}));
  EXPECT_THROW(std::ignore = masked_table_view::try_from(state_container_view(c)),
               std::logic_error);
}

/**
 * @brief The minimal happy path: `[valid_mask, cudf_column]` unpacks into a view whose
 *        `row_count`, `mask`, and single-column `columns` reference the original storage.
 */
TEST(MaskedTableViewTryFrom, SingleOwnedColumnSucceeds)
{
  state_container c;
  auto mask_col = bool_column_wrapper{true, false, true}.release();
  auto data_col = int64_column_wrapper{10, 20, 30}.release();

  auto const* mask_raw = mask_col.get();
  auto const* data_raw = data_col.get();

  c.push_back(make_shared_state(state_kind::valid_mask{std::move(mask_col)}));
  c.push_back(make_shared_state(state_kind::cudf_column{std::move(data_col)}));

  auto view = masked_table_view::try_from(state_container_view(c));
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 3);
  EXPECT_EQ(view->mask.size(), 3);
  EXPECT_EQ(view->mask.head(), mask_raw->view().head());
  ASSERT_EQ(view->columns.num_columns(), 1);
  EXPECT_EQ(view->columns.column(0).head(), data_raw->view().head());
}

/**
 * @brief Multiple data columns after the leading `valid_mask` all land in the returned
 *        `columns` table view in their original order.
 */
TEST(MaskedTableViewTryFrom, MultipleColumnsSucceeds)
{
  state_container c;
  c.push_back(make_shared_state(
    state_kind::valid_mask{bool_column_wrapper{true, false, true, true}.release()}));
  c.push_back(
    make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3, 4}.release()}));
  c.push_back(
    make_shared_state(state_kind::cudf_column{int64_column_wrapper{5, 6, 7, 8}.release()}));
  c.push_back(
    make_shared_state(state_kind::cudf_column{int64_column_wrapper{9, 10, 11, 12}.release()}));

  auto view = masked_table_view::try_from(state_container_view(c));
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 4);
  EXPECT_EQ(view->columns.num_columns(), 3);
  EXPECT_EQ(view->columns.num_rows(), 4);
}

/**
 * @brief Owned (`cudf_column`) and borrowed (`cudf_column_view`) slots may be freely mixed
 *        after the leading `valid_mask`. The named wrapper variable backs the borrowed view
 *        for the duration of the test.
 */
TEST(MaskedTableViewTryFrom, MixedOwnedAndBorrowedColumnsSucceeds)
{
  int64_column_wrapper borrowed_wrapper{100, 200, 300};
  cudf::column_view const borrowed_view = borrowed_wrapper;

  state_container c;
  c.push_back(
    make_shared_state(state_kind::valid_mask{bool_column_wrapper{true, true, false}.release()}));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()}));
  c.push_back(make_shared_state(state_kind::cudf_column_view{borrowed_view}));

  auto view = masked_table_view::try_from(state_container_view(c));
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 3);
  ASSERT_EQ(view->columns.num_columns(), 2);
  EXPECT_EQ(view->columns.column(1).head(), borrowed_view.head());
}

// ---------------------------------------------------------------------------
// inject_valid_mask_as_nulls
// ---------------------------------------------------------------------------

/**
 * @brief A non-BOOL8 mask is a programmer error and must fail loudly rather than producing a
 *        silently-wrong result.
 */
TEST(InjectValidMaskAsNulls, NonBoolMaskThrows)
{
  auto bad_mask = int64_column_wrapper{1, 0, 1}.release();
  auto column   = int64_column_wrapper{10, 20, 30}.release();
  cudf::table_view const columns_view({column->view()});

  EXPECT_THROW(std::ignore = inject_valid_mask_as_nulls(bad_mask->view(), columns_view),
               std::logic_error);
}

/**
 * @brief A mask with a different row count than the columns is a programmer error.
 */
TEST(InjectValidMaskAsNulls, MaskRowCountMismatchThrows)
{
  auto mask   = bool_column_wrapper{true, false}.release();
  auto column = int64_column_wrapper{10, 20, 30}.release();
  cudf::table_view const columns_view({column->view()});

  EXPECT_THROW(std::ignore = inject_valid_mask_as_nulls(mask->view(), columns_view),
               std::logic_error);
}

/**
 * @brief A `false` (or null) bit in the mask becomes a null bit in every output column at the
 *        same row index.
 */
TEST(InjectValidMaskAsNulls, MasksOutFalseRows)
{
  auto mask   = bool_column_wrapper{true, false, true}.release();
  auto column = int64_column_wrapper{10, 20, 30}.release();
  cudf::table_view const columns_view({column->view()});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 1u);
  auto const& result = injection.columns[0];
  ASSERT_TRUE(result.nullable());
  EXPECT_EQ(result.null_count(), 1);

  // Null-row payloads are don't-cares: COLUMNS_EQUIVALENT treats two nulls as equal without
  // reading their values, so the sentinel below is never compared. (Other tests follow suit.)
  int64_column_wrapper const expected({10, -999, 30}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result, expected);
}

/**
 * @brief When the mask is all-true, existing nulls in the input column must be preserved
 *        (the AND-merge cannot revive a null).
 */
TEST(InjectValidMaskAsNulls, PreservesExistingNulls)
{
  auto input_col = int64_column_wrapper({0, 20, 30}, {false, true, true}).release();
  auto mask      = bool_column_wrapper{true, true, true}.release();
  cudf::table_view const columns_view({input_col->view()});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 1u);
  auto const& result = injection.columns[0];
  ASSERT_TRUE(result.nullable());
  EXPECT_EQ(result.null_count(), 1);

  // Null-row payload is a don't-care (see MasksOutFalseRows).
  int64_column_wrapper const expected({-999, 20, 30}, {false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result, expected);
}

/**
 * @brief AND-merge semantics: the union of existing-null rows and mask-false rows becomes the
 *        null set of the output column.
 */
TEST(InjectValidMaskAsNulls, CombinesMaskWithExistingNulls)
{
  auto input_col = int64_column_wrapper({0, 20, 30}, {false, true, true}).release();
  auto mask      = bool_column_wrapper{true, false, true}.release();
  cudf::table_view const columns_view({input_col->view()});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 1u);
  auto const& result = injection.columns[0];
  EXPECT_EQ(result.null_count(), 2);

  // Null-row payloads are don't-cares (see MasksOutFalseRows); only the valid row carries a value.
  int64_column_wrapper const expected({-999, -999, 30}, {false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result, expected);
}

/**
 * @brief The same mask is applied independently to every input column.
 */
TEST(InjectValidMaskAsNulls, AppliesAcrossMultipleColumns)
{
  auto mask = bool_column_wrapper{true, false, true}.release();
  auto c0   = int64_column_wrapper{10, 20, 30}.release();
  auto c1   = int64_column_wrapper{40, 50, 60}.release();
  cudf::table_view const columns_view({c0->view(), c1->view()});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 2u);
  EXPECT_EQ(injection.columns[0].null_count(), 1);
  EXPECT_EQ(injection.columns[1].null_count(), 1);

  // Null-row payloads are don't-cares (see MasksOutFalseRows).
  int64_column_wrapper const expected0({10, -999, 30}, {true, false, true});
  int64_column_wrapper const expected1({40, -999, 60}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(injection.columns[0], expected0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(injection.columns[1], expected1);
}

/**
 * @brief Nested-type columns (strings, lists, structs) carry data in child columns. The
 *        reconstructed `column_view` must preserve them; otherwise the row data is lost.
 *
 * Uses an all-`true` mask so the rebuild doesn't introduce nulls — the equivalence check
 * then catches any dropped children directly via mismatched data.
 */
TEST(InjectValidMaskAsNulls, PreservesColumnChildren)
{
  auto mask    = bool_column_wrapper{true, true, true}.release();
  auto strings = cudf::test::strings_column_wrapper{"alpha", "beta", "gamma"}.release();
  ASSERT_GT(strings->view().num_children(), 0)
    << "string columns must have at least one child to make this test meaningful";

  cudf::table_view const columns_view({strings->view()});
  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 1u);
  auto const& result = injection.columns[0];
  EXPECT_EQ(result.num_children(), strings->view().num_children());

  cudf::test::strings_column_wrapper const expected{"alpha", "beta", "gamma"};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result, expected);
}

/**
 * @brief A sliced (non-zero `offset()`) fixed-width column must still mask out the correct rows.
 *
 * The merged null mask is zero-based, so the rebuilt view must fold the slice offset into the
 * data pointer and reset `offset()` to 0. A regression would read null bits at `offset + i`
 * instead of `i`, mismatching the data (and reading past the merged mask).
 */
TEST(InjectValidMaskAsNulls, MasksOutFalseRowsOnSlicedColumn)
{
  auto base   = int64_column_wrapper{10, 20, 30, 40, 50}.release();
  auto sliced = cudf::slice(base->view(), {1, 4}).front();  // {20, 30, 40}, offset 1
  ASSERT_EQ(sliced.offset(), 1);

  auto mask = bool_column_wrapper{true, false, true}.release();
  cudf::table_view const columns_view({sliced});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 1u);
  auto const& result = injection.columns[0];
  EXPECT_EQ(result.offset(), 0);
  EXPECT_EQ(result.null_count(), 1);

  // Null-row payload is a don't-care (see MasksOutFalseRows).
  int64_column_wrapper const expected({20, -999, 40}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result, expected);
}

/**
 * @brief On a sliced column, the existing-null read must also honor the offset: `bitmask_and`
 *        reads the original mask at `begin_bit = offset`, and the result is the union of
 *        existing-null and mask-false rows over the *logical* rows.
 */
TEST(InjectValidMaskAsNulls, CombinesMaskWithExistingNullsOnSlicedColumn)
{
  // Full validity: {T, F, T, T, T}; slice [1, 4) -> rows {20(null), 30, 40}.
  auto base = int64_column_wrapper({10, 20, 30, 40, 50}, {true, false, true, true, true}).release();
  auto sliced = cudf::slice(base->view(), {1, 4}).front();
  ASSERT_EQ(sliced.offset(), 1);

  auto mask = bool_column_wrapper{true, true, false}.release();  // masks out logical row 2
  cudf::table_view const columns_view({sliced});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 1u);
  auto const& result = injection.columns[0];
  EXPECT_EQ(result.null_count(), 2);

  // row 0: existing null; row 1: valid; row 2: masked out. Null-row payloads are don't-cares
  // (see MasksOutFalseRows), so only the valid row carries a real value.
  int64_column_wrapper const expected({-999, 30, -999}, {false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result, expected);
}

/**
 * @brief A sliced strings column must be masked without losing its rows. Strings store chars in
 *        the parent data buffer (addressed by absolute offset values) and the row offsets in a
 *        child column; the offset fold advances the offsets-child head, not the parent head. A
 *        regression here would read the wrong strings or corrupt the column.
 *
 * Masking a string row only flips its null bit -- the row's chars stay in place -- so the
 * masked-out row is a "non-empty null". That is harmless for null-propagating compute (the AST
 * evaluator never reads a null row's value), but cuDF's column comparator rejects it. We assert
 * the non-empty null is present, then purge it to empty; purging also reads each valid row's
 * chars through the folded offsets, so the comparison confirms the fold resolved them correctly.
 */
TEST(InjectValidMaskAsNulls, MasksAndPreservesSlicedStringColumn)
{
  auto base =
    cudf::test::strings_column_wrapper{"alpha", "beta", "gamma", "delta", "epsilon"}.release();
  auto sliced = cudf::slice(base->view(), {1, 4}).front();  // {"beta", "gamma", "delta"}, offset 1
  ASSERT_EQ(sliced.offset(), 1);

  auto mask = bool_column_wrapper{true, false, true}.release();
  cudf::table_view const columns_view({sliced});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);

  ASSERT_EQ(injection.columns.size(), 1u);
  auto const& result = injection.columns[0];
  EXPECT_EQ(result.offset(), 0);
  EXPECT_EQ(result.null_count(), 1);
  EXPECT_TRUE(cudf::has_nonempty_nulls(result));

  // Purging sanitizes the masked-out row to an empty string (its payload is a don't-care; see
  // MasksOutFalseRows) so the comparator accepts the column.
  auto const purged = cudf::purge_nonempty_nulls(result);
  cudf::test::strings_column_wrapper const expected({"beta", "", "delta"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(purged->view(), expected);
}

/**
 * @brief End-to-end with the real consumer: feeding injected columns to `cudf::compute_column`
 *        (the AST evaluator) makes masked-out rows propagate to null in the result. Uses a sliced
 *        (non-zero offset) input so the offset fold is exercised through the evaluator itself.
 */
TEST(InjectValidMaskAsNulls, InjectedColumnsPropagateNullsThroughAst)
{
  auto base   = int64_column_wrapper{10, 20, 30, 40, 50}.release();
  auto sliced = cudf::slice(base->view(), {1, 4}).front();  // {20, 30, 40}, offset 1
  auto mask   = bool_column_wrapper{true, false, true}.release();
  cudf::table_view const columns_view({sliced});

  auto injection   = inject_valid_mask_as_nulls(mask->view(), columns_view);
  auto const table = injection.as_table_view();

  // Evaluate `col0 + 5`. The masked-out row is null on input, so the result is null there too.
  auto five             = cudf::numeric_scalar<int64_t>(5);
  auto const literal    = cudf::ast::literal(five);
  auto const col_ref    = cudf::ast::column_reference(0);
  auto const expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref, literal);
  auto const result     = cudf::compute_column(table, expression);

  EXPECT_EQ(result->null_count(), 1);
  // row 1's payload is a don't-care (null); sentinel makes that explicit.
  int64_column_wrapper const expected({25, -999, 45}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

/**
 * @brief Same end-to-end check for strings. A masked-out string row is a non-empty null; the AST
 *        evaluator must treat it as null (propagate) rather than read its underlying chars, and
 *        the offset fold must resolve the valid rows' chars correctly for the comparison.
 */
TEST(InjectValidMaskAsNulls, InjectedStringColumnsPropagateNullsThroughAst)
{
  auto base =
    cudf::test::strings_column_wrapper{"alpha", "beta", "gamma", "delta", "epsilon"}.release();
  auto sliced = cudf::slice(base->view(), {1, 4}).front();  // {"beta", "gamma", "delta"}, offset 1
  auto mask   = bool_column_wrapper{true, false, true}.release();  // masks out "gamma"
  cudf::table_view const columns_view({sliced});

  auto injection   = inject_valid_mask_as_nulls(mask->view(), columns_view);
  auto const table = injection.as_table_view();

  // Evaluate `col0 == "beta"`.
  auto needle           = cudf::string_scalar("beta");
  auto const literal    = cudf::ast::literal(needle);
  auto const col_ref    = cudf::ast::column_reference(0);
  auto const expression = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);
  auto const result     = cudf::compute_column(table, expression);

  EXPECT_EQ(result->null_count(), 1);
  // row 0: "beta" == "beta" -> true; row 1: masked -> null (payload don't-care); row 2: false.
  bool_column_wrapper const expected({true, false, false}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

/**
 * @brief `as_table_view()` exposes the injected columns as a `cudf::table_view` so downstream
 *        compute (e.g., the AST evaluator) can consume them without further bookkeeping.
 */
TEST(InjectValidMaskAsNulls, AsTableViewExposesAllColumns)
{
  auto mask = bool_column_wrapper{true, true, true}.release();
  auto c0   = int64_column_wrapper{1, 2, 3}.release();
  auto c1   = int64_column_wrapper{4, 5, 6}.release();
  cudf::table_view const columns_view({c0->view(), c1->view()});

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view);
  auto const tv  = injection.as_table_view();
  EXPECT_EQ(tv.num_columns(), 2);
  EXPECT_EQ(tv.num_rows(), 3);
}
