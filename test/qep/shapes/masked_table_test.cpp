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
#include <vector>

using gqe::qep::inject_valid_mask_as_nulls;
using gqe::qep::make_shared_state;
using gqe::qep::masked_table_view;
using gqe::qep::shared_state;
using gqe::qep::state_container;
using gqe::qep::state_container_builder;
using gqe::qep::state_container_view;
namespace state_kind = gqe::qep::state_kind;

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

// ---------------------------------------------------------------------------
// state_container_builder
// ---------------------------------------------------------------------------

TEST(StateContainerBuilder, PlacesOwnedValidMaskBeforePreviouslyAddedColumns)
{
  auto first  = make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2}.release()},
                                 gqe::query_stream::default_stream());
  auto second = make_shared_state(state_kind::cudf_column{int64_column_wrapper{3, 4}.release()},
                                  gqe::query_stream::default_stream());

  auto output = state_container_builder()
                  .add_state(first)
                  .add_state(state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
                             gqe::query_stream::default_stream())
                  .add_state(second)
                  .build();

  ASSERT_EQ(output.size(), 3);
  EXPECT_TRUE(std::holds_alternative<state_kind::valid_mask>(*output[0]));
  EXPECT_EQ(output[1], first);
  EXPECT_EQ(output[2], second);
}

TEST(StateContainerBuilder, PlacesForwardedBorrowedValidMaskFirst)
{
  bool_column_wrapper mask{true, false};
  shared_state forwarded_mask = make_shared_state(state_kind::valid_mask_view{mask});
  auto column = make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2}.release()},
                                  gqe::query_stream::default_stream());

  auto output = state_container_builder().add_state(column).add_state(forwarded_mask).build();

  ASSERT_EQ(output.size(), 2);
  EXPECT_EQ(output[0], forwarded_mask);
  EXPECT_EQ(output[1], column);
}

TEST(StateContainerBuilder, RejectsDuplicateValidMasks)
{
  bool_column_wrapper borrowed_mask{true, false};
  state_container_builder builder;
  builder.add_state(state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
                    gqe::query_stream::default_stream());

  EXPECT_THROW(builder.add_state(state_kind::valid_mask_view{borrowed_mask}),
               std::invalid_argument);
}

TEST(StateContainerBuilder, CanBeReusedAfterBuild)
{
  state_container_builder builder;
  auto first = builder
                 .add_state(state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
                            gqe::query_stream::default_stream())
                 .add_state(state_kind::cudf_column{int64_column_wrapper{1, 2}.release()},
                            gqe::query_stream::default_stream())
                 .build();
  auto second = builder
                  .add_state(state_kind::valid_mask_view{
                    std::get<state_kind::valid_mask>(*first.front()).column().view()})
                  .add_state(state_kind::cudf_column{int64_column_wrapper{3, 4}.release()},
                             gqe::query_stream::default_stream())
                  .build();

  ASSERT_EQ(second.size(), 2);
  EXPECT_TRUE(std::holds_alternative<state_kind::valid_mask_view>(*second[0]));
  EXPECT_TRUE(std::holds_alternative<state_kind::cudf_column>(*second[1]));
}

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
 *        silently reject.
 */
TEST(MaskedTableViewTryFrom, LoneValidMaskSlotThrows)
{
  state_container c;
  c.push_back(make_shared_state(state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
                                gqe::query_stream::default_stream()));
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
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()},
                                gqe::query_stream::default_stream()));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{4, 5, 6}.release()},
                                gqe::query_stream::default_stream()));
  EXPECT_FALSE(masked_table_view::try_from(state_container_view(c)).has_value());
}

TEST(ValidMaskState, RejectsNullAndNonBoolColumnsAtConstruction)
{
  EXPECT_THROW(std::ignore = make_shared_state(state_kind::valid_mask{nullptr},
                                               gqe::query_stream::default_stream()),
               std::invalid_argument);
  EXPECT_THROW(
    std::ignore = make_shared_state(state_kind::valid_mask{int64_column_wrapper{1, 0}.release()},
                                    gqe::query_stream::default_stream()),
    std::invalid_argument);
}

TEST(ValidMaskViewState, RejectsNonBoolColumnsAtConstruction)
{
  int64_column_wrapper mask{1};
  EXPECT_THROW(std::ignore = make_shared_state(state_kind::valid_mask_view{mask}),
               std::invalid_argument);
}

TEST(MaskedTableViewTryFrom, UnsupportedDataKindReturnsNullopt)
{
  auto mask = make_shared_state(state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
                                gqe::query_stream::default_stream());
  auto column = make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2}.release()},
                                  gqe::query_stream::default_stream());
  auto unsupported = make_shared_state(state_kind::row_count{2});
  state_container input{mask, column, unsupported};
  EXPECT_FALSE(masked_table_view::try_from(input).has_value());
}

TEST(MaskedTableViewTryFrom, NullDataStorageThrows)
{
  state_container null_slot;
  null_slot.push_back(make_shared_state(state_kind::valid_mask{bool_column_wrapper{true}.release()},
                                        gqe::query_stream::default_stream()));
  null_slot.push_back(nullptr);
  EXPECT_THROW(std::ignore = masked_table_view::try_from(state_container_view(null_slot)),
               std::logic_error);

  state_container null_owned;
  null_owned.push_back(
    make_shared_state(state_kind::valid_mask{bool_column_wrapper{true}.release()},
                      gqe::query_stream::default_stream()));
  null_owned.push_back(
    make_shared_state(state_kind::cudf_column{nullptr}, gqe::query_stream::default_stream()));
  EXPECT_THROW(std::ignore = masked_table_view::try_from(state_container_view(null_owned)),
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

  auto const* mask_head = mask_col->view().head();
  auto const* data_raw  = data_col.get();

  c.push_back(make_shared_state(state_kind::valid_mask{std::move(mask_col)},
                                gqe::query_stream::default_stream()));
  c.push_back(make_shared_state(state_kind::cudf_column{std::move(data_col)},
                                gqe::query_stream::default_stream()));

  auto view = masked_table_view::try_from(state_container_view(c));
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 3);
  EXPECT_EQ(view->mask.size(), 3);
  EXPECT_EQ(view->mask.head(), mask_head);
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
    state_kind::valid_mask{bool_column_wrapper{true, false, true, true}.release()},
    gqe::query_stream::default_stream()));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3, 4}.release()},
                                gqe::query_stream::default_stream()));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{5, 6, 7, 8}.release()},
                                gqe::query_stream::default_stream()));
  c.push_back(
    make_shared_state(state_kind::cudf_column{int64_column_wrapper{9, 10, 11, 12}.release()},
                      gqe::query_stream::default_stream()));

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
    make_shared_state(state_kind::valid_mask{bool_column_wrapper{true, true, false}.release()},
                      gqe::query_stream::default_stream()));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()},
                                gqe::query_stream::default_stream()));
  c.push_back(make_shared_state(state_kind::cudf_column_view{borrowed_view}));

  auto view = masked_table_view::try_from(state_container_view(c));
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 3);
  ASSERT_EQ(view->columns.num_columns(), 2);
  EXPECT_EQ(view->columns.column(1).head(), borrowed_view.head());
}

TEST(MaskedTableViewTryFrom, MaskRowCountMismatchThrows)
{
  state_container c;
  c.push_back(make_shared_state(state_kind::valid_mask{bool_column_wrapper{true}.release()},
                                gqe::query_stream::default_stream()));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2}.release()},
                                gqe::query_stream::default_stream()));

  EXPECT_THROW(std::ignore = masked_table_view::try_from(state_container_view(c)),
               std::logic_error);
}

TEST(MaskedTableViewTryFrom, BorrowedMaskMatchesOwnedMask)
{
  bool_column_wrapper borrowed_mask{true, false, true};
  int64_column_wrapper borrowed_column{10, 20, 30};
  auto owned =
    state_container_builder()
      .add_state(state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(state_kind::cudf_column{int64_column_wrapper{10, 20, 30}.release()},
                 gqe::query_stream::default_stream())
      .build();
  auto borrowed = state_container_builder()
                    .add_state(state_kind::valid_mask_view{borrowed_mask})
                    .add_state(state_kind::cudf_column_view{borrowed_column})
                    .build();

  auto const owned_view    = masked_table_view::try_from(owned);
  auto const borrowed_view = masked_table_view::try_from(borrowed);
  ASSERT_TRUE(owned_view.has_value());
  ASSERT_TRUE(borrowed_view.has_value());
  EXPECT_EQ(gqe::qep::get_row_count(borrowed), 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(owned_view->mask, borrowed_view->mask);
  CUDF_TEST_EXPECT_TABLES_EQUAL(owned_view->columns, borrowed_view->columns);
  EXPECT_FALSE(gqe::qep::to_table_view(borrowed).has_value());
}

TEST(MaskedTableViewTryFrom, BorrowedMaskSupportsMixedDataColumns)
{
  bool_column_wrapper mask{true, true, false};
  int64_column_wrapper borrowed_column{100, 200, 300};
  auto input = state_container_builder()
                 .add_state(state_kind::valid_mask_view{mask})
                 .add_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()},
                            gqe::query_stream::default_stream())
                 .add_state(state_kind::cudf_column_view{borrowed_column})
                 .build();

  auto const view = masked_table_view::try_from(input);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 3);
  EXPECT_EQ(view->columns.num_columns(), 2);
  EXPECT_EQ(view->columns.column(1).head(), cudf::column_view(borrowed_column).head());
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

TEST(InjectValidMaskAsNulls, SelectiveIndicesAreDeduplicated)
{
  auto mask = bool_column_wrapper{true, false, true}.release();
  auto c0   = int64_column_wrapper{10, 20, 30}.release();
  auto c1   = int64_column_wrapper{40, 50, 60}.release();
  auto c2   = int64_column_wrapper{70, 80, 90}.release();
  cudf::table_view const columns_view({c0->view(), c1->view(), c2->view()});
  std::vector<cudf::size_type> const selected{2, 0, 2};

  auto injection = inject_valid_mask_as_nulls(mask->view(), columns_view, selected);

  ASSERT_EQ(injection.columns.size(), 3u);
  EXPECT_EQ(injection.per_column_null_masks.size(), 2u);
  EXPECT_EQ(injection.columns[1].head(), c1->view().head());
  EXPECT_EQ(injection.columns[1].null_mask(), c1->view().null_mask());
  EXPECT_EQ(injection.columns[0].null_count(), 1);
  EXPECT_EQ(injection.columns[2].null_count(), 1);
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

TEST(InjectValidMaskAsNulls, RejectsSlicedNestedColumn)
{
  cudf::test::lists_column_wrapper<int64_t> owner{{1, 2}, {3}, {4, 5}, {6}};
  auto const sliced = cudf::slice(owner, {1, 3}).front();
  ASSERT_NE(sliced.offset(), 0);

  auto mask = bool_column_wrapper{true, false}.release();
  cudf::table_view const columns_view({sliced});

  EXPECT_THROW(std::ignore = inject_valid_mask_as_nulls(mask->view(), columns_view),
               std::logic_error);
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
