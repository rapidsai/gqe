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

#include <gqe/qep/operator/fetch.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

/** @brief Build a plain single-column `int64` table container. */
gqe::qep::state_container one_column_container(std::vector<int64_t> values)
{
  return gqe::qep::state_container_builder()
    .add_state(
      gqe::qep::state_kind::cudf_column{
        int64_column_wrapper(values.begin(), values.end()).release()},
      gqe::query_stream::default_stream())
    .build();
}

/** @brief Build a masked single-column `int64` table container `[valid_mask, column]`. */
gqe::qep::state_container masked_container(std::vector<int64_t> values, std::vector<bool> valid)
{
  return gqe::qep::state_container_builder()
    .add_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper(valid.begin(), valid.end()).release()},
      gqe::query_stream::default_stream())
    .add_state(
      gqe::qep::state_kind::cudf_column{
        int64_column_wrapper(values.begin(), values.end()).release()},
      gqe::query_stream::default_stream())
    .build();
}

/** @brief Build a single-column indirect table whose position list selects rows of `source`. */
gqe::qep::state_container indirect_container(std::vector<int64_t> source,
                                             std::vector<cudf::size_type> positions)
{
  auto source_columns = gqe::qep::state_container_builder()
                          .add_state(
                            gqe::qep::state_kind::cudf_column{
                              int64_column_wrapper(source.begin(), source.end()).release()},
                            gqe::query_stream::default_stream())
                          .build();
  auto relation      = gqe::qep::indirect_table_source_state::make(std::move(source_columns));
  auto gather_source = gqe::qep::indirect_gather_source_state::make(
    std::move(relation),
    gqe::qep::make_position_list(int32_column_wrapper(positions.begin(), positions.end()).release(),
                                 gqe::query_stream::default_stream()));
  return {gqe::qep::indirect_column_state::make(std::move(gather_source), 0)};
}

/** @brief Build a masked indirect single-column logical relation. */
gqe::qep::state_container masked_indirect_container(std::vector<int64_t> source,
                                                    std::vector<cudf::size_type> positions,
                                                    std::vector<bool> valid)
{
  auto indirect = indirect_container(std::move(source), std::move(positions));
  gqe::qep::state_container_builder builder;
  builder.add_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper(valid.begin(), valid.end()).release()},
    gqe::query_stream::default_stream());
  builder.add_state(indirect.front());
  return builder.build();
}

class QepFetchTaskTest : public gqe::test::BaseFixture {
 protected:
  // The members are protected because every TEST_F body runs in a class derived from this
  // fixture and reads `ctx_ref`.
  QepFetchTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::context_reference ctx_ref;
};

/** @brief Negative bounds and an overflowing window are rejected at construction. */
TEST_F(QepFetchTaskTest, RejectsInvalidConstruction)
{
  EXPECT_THROW(gqe::qep::fetch_task(-1, 1), std::logic_error);
  EXPECT_THROW(gqe::qep::fetch_task(1, -1), std::logic_error);
  EXPECT_THROW(gqe::qep::fetch_task(1, std::numeric_limits<int64_t>::max()), std::overflow_error);
}

/** @brief A chunk entirely inside the window passes its state slots through without copying. */
TEST_F(QepFetchTaskTest, ChunkInsideWindowPassesStateSlotsThrough)
{
  gqe::qep::fetch_task const task{/*offset=*/0, /*count=*/10};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const input       = one_column_container({1, 2, 3});
  auto const* input_slot = input[0].get();
  auto const result      = task.next(input, accumulator, ctx_ref);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ((*result)[0].get(), input_slot);
}

/** @brief A chunk crossing the end of the window emits only the in-window rows. */
TEST_F(QepFetchTaskTest, TruncatesChunkCrossingTheQuota)
{
  gqe::qep::fetch_task const task{/*offset=*/0, /*count=*/3};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const result = task.next(one_column_container({1, 2, 3, 4, 5}), accumulator, ctx_ref);

  ASSERT_TRUE(result.has_value());
  auto const table = gqe::qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int64_column_wrapper({1, 2, 3}));
}

/** @brief The offset skips leading rows within a chunk. */
TEST_F(QepFetchTaskTest, SkipsOffsetRowsWithinAChunk)
{
  gqe::qep::fetch_task const task{/*offset=*/2, /*count=*/2};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const result = task.next(one_column_container({1, 2, 3, 4, 5}), accumulator, ctx_ref);

  ASSERT_TRUE(result.has_value());
  auto const table = gqe::qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int64_column_wrapper({3, 4}));
}

/**
 * @brief A window spanning several chunks slices the boundary chunk, passes the inside chunk
 *        through, and early-exits past the window.
 *
 * The window is `[2, 6)` over the stream. Chunk 1 covers `[0, 3)` and emits a boundary slice.
 * Chunk 2 covers `[3, 6)` entirely inside, so its state slots pass through. Chunk 3 covers
 * `[6, 9)` and returns the early-exit signal.
 */
TEST_F(QepFetchTaskTest, WindowSpansChunksWithPassThroughMiddleChunk)
{
  gqe::qep::fetch_task const task{/*offset=*/2, /*count=*/4};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const first = task.next(one_column_container({1, 2, 3}), accumulator, ctx_ref);
  ASSERT_TRUE(first.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(*first)->column(0),
                                 int64_column_wrapper({3}));

  auto const middle_input = one_column_container({4, 5, 6});
  auto const* middle_slot = middle_input[0].get();
  auto const middle       = task.next(middle_input, accumulator, ctx_ref);
  ASSERT_TRUE(middle.has_value());
  EXPECT_EQ((*middle)[0].get(), middle_slot);

  EXPECT_FALSE(task.next(one_column_container({7, 8, 9}), accumulator, ctx_ref).has_value());
}

/** @brief A chunk entirely before the offset emits a schema-correct zero-row output. */
TEST_F(QepFetchTaskTest, ChunkBeforeOffsetEmitsSchemaCorrectZeroRowOutput)
{
  gqe::qep::fetch_task const task{/*offset=*/10, /*count=*/5};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const result = task.next(one_column_container({1, 2, 3}), accumulator, ctx_ref);

  ASSERT_TRUE(result.has_value());
  auto const table = gqe::qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->num_columns(), 1);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->column(0).type().id(), cudf::type_id::INT64);
}

/** @brief A zero-count fetch emits the schema exactly once, then early-exits. */
TEST_F(QepFetchTaskTest, ZeroCountEmitsSchemaOnceThenEarlyExits)
{
  gqe::qep::fetch_task const task{/*offset=*/0, /*count=*/0};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const first = task.next(one_column_container({1, 2, 3}), accumulator, ctx_ref);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(gqe::qep::to_table_view(*first)->num_rows(), 0);

  EXPECT_FALSE(task.next(one_column_container({4, 5}), accumulator, ctx_ref).has_value());
}

/** @brief Count-only chunks truncate the count against the window and then early-exit. */
TEST_F(QepFetchTaskTest, CountOnlyChunksTruncateTheCount)
{
  gqe::qep::fetch_task const task{/*offset=*/1, /*count=*/4};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const first = task.next(gqe::qep::make_row_count_container(3), accumulator, ctx_ref);
  ASSERT_TRUE(first.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*first), testing::Optional(2));

  auto const second = task.next(gqe::qep::make_row_count_container(3), accumulator, ctx_ref);
  ASSERT_TRUE(second.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*second), testing::Optional(2));

  EXPECT_FALSE(task.next(gqe::qep::make_row_count_container(3), accumulator, ctx_ref).has_value());
}

/**
 * @brief The emitted boundary slice is an owned copy, so it stays valid after the input container
 *        is destroyed.
 */
TEST_F(QepFetchTaskTest, BoundarySliceOwnsItsStorage)
{
  gqe::qep::fetch_task const task{/*offset=*/0, /*count=*/3};
  auto const accumulator = task.initialize({}, ctx_ref);

  std::optional<gqe::qep::state_container> result;
  {
    auto const input = one_column_container({1, 2, 3, 4, 5});
    result           = task.next(input, accumulator, ctx_ref);
  }

  ASSERT_TRUE(result.has_value());
  auto const table = gqe::qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int64_column_wrapper({1, 2, 3}));
}

/** @brief A masked input window counts only compacted surviving rows. */
TEST_F(QepFetchTaskTest, CompactsMaskedInputBeforeWindowIntersection)
{
  gqe::qep::fetch_task const task{/*offset=*/1, /*count=*/1};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const input  = masked_container({1, 2, 3}, {true, false, true});
  auto const result = task.next(input, accumulator, ctx_ref);
  ASSERT_TRUE(result.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(*result)->column(0),
                                 int64_column_wrapper{3});
}

/** @brief Indirect and masked-indirect inputs are fully compacted before slicing. */
TEST_F(QepFetchTaskTest, MaterializesIndirectAndMaskedIndirectInput)
{
  gqe::qep::fetch_task const task{/*offset=*/0, /*count=*/10};
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const input  = indirect_container({10, 20, 30}, {2, 0});
  auto const result = task.next(input, accumulator, ctx_ref);
  ASSERT_TRUE(result.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(*result)->column(0),
                                 int64_column_wrapper({30, 10}));

  gqe::qep::fetch_task const masked_task{/*offset=*/0, /*count=*/10};
  auto const masked_accumulator = masked_task.initialize({}, ctx_ref);
  auto const masked_input = masked_indirect_container({10, 20, 30}, {2, 0, 1}, {true, false, true});
  auto const masked_result = masked_task.next(masked_input, masked_accumulator, ctx_ref);
  ASSERT_TRUE(masked_result.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(*masked_result)->column(0),
                                 int64_column_wrapper({30, 20}));
}

/**
 * @brief Sibling task clones share the window through the accumulator, as the executor's
 *        per-partition adapters do.
 */
TEST_F(QepFetchTaskTest, ClonedTasksShareTheAccumulatorQuota)
{
  gqe::qep::fetch_task const task{/*offset=*/0, /*count=*/4};
  auto const clone       = gqe::qep::clone_qep_task_as<gqe::qep::fetch_task>(task);
  auto const accumulator = task.initialize({}, ctx_ref);

  auto const first = task.next(one_column_container({1, 2, 3}), accumulator, ctx_ref);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(gqe::qep::to_table_view(*first)->num_rows(), 3);

  // The sibling clone works against the same accumulator, so it sees a remaining window of one.
  auto const second = clone->next(one_column_container({4, 5, 6}), accumulator, ctx_ref);
  ASSERT_TRUE(second.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(*second)->column(0),
                                 int64_column_wrapper({4}));

  EXPECT_FALSE(clone->next(one_column_container({7}), accumulator, ctx_ref).has_value());
}
