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

// Which form gets chosen and how big it is: an answer is correct under every form, so the tests in
// unique_indices_tests.cpp cannot say which one ran.

#include <gqe/executor/perfect_hash/detail/xor_hash_table.cuh>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using gqe::perfect_hash::xor_hash_table::TableForm;
using Hasher = gqe::perfect_hash::xor_hash_table::IdentityHasher<>;
using Table  = gqe::perfect_hash::xor_hash_table::HashTable<int32_t, Hasher>;

constexpr int64_t KEYS = 145'000'000;  // Roughly what a 2^31 plan holds in the traces.

gqe::perfect_hash::xor_hash_table::TablePlan plan(int bits, int64_t keys = KEYS)
{
  return Table::plan_table(keys, Hasher(bits));
}

int64_t flat_bytes(int bits) { return (int64_t{1} << bits) * int64_t{sizeof(int32_t)}; }

int64_t sparse_bytes(int bits, int64_t keys = KEYS)
{
  return gqe::perfect_hash::sparse_table::byte_count(int64_t{1} << bits, keys, sizeof(int32_t));
}

// Every perfect key space up to the cap is stored sparsely, from a group key's few dozen slots up.
TEST(table_plan_test, perfect_key_spaces_up_to_the_cap_are_sparse)
{
  for (int bits : {5, 9, 16, 22, 23, 28, 31, 32, 33}) {
    // Five percent occupancy, so that the key count stays a possible one at every width.
    auto const keys = (int64_t{1} << bits) / 20 + 1;
    auto const p    = plan(bits, keys);
    EXPECT_EQ(p.form, TableForm::Sparse) << "at 2^" << bits;
    EXPECT_EQ(p.key_space, int64_t{1} << bits) << "at 2^" << bits;
    // Sized by the key count, which is the point: it does not grow with the key space.
    EXPECT_EQ(p.values, keys) << "at 2^" << bits;
  }
}

// unique_indices takes whole rows, so keys_numel is the row count and the key space is the bound.
TEST(table_plan_test, a_low_cardinality_key_is_sized_by_the_key_space)
{
  constexpr int64_t ROWS = 400'000'000;  // A partition of lineitem, near enough.

  for (int bits : {5, 9, 16, 22, 23, 24, 28}) {
    auto const slots = int64_t{1} << bits;
    auto const p     = plan(bits, ROWS);
    EXPECT_EQ(p.form, TableForm::Sparse) << "at 2^" << bits;
    EXPECT_EQ(p.values, slots) << "at 2^" << bits;
    EXPECT_LT(p.values, ROWS) << "at 2^" << bits;
  }

  // What that minimum bounds: with values capped at the key space, full occupancy costs 1.042x.
  for (int bits : {23, 28, 31, 33}) {
    auto const worst = sparse_bytes(bits, int64_t{1} << bits);  // Every slot present.
    EXPECT_LE(worst, flat_bytes(bits) * 1042 / 1000) << "at 2^" << bits;
  }
}

// The two ways to have no perfect form: past the slot cap, or not injective.
TEST(table_plan_test, probing_is_reached_past_the_cap_or_without_injectivity)
{
  EXPECT_EQ(gqe::perfect_hash::xor_hash_table::max_perfect_slots, int64_t{1} << 33);
  EXPECT_EQ(plan(33).form, TableForm::Sparse);
  EXPECT_EQ(plan(34).form, TableForm::Probing);

  // Probing is sized by the key count, not the key space, so it carries no bitmap.
  EXPECT_EQ(plan(34).key_space, 0);
  EXPECT_GT(plan(34).values, KEYS);

  // And a hasher that is not injective in 64 bits offers nothing perfect whatever its width.
  EXPECT_FALSE(Hasher(65).is_perfect());
  EXPECT_EQ(plan(65).form, TableForm::Probing);
}

// The sizing claim computed: 4 bytes per slot before, 1.33 bits plus 4 bytes per key after.
TEST(table_plan_test, sparse_is_smaller_at_the_occupancy_these_tables_run_at)
{
  for (int bits : {28, 31, 32, 33}) {
    auto const slots = int64_t{1} << bits;
    EXPECT_LT(sparse_bytes(bits, slots / 20), flat_bytes(bits)) << "at 2^" << bits;
  }

  // The two sizes the issue quotes, pinned so a change to either is deliberate.
  EXPECT_EQ(flat_bytes(31), int64_t{8} << 30);
  EXPECT_EQ(flat_bytes(33), int64_t{32} << 30);

  // One sector per 192 slots, and nothing else, once the values are taken out.
  auto const bitmap_only = sparse_bytes(31, 0);
  EXPECT_EQ(bitmap_only, gqe::perfect_hash::sparse_table::block_count(int64_t{1} << 31) * 32);
  EXPECT_LT(bitmap_only, int64_t{350} << 20);  // 341 MiB at 2^31.
}

}  // namespace
