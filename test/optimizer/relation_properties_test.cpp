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

#include <gqe/optimizer/relation_properties.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <string>
#include <vector>

using props_t = gqe::optimizer::relation_properties;
using uk_t    = std::vector<cudf::size_type>;
using keys_t  = std::vector<uk_t>;

// ---------------------------------------------------------------------------
// add_unique_key
// ---------------------------------------------------------------------------

TEST(RelationPropertiesUniqueKeys, AddUniqueKeySorts)
{
  props_t props;
  props.add_unique_key({2, 0, 1});
  ASSERT_EQ(props.unique_keys().size(), 1u);
  EXPECT_EQ(props.unique_keys()[0], (uk_t{0, 1, 2}));
}

TEST(RelationPropertiesUniqueKeys, AddUniqueKeyIdempotent)
{
  props_t props;
  props.add_unique_key({0, 1});
  props.add_unique_key({0, 1});
  EXPECT_EQ(props.unique_keys().size(), 1u);
}

TEST(RelationPropertiesUniqueKeys, AddUniqueKeyIdempotentDifferentOrder)
{
  // {1, 0} and {0, 1} are the same key after sorting.
  props_t props;
  props.add_unique_key({1, 0});
  props.add_unique_key({0, 1});
  EXPECT_EQ(props.unique_keys().size(), 1u);
}

TEST(RelationPropertiesUniqueKeys, AddUniqueKeyDeduplicatesIndices)
{
  // {1, 1, 2} → dedup → {1, 2}
  props_t props;
  props.add_unique_key({1, 1, 2});
  ASSERT_EQ(props.unique_keys().size(), 1u);
  EXPECT_EQ(props.unique_keys()[0], (uk_t{1, 2}));
}

TEST(RelationPropertiesUniqueKeys, AddUniqueKeyEmptyIgnored)
{
  props_t props;
  props.add_unique_key({});
  EXPECT_TRUE(props.unique_keys().empty());
}

TEST(RelationPropertiesUniqueKeys, AddUniqueKeySingletonAndCompositeIndependent)
{
  props_t props;
  props.add_unique_key({0});
  props.add_unique_key({0, 1});
  EXPECT_EQ(props.unique_keys().size(), 2u);
}

// ---------------------------------------------------------------------------
// remove_unique_key
// ---------------------------------------------------------------------------

TEST(RelationPropertiesUniqueKeys, RemoveUniqueKeyRemoves)
{
  props_t props;
  props.add_unique_key({0, 1});
  props.add_unique_key({2});
  props.remove_unique_key({0, 1});
  ASSERT_EQ(props.unique_keys().size(), 1u);
  EXPECT_EQ(props.unique_keys()[0], (uk_t{2}));
}

TEST(RelationPropertiesUniqueKeys, RemoveUniqueKeyMissingNoOp)
{
  props_t props;
  props.add_unique_key({0, 1});
  props.remove_unique_key({99});
  EXPECT_EQ(props.unique_keys().size(), 1u);
}

TEST(RelationPropertiesUniqueKeys, RemoveUniqueKeyOrderInsensitive)
{
  props_t props;
  props.add_unique_key({0, 1, 2});
  props.remove_unique_key({2, 0, 1});  // unsorted form of the same key
  EXPECT_TRUE(props.unique_keys().empty());
}

// ---------------------------------------------------------------------------
// covers_unique_key
// ---------------------------------------------------------------------------

TEST(RelationPropertiesUniqueKeys, CoversUniqueKeyExactMatch)
{
  props_t props;
  props.add_unique_key({0, 1});
  EXPECT_TRUE(props.covers_unique_key({0, 1}));
}

TEST(RelationPropertiesUniqueKeys, CoversUniqueKeySuperset)
{
  props_t props;
  props.add_unique_key({0, 1});
  // A superset of a unique key-set still covers it.
  EXPECT_TRUE(props.covers_unique_key({0, 1, 2}));
}

TEST(RelationPropertiesUniqueKeys, CoversUniqueKeyPartialNotCovered)
{
  props_t props;
  props.add_unique_key({0, 1});
  // Covering only part of a composite key-set is not enough.
  EXPECT_FALSE(props.covers_unique_key({0}));
  EXPECT_FALSE(props.covers_unique_key({2, 3}));
}

TEST(RelationPropertiesUniqueKeys, CoversUniqueKeyNoKeysRegistered)
{
  props_t props;
  EXPECT_FALSE(props.covers_unique_key({0}));
}

// ---------------------------------------------------------------------------
// operator== considers unique keys
// ---------------------------------------------------------------------------

TEST(RelationPropertiesUniqueKeys, EqualityConsidersUniqueKeysDiffer)
{
  props_t a, b;
  a.add_unique_key({0, 1});
  EXPECT_NE(a, b);
}

TEST(RelationPropertiesUniqueKeys, EqualityConsidersUniqueKeysEqual)
{
  props_t a, b;
  a.add_unique_key({0, 1});
  b.add_unique_key({0, 1});
  EXPECT_EQ(a, b);
}

TEST(RelationPropertiesUniqueKeys, EqualityConsidersUniqueKeysDifferentKeys)
{
  props_t a, b;
  a.add_unique_key({0, 1});
  b.add_unique_key({1, 2});
  EXPECT_NE(a, b);
}

// ---------------------------------------------------------------------------
// to_string includes unique_keys when non-empty, omits when empty
// ---------------------------------------------------------------------------

TEST(RelationPropertiesUniqueKeys, ToStringIncludesUniqueKeysWhenPresent)
{
  props_t props;
  props.add_unique_key({0, 1});
  EXPECT_NE(props.to_string().find("unique_keys"), std::string::npos);
}

TEST(RelationPropertiesUniqueKeys, ToStringOmitsUniqueKeysWhenEmpty)
{
  props_t props;
  EXPECT_EQ(props.to_string().find("unique_keys"), std::string::npos);
}
