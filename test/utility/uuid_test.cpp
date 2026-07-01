/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/rpc/serialization/uuid.hpp>

#include <gqe/utility/uuid.hpp>

#include <gtest/gtest.h>

#include <boost/uuid/uuid.hpp>

#include <cstring>
#include <tuple>

namespace gqe::utility {
namespace {

constexpr auto test_uuid_str = "550e8400-e29b-41d4-a716-446655440000";

TEST(UuidTest, GenerateProducesValidUuid)
{
  auto id    = uuid::generate();
  auto bytes = id.bytes();

  boost::uuids::uuid boost_uuid;
  std::memcpy(&boost_uuid, bytes.data(), sizeof(boost_uuid));

  // RFC 4122 version nibble (byte 6, high nibble) must be 0b0100 for UUID v4.
  EXPECT_EQ(boost_uuid.version(), boost::uuids::uuid::version_random_number_based);
  // RFC 4122 variant bits (byte 8, bits 6-7) must be 0b10.
  EXPECT_EQ(boost_uuid.variant(), boost::uuids::uuid::variant_rfc_4122);
  EXPECT_FALSE(boost_uuid.is_nil());
}

TEST(UuidTest, StringRoundTrip)
{
  auto id  = uuid::from_string(test_uuid_str);
  auto str = id.to_string();
  EXPECT_EQ(str, test_uuid_str);
}

TEST(UuidTest, BytesRoundTrip)
{
  auto original = uuid::from_string(test_uuid_str);
  auto bytes    = original.bytes();
  auto restored = uuid::from_bytes(bytes);
  EXPECT_EQ(original, restored);
}

TEST(UuidTest, RpcSerializationRoundTrip)
{
  auto original = uuid::from_string(test_uuid_str);
  auto proto    = rpc::serialize_uuid(original);
  auto restored = rpc::deserialize_uuid(proto);
  EXPECT_EQ(original, restored);
}

TEST(UuidTest, FromStringRejectsInvalid)
{
  EXPECT_THROW(std::ignore = uuid::from_string("not-a-uuid"), std::invalid_argument);
  EXPECT_THROW(std::ignore = uuid::from_string(""), std::invalid_argument);
}

}  // namespace
}  // namespace gqe::utility
