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

#include <gqe/types.hpp>

#include <gtest/gtest.h>

namespace gqe {
namespace {

TEST(TypesTest, IoEngineTypeRoundTrip)
{
  for (auto engine : {io_engine_type::io_uring, io_engine_type::psync, io_engine_type::automatic}) {
    EXPECT_EQ(from_string<io_engine_type>(to_string(engine)), engine);
  }
}

TEST(TypesTest, CompressionFormatRoundTrip)
{
  for (auto fmt : {compression_format::none,
                   compression_format::ans,
                   compression_format::lz4,
                   compression_format::snappy,
                   compression_format::gdeflate,
                   compression_format::deflate,
                   compression_format::cascaded,
                   compression_format::zstd,
                   compression_format::gzip,
                   compression_format::bitcomp}) {
    EXPECT_EQ(from_string<compression_format>(to_string(fmt)), fmt);
  }
}

TEST(TypesTest, DecompressionBackendRoundTrip)
{
  for (auto backend :
       {decompression_backend::default_, decompression_backend::sm, decompression_backend::de}) {
    EXPECT_EQ(from_string<decompression_backend>(to_string(backend)), backend);
  }
}

}  // namespace
}  // namespace gqe
