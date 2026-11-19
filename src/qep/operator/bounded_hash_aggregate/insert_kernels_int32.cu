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

#include "insert_kernels.cuh"

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

#define GQE_BHA_INSTANTIATE_INSERT(Key, Input, Value, Kind, Nullability) \
  template void launch_insert<Key, Input, Value, Kind, Nullability>(     \
    keyed_accumulator_state&, insert_column_args<Key, Input> const&, cuda::stream_ref);

GQE_BHA_INSERT_VALUE_INSTANTIATIONS(GQE_BHA_INSTANTIATE_INSERT, int32_t, nullability::nullable)
GQE_BHA_INSERT_COUNT_INSTANTIATIONS(GQE_BHA_INSTANTIATE_INSERT, int32_t)

#undef GQE_BHA_INSTANTIATE_INSERT

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
