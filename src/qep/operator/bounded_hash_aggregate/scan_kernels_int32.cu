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

#include "scan_kernels.cuh"

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

#define GQE_BHA_INSTANTIATE_SCAN(Key, Value)                                                      \
  template std::unique_ptr<keyed_accumulator_storage> make_map_storage<Key, Value>(               \
    int64_t, double, cuda::stream_ref, rmm::device_async_resource_ref);                           \
  template void launch_initialize_map<Key, Value>(                                                \
    keyed_accumulator_state&, cudf::aggregation::Kind, cuda::stream_ref);                         \
  template void launch_scan<Key, Value>(typename slot_layout<Key, Value>::storage_element const*, \
                                        int64_t,                                                  \
                                        int64_t,                                                  \
                                        uint8_t const*,                                           \
                                        int64_t const*,                                           \
                                        Key*,                                                     \
                                        scan_output_t<Value>*,                                    \
                                        uint8_t*,                                                 \
                                        cudf::size_type*,                                         \
                                        cuda::stream_ref);                                        \
  template void launch_append_out_of_band<Key, Value>(control_block const*,                       \
                                                      bool,                                       \
                                                      bool,                                       \
                                                      cudf::size_type,                            \
                                                      Key*,                                       \
                                                      scan_output_t<Value>*,                      \
                                                      uint8_t*,                                   \
                                                      cuda::stream_ref);

GQE_BHA_SCAN_INSTANTIATIONS(GQE_BHA_INSTANTIATE_SCAN, int32_t)

#undef GQE_BHA_INSTANTIATE_SCAN

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
