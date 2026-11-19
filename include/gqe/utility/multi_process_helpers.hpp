/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

namespace gqe::utility {
namespace multi_process {

/**
 * @brief Check if the current process is rank 0.
 *
 * @return true if this PE is rank 0, false otherwise.
 */
bool nvshmem_rank_zero();

/**
 * @brief Get the NVSHMEM rank of the current process.
 *
 * @return The PE number of the calling process.
 */
int nvshmem_rank();

}  // namespace multi_process
}  // namespace gqe::utility
