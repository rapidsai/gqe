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

#pragma once

namespace gqe::qep {

/**
 * @brief Memory residency of a pipeline's source morsels.
 *
 * Residency describes whether the bytes backing a source work item are stored in CPU (`host`) or
 * GPU (`device`) memory. It does not describe the residency of the task's emitted output.
 */
enum class data_residency {
  host,
  device,
};

/**
 * @brief Scheduling metadata for a pipeline's source morsels.
 */
struct source_morsel_metadata {
  data_residency residency;
};

/**
 * @brief Trait for tasks that expose metadata about their source morsels.
 */
class input_residency_trait {
 public:
  virtual ~input_residency_trait() = default;

  [[nodiscard]] virtual source_morsel_metadata input_residency() const noexcept = 0;
};

}  // namespace gqe::qep
