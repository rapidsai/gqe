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

#include <gqe/qep/task.hpp>

#include <gqe/qep/query_execution_plan.hpp>

#include <utility>

namespace gqe {
namespace qep {

// -----------------------------------------------------------------------------
// optional_transform_task
// -----------------------------------------------------------------------------

void optional_transform_task::accept(qep_visitor& visitor) const { visitor.visit(*this); }

// -----------------------------------------------------------------------------
// fold_task
// -----------------------------------------------------------------------------

void fold_task::accept(qep_visitor& visitor) const { visitor.visit(*this); }

state_container fold_task::finalize(state_container&& accumulator,
                                    context_reference /* ctx_ref */,
                                    rmm::cuda_stream_view /* stream */,
                                    rmm::device_async_resource_ref /* mr */) const
{
  return std::move(accumulator);
}

// -----------------------------------------------------------------------------
// iterate_task
// -----------------------------------------------------------------------------

void iterate_task::accept(qep_visitor& visitor) const { visitor.visit(*this); }

void iterate_task::finalize(state_container&& iterator,
                            context_reference /* ctx_ref */,
                            rmm::cuda_stream_view /* stream */,
                            rmm::device_async_resource_ref /* mr */) const
{
  // Trivial finalizer: take ownership so the iterator state is destroyed when this
  // function returns.
  [[maybe_unused]] auto consumed = std::move(iterator);
}

// -----------------------------------------------------------------------------
// stateful_transform_task
// -----------------------------------------------------------------------------

void stateful_transform_task::accept(qep_visitor& visitor) const { visitor.visit(*this); }

void stateful_transform_task::finalize(state_container&& accumulator,
                                       context_reference /* ctx_ref */,
                                       rmm::cuda_stream_view /* stream */,
                                       rmm::device_async_resource_ref /* mr */) const
{
  // Trivial finalizer: take ownership so the accumulator state is destroyed when this
  // function returns.
  [[maybe_unused]] auto consumed = std::move(accumulator);
}

}  // namespace qep
}  // namespace gqe
