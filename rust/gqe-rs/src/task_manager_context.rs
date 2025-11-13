/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Task manager context.

use crate::error::Result;
use cxx::UniquePtr;

/// Task manager context for execution on the node.
pub struct TaskManagerContext(pub(crate) UniquePtr<gqe_sys::TaskManagerContext>);

impl TaskManagerContext {
    /// Returns a new db context instance.
    pub fn new() -> Result<Self> {
        Ok(Self(gqe_sys::new_task_manager_context()?))
    }
}

unsafe impl Send for TaskManagerContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_task_manager_context() {
        let _ = TaskManagerContext::new();
    }
}
