/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
