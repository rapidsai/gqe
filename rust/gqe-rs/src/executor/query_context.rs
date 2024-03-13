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

//! Query context.

use super::OptimizationParameters;
use crate::error::Result;
use cxx::UniquePtr;

/// Query context of the current query.
pub struct QueryContext(pub(crate) UniquePtr<gqe_sys::QueryContext>);

impl QueryContext {
    /// Returns a new query context instance.
    pub fn new(parameters: &OptimizationParameters) -> Result<Self> {
        Ok(Self(gqe_sys::new_query_context(&parameters)?))
    }
}

unsafe impl Send for QueryContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_query_context() {
        let _ = QueryContext::new(&OptimizationParameters::new().unwrap());
    }
}
