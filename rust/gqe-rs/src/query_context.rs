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

//! Query context.

use crate::error::Result;
use crate::executor::OptimizationParameters;
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
