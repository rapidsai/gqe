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

/// Optimization parameters.
use crate::error::Result;

/// Parameters indicating which optimizations are enabled and their settings.
#[derive(Debug)]
pub struct OptimizationParameters(pub(crate) gqe_sys::OptimizationParameters);

impl OptimizationParameters {
    /// Returns a new optimization parameters instance.
    pub fn new() -> Result<Self> {
        Ok(Self(gqe_sys::new_optimization_parameters()?))
    }
}

impl std::ops::Deref for OptimizationParameters {
    type Target = gqe_sys::OptimizationParameters;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for OptimizationParameters {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_optimization_parameters() {
        let _ = OptimizationParameters::new().unwrap();
    }
}
