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
