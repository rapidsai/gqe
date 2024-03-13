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

//! Logical query plan components.

use crate::api::Catalog;
use crate::error::Result;
use cxx::{SharedPtr, UniquePtr};

/// A logical query plan.
#[derive(Clone)]
pub struct LogicalPlan(pub(crate) SharedPtr<gqe_sys::LogicalRelation>);

/// A parser for Substrait query plans.
///
/// The Substrait parser converts a Substrait query plan into GQE logical query
/// plan.
pub struct SubstraitParser<'a>(UniquePtr<gqe_sys::SubstraitParser<'a>>);

impl<'a> SubstraitParser<'a> {
    /// Returns a new Substrait parser instance.
    pub fn new<'b: 'a>(catalog: &'b mut Catalog) -> Result<Self> {
        Ok(Self(gqe_sys::new_substrait_parser(catalog.0.pin_mut())?))
    }

    /// Parse a Substrait binary file into a GQE logical plan.
    pub fn from_file(&mut self, substrait_file: &str) -> Result<Vec<LogicalPlan>> {
        let result = self
            .0
            .pin_mut()
            .from_file(substrait_file)?
            .into_iter()
            .map(|slp| LogicalPlan(slp.ptr.clone()))
            .collect();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_substrait_parser() {
        let _ = SubstraitParser::new(&mut Catalog::default()).unwrap();
    }
}
