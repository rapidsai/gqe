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
