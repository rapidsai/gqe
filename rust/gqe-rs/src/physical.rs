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

//! Phyiscal query plan components.

use crate::api::Catalog;
use crate::error::Result;
use crate::logical::LogicalPlan;
use cxx::{SharedPtr, UniquePtr};

/// A physical query plan.
#[derive(Clone)]
pub struct PhysicalPlan(pub(crate) SharedPtr<gqe_sys::PhysicalRelation>);

/// A builder to convert a logical query plan into a physical query plan.
pub struct PhysicalPlanBuilder<'a>(UniquePtr<gqe_sys::PhysicalPlanBuilder<'a>>);

impl<'a> PhysicalPlanBuilder<'a> {
    /// Returns a new builder instance.
    pub fn new<'b: 'a>(catalog: &'b mut Catalog) -> Result<Self> {
        Ok(Self(gqe_sys::new_physical_plan_builder(
            catalog.0.pin_mut(),
        )?))
    }

    /// Builds a physical query plan.
    pub fn build(&mut self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        Ok(PhysicalPlan(self.0.pin_mut().build(&logical_plan.0)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ColumnSchema;
    use crate::api::{DataType, PartitioningSchemaKind, StorageKind};
    use crate::storage::new_read_relation;

    #[test]
    fn new_physical_plan_builder() {
        let _ = PhysicalPlanBuilder::new(&mut Catalog::default()).unwrap();
    }

    #[test]
    fn build_a_physical_plan() {
        let table_name = "a_table";
        let column_name = "a_column";
        let data_type = DataType::Int32;

        let mut cat = Catalog::default();
        cat.register_table(
            table_name,
            &[ColumnSchema {
                column_name: column_name.to_string(),
                data_type,
            }],
            &StorageKind::SystemMemory,
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();

        let logical_plan =
            new_read_relation(&[], &[column_name.to_string()], &[data_type], table_name).unwrap();

        let mut builder = PhysicalPlanBuilder::new(&mut cat).unwrap();
        builder.build(&logical_plan).unwrap();
    }
}
