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
