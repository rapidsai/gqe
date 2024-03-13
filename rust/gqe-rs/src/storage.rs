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

//! Storage components.

use crate::api::DataType;
use crate::error::Result;
use crate::logical::LogicalPlan;

/// Returns a new logical read relation.
pub fn new_read_relation(
    subquery_relations: &[LogicalPlan],
    column_names: &[String],
    column_types: &[DataType],
    table_name: &str,
) -> Result<LogicalPlan> {
    let sqrs = subquery_relations
        .iter()
        .map(|r| r.0.clone())
        .collect::<Vec<_>>();

    Ok(LogicalPlan(gqe_sys::new_read_relation(
        sqrs.as_slice(),
        column_names,
        column_types,
        table_name,
    )?))
}

/// Returns a new logical write relation.
pub fn new_write_relation(
    input_relation: LogicalPlan,
    column_names: &[String],
    column_types: &[DataType],
    table_name: &str,
) -> Result<LogicalPlan> {
    Ok(LogicalPlan(gqe_sys::new_write_relation(
        input_relation.0,
        column_names,
        column_types,
        table_name,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_read_relation_test() {
        let _ = new_read_relation(
            &[],
            &["a_column".to_string()],
            &[DataType::Int32],
            "a_table",
        )
        .unwrap();
    }

    #[test]
    fn new_write_relation_test() {
        let logical_plan = new_read_relation(
            &[],
            &["a_column".to_string()],
            &[DataType::Int32],
            "a_table",
        )
        .unwrap();

        let _ = new_write_relation(
            logical_plan,
            &["a_column".to_string()],
            &[DataType::Int32],
            "a_table",
        )
        .unwrap();
    }
}
