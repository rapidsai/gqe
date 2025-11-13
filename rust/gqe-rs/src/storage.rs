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
