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

use crate::api::Catalog;
use crate::error::Result;
use crate::physical::PhysicalPlan;
use crate::query_context::QueryContext;
use crate::task_manager_context::TaskManagerContext;
use cxx::UniquePtr;

/// A task graph consisting of GQE tasks.
///
/// GQE tasks are currently not exposed directly to reduce the FFI surface area.
/// Thus, the task graph is an opaque wrapper type.
pub struct TaskGraph(UniquePtr<gqe_sys::TaskGraph>);

/// A builder for generating a task graph from a physical plan.
pub struct TaskGraphBuilder<'a>(UniquePtr<gqe_sys::TaskGraphBuilder<'a>>);

impl<'a> TaskGraphBuilder<'a> {
    /// Returns a new task graph builder.
    pub fn new<'b: 'a, 'c: 'a, 'd: 'a>(
        task_manager_context: &'b mut TaskManagerContext,
        query_context: &'c mut QueryContext,
        catalog: &'d mut Catalog,
    ) -> Result<Self> {
        Ok(Self(gqe_sys::new_task_graph_builder(
            task_manager_context.0.pin_mut(),
            query_context.0.pin_mut(),
            catalog.0.pin_mut(),
        )?))
    }

    /// Generates a new task graph.
    pub fn build(&mut self, physical_plan: &mut PhysicalPlan) -> Result<TaskGraph> {
        Ok(TaskGraph(
            self.0.pin_mut().build_task_graph(physical_plan.0.clone())?,
        ))
    }
}

/// Execute the task graph on a single GPU.
///
/// After this function call, the result tables of the task graph in
/// `task_graph_to_execute` are available to the local GPU.
pub fn execute_task_graph_single_gpu(
    task_manager_context: &mut TaskManagerContext,
    query_context: &mut QueryContext,
    task_graph: &TaskGraph,
) -> Result<()> {
    Ok(gqe_sys::execute_task_graph_single_gpu(
        task_manager_context.0.pin_mut(),
        query_context.0.pin_mut(),
        &task_graph.0,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ColumnSchema, DataType, PartitioningSchemaKind, StorageKind};
    use crate::executor::OptimizationParameters;
    use crate::physical::PhysicalPlanBuilder;
    use crate::storage::new_read_relation;

    #[test]
    fn new_task_graph_builder() {
        let opms = OptimizationParameters::new().unwrap();
        let mut ctx = QueryContext::new(&opms).unwrap();
        let _ = TaskGraphBuilder::new(&mut ctx, &mut Catalog::default()).unwrap();
    }

    // FIXME: This test doesn't work because the task graph builder can't create
    // read tasks on an empty table.
    #[ignore]
    #[test]
    fn build_a_task_graph() {
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

        let mut physical_plan = {
            let mut ppb = PhysicalPlanBuilder::new(&mut cat).unwrap();
            ppb.build(&logical_plan).unwrap()
        };

        let opms = OptimizationParameters::new().unwrap();
        let mut ctx = QueryContext::new(&opms).unwrap();
        let mut tgb = TaskGraphBuilder::new(&mut ctx, &mut cat).unwrap();

        let _ = tgb.build(&mut physical_plan).unwrap();
    }
}
