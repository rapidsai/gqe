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

use gqe_rs::api::Catalog;
use gqe_rs::api::{ColumnSchema, DataType, PartitioningSchemaKind, StorageKind};
use gqe_rs::executor::{self, OptimizationParameters, TaskGraphBuilder};
use gqe_rs::logical::SubstraitParser;
use gqe_rs::physical::PhysicalPlanBuilder;
use gqe_rs::query_context::QueryContext;
use gqe_rs::task_manager_context::TaskManagerContext;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let query_file = args[1].clone();
    let parquet_paths = vec![args[2].clone()];

    let tpch_cols = vec![
        ColumnSchema {
            column_name: "l_quantity".into(),
            data_type: DataType::Float64,
        },
        ColumnSchema {
            column_name: "l_extendedprice".into(),
            data_type: DataType::Float64,
        },
        ColumnSchema {
            column_name: "l_discount".into(),
            data_type: DataType::Float64,
        },
        ColumnSchema {
            column_name: "l_shipdate".into(),
            data_type: DataType::TimestampDays,
        },
    ];

    let mut catalog = Catalog::default();
    catalog
        .register_table(
            "lineitem",
            tpch_cols.as_slice(),
            &StorageKind::ParquetFile {
                file_paths: parquet_paths,
            },
            &PartitioningSchemaKind::Automatic,
        )
        .expect("Failed to register table.");

    let parameters =
        OptimizationParameters::new().expect("Failed to construct optimization parameters");
    
    let mut task_manager_ctx = TaskManagerContext::new().expect("Failed to construct a db context.");;
    let mut query_ctx = QueryContext::new(&parameters).expect("Failed to construct a query context.");

    println!("{:?}", parameters);

    // FIXME: Check if GQE really needs mutable references of catalog in all methods. A non-mutable reference would be more ergonomic.
    let logical_plan = {
        let mut parser =
            SubstraitParser::new(&mut catalog).expect("Failed to construct a Substrait parser.");
        let logical_plan = parser
            .from_file(&query_file)
            .expect("Failed to parse the Substrait file.");
        logical_plan
    };

    assert_eq!(logical_plan.len(), 1);

    let mut physical_plan = {
        let mut physical_plan_builder = PhysicalPlanBuilder::new(&mut catalog)
            .expect("Failed to construct a physical plan builder.");
        let physical_plan = physical_plan_builder
            .build(logical_plan.first().unwrap())
            .expect("Failed to build the physical plan.");
        physical_plan
    };

    let task_graph = {
        let mut task_graph_builder = TaskGraphBuilder::new(&mut task_manager_ctx, &mut query_ctx, &mut catalog)
            .expect("Failed to construct a task graph builder.");
        let task_graph = task_graph_builder
            .build(&mut physical_plan)
            .expect("Failed to build the task graph.");
        task_graph
    };

    executor::execute_task_graph_single_gpu(&mut task_manager_ctx, &mut query_ctx, &task_graph)
        .expect("Failed to execute the task graph.");

    println!("Executed");
}
