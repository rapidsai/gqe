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

//! Query executor components.
//!
//! The executor contains the components that execute a task graph on a GPU.
//!
//! # Example
//!
//! Here is a usage example for the query context.
//!
//! ```
//! # use gqe_rs::error::Result;
//! # use gqe_rs::executor::*;
//! # use gqe_rs::query_context;
//! #
//! # fn test() -> Result<()> {
//! #
//! // Initialize with default parameters
//! let mut opms = OptimizationParameters::new()?;
//! // Set a numeric parameter
//! opms.max_num_workers = 5;
//! // Set a string parameter
//! opms.log_level = "warn".to_string();
//! // Finalize parameters to read-only
//! let opms = opms;
//!
//! // Create a new query context
//! let query_context = QueryContext::new(&opms)?;
//! #
//! # Ok(())
//! # }
//! ```

mod optimization_parameters;
mod task_graph;

pub use optimization_parameters::*;
pub use task_graph::*;
