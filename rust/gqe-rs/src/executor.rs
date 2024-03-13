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
mod query_context;
mod task_graph;

pub use optimization_parameters::*;
pub use query_context::*;
pub use task_graph::*;
