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

//! Types and their helpers.

use crate::utility::CpuSet;

// Directly export types in public API.

/// Specifies the kind of operating system page.
pub type PageKind = gqe_sys::PageKindType;

/// A [cuDF data type](https://docs.rapids.ai/api/libcudf/stable/group__utility__types.html#gadf077607da617d1dadcc5417e2783539).
pub type DataType = gqe_sys::DataType;

/// The schema of a column.
pub type ColumnSchema = gqe_sys::ColumnSchema;

/// Storage kind of a table.
///
/// The storage kind declares the physical representation of a table. For
/// example, the storage kind can be in-memory or a file. Some storage kinds
/// also take a location hint that specifies where the table should be stored,
/// e.g., on which NUMA node.
#[derive(Debug)]
pub enum StorageKind {
    SystemMemory,
    NumaMemory {
        numa_node_set: CpuSet,
        page_kind: PageKind,
    },
    PinnedMemory,
    DeviceMemory {
        device_id: i32,
    },
    ManagedMemory,
    ParquetFile {
        file_paths: Vec<String>,
    },
}

impl StorageKind {
    pub(crate) fn to_ffi(&self) -> (gqe_sys::StorageKindType, Option<Box<dyn StorageInfo + '_>>) {
        match self {
            StorageKind::SystemMemory => (gqe_sys::StorageKindType::SystemMemory, None),
            StorageKind::NumaMemory {
                numa_node_set,
                page_kind,
            } => (
                gqe_sys::StorageKindType::NumaMemory,
                Some(Box::new(gqe_sys::NumaMemoryInfo {
                    numa_node_set: numa_node_set.as_slice(),
                    numa_node_set_bytes: numa_node_set.bytes() as i32,
                    page_kind: *page_kind,
                })),
            ),
            StorageKind::PinnedMemory => (gqe_sys::StorageKindType::PinnedMemory, None),
            StorageKind::DeviceMemory { device_id } => (
                gqe_sys::StorageKindType::DeviceMemory,
                Some(Box::new(gqe_sys::DeviceMemoryInfo {
                    device_id: *device_id,
                })),
            ),
            StorageKind::ManagedMemory => (gqe_sys::StorageKindType::ManagedMemory, None),
            StorageKind::ParquetFile { file_paths } => (
                gqe_sys::StorageKindType::ParquetFile,
                Some(Box::new(gqe_sys::ParquetFileInfo {
                    file_paths: file_paths.clone(),
                })),
            ),
        }
    }
}

/// Partitioning schema kind of a table.
///
/// The partitioning schema kind declares how records are assigned to table
/// partitions.
pub enum PartitioningSchemaKind {
    Automatic,
    None,
    Key { columns: Vec<String> },
}

impl PartitioningSchemaKind {
    pub(crate) fn to_ffi(
        &self,
    ) -> (
        gqe_sys::PartitioningSchemaKindType,
        Option<Box<dyn PartitioningSchemaInfo>>,
    ) {
        match self {
            PartitioningSchemaKind::Automatic => {
                (gqe_sys::PartitioningSchemaKindType::Automatic, None)
            }
            PartitioningSchemaKind::None => (gqe_sys::PartitioningSchemaKindType::None, None),
            PartitioningSchemaKind::Key { columns } => (
                gqe_sys::PartitioningSchemaKindType::Key,
                Some(Box::new(gqe_sys::KeySchemaInfo {
                    columns: columns.clone(),
                })),
            ),
        }
    }
}

/// A helper trait for converting `gqe_sys::*Info` structs into a uniform
/// reference type.
pub(crate) trait StorageInfo {}

impl StorageInfo for gqe_sys::NumaMemoryInfo<'_> {}
impl StorageInfo for gqe_sys::DeviceMemoryInfo {}
impl StorageInfo for gqe_sys::ParquetFileInfo {}

/// A helper trait for converting `*Info` structs into a uniform reference type.
pub(crate) trait PartitioningSchemaInfo {}

impl PartitioningSchemaInfo for gqe_sys::KeySchemaInfo {}
