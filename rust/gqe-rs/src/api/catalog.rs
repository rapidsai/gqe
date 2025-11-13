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

//! The GQE schema catalog.

use super::types::{
    DataType, PartitioningSchemaInfo, PartitioningSchemaKind, StorageInfo, StorageKind,
};
use crate::api::ColumnSchema;
use crate::error::Result;
use cxx::UniquePtr;

/// A catalog of table schemas.
pub struct Catalog(pub(crate) UniquePtr<gqe_sys::Catalog>);

impl Catalog {
    /// Register a new table into the catalog.
    pub fn register_table(
        &mut self,
        table_name: &str,
        columns: &[ColumnSchema],
        storage: &StorageKind,
        partitioning_schema: &PartitioningSchemaKind,
    ) -> Result<()> {
        let (storage_type, storage_info) = storage.to_ffi();
        let storage_info_ptr = if let Some(ref info) = storage_info {
            info.as_ref() as *const dyn StorageInfo as *const gqe_sys::c_void
        } else {
            std::ptr::null()
        };
        let (ps_type, ps_info) = partitioning_schema.to_ffi();
        let ps_info_ptr = if let Some(ref info) = ps_info {
            info.as_ref() as *const dyn PartitioningSchemaInfo as *const gqe_sys::c_void
        } else {
            std::ptr::null()
        };

        unsafe {
            self.0.pin_mut().register_table(
                table_name,
                columns,
                storage_type,
                storage_info_ptr,
                ps_type,
                ps_info_ptr,
            )
        }?;

        Ok(())
    }

    /// Return the names of all columns in the table in the user-defined order.
    pub fn column_names(&self, table_name: &str) -> Result<Vec<String>> {
        let names = self.0.column_names(table_name)?;

        let result = names
            .iter()
            .map(|cxx| Ok(cxx.to_str()?.to_owned()))
            .collect::<Result<_>>()?;

        Ok(result)
    }

    /// Return the data type of a column in the catalog.
    pub fn column_type(&self, table_name: &str, column_name: &str) -> Result<DataType> {
        Ok(self.0.column_type(table_name, column_name)?)
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self(gqe_sys::new_catalog())
    }
}

unsafe impl Send for Catalog {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::PageKind;
    use crate::utility::CpuSet;

    #[test]
    fn create_default_catalog() {
        let _ = Catalog::default();
    }

    fn register_table_helper(
        storage: &StorageKind,
        partitioning_schema: &PartitioningSchemaKind,
    ) -> Result<()> {
        let mut cat = Catalog::default();
        let cols = [ColumnSchema {
            column_name: "a_column".to_string(),
            data_type: DataType::Int32,
        }];

        cat.register_table("a_name", &cols, storage, partitioning_schema)
    }

    #[test]
    fn register_table_system_memory() {
        register_table_helper(
            &StorageKind::SystemMemory,
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();
    }

    #[test]
    fn register_table_numa_memory() {
        let mut numa_node_set = CpuSet::new();
        numa_node_set.add(0);

        register_table_helper(
            &StorageKind::NumaMemory {
                numa_node_set,
                page_kind: PageKind::SystemDefault,
            },
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();
    }

    #[test]
    fn register_table_pinned_memory() {
        register_table_helper(
            &StorageKind::PinnedMemory,
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();
    }

    #[test]
    fn register_table_device_memory() {
        register_table_helper(
            &StorageKind::DeviceMemory { device_id: 0 },
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();
    }

    #[test]
    fn register_table_managed_memory() {
        register_table_helper(
            &StorageKind::ManagedMemory,
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();
    }

    #[test]
    fn register_table_parquet_file() {
        register_table_helper(
            &StorageKind::ParquetFile {
                file_paths: vec!["a/test/path".to_string()],
            },
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();
    }

    #[test]
    fn register_table_key_schema() {
        register_table_helper(
            &StorageKind::SystemMemory,
            &PartitioningSchemaKind::Key {
                columns: vec!["a_column".to_string()],
            },
        )
        .unwrap();
    }

    #[test]
    fn register_table_none_schema() {
        register_table_helper(&StorageKind::SystemMemory, &PartitioningSchemaKind::None).unwrap();
    }

    #[test]
    fn column_names() {
        let table_name = "a_name";
        let col_names = ["a_column", "another_column"];

        let cols = col_names.map(|n| ColumnSchema {
            column_name: n.to_string(),
            data_type: DataType::Int32,
        });

        let mut cat = Catalog::default();
        cat.register_table(
            table_name,
            &cols,
            &StorageKind::SystemMemory,
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();
        let res_names = cat.column_names(table_name).unwrap();

        assert_eq!(col_names.as_slice(), res_names.as_slice());
    }

    #[test]
    fn column_types() {
        let table_name = "a_name";
        let col_schema = [("a", DataType::Int64), ("b", DataType::Float32)];

        let cols = col_schema.map(|(col_name, data_type)| ColumnSchema {
            column_name: col_name.to_string(),
            data_type,
        });

        let mut cat = Catalog::default();
        cat.register_table(
            table_name,
            &cols,
            &StorageKind::SystemMemory,
            &PartitioningSchemaKind::Automatic,
        )
        .unwrap();

        let res_types = col_schema.map(|(n, _)| cat.column_type(table_name, n).unwrap());

        let col_types = col_schema.map(|(_, t)| t);
        assert_eq!(col_types.as_slice(), res_types.as_slice());
    }
}
