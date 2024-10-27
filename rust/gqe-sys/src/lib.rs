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

//! Low-level Rust bindings for GQE.
//!
//! The design of the bindings API is as follows.
//!
//! # Function & Method Arguments
//!
//! A limitation of the FFI is that Rust types cannot be constructed in C++, and vice-versa. The reason is that the memory allocators of Rust and C++ are
//! not the same. This also means that foreign objects must be freed by the foreign allocator.
//!
//! Thus, arguments from Rust to C++ must be constructable by Rust. These
//! are Rust-native types that have an opaque C++ representation.
//!
//! # Function & Method Return Types
//!
//! This is the opposite case of argument types above.
//!
//! Return types from C++ to Rust must be constructable by C++. Thus, these are
//! C++-native types that have an opaque Rust representation.
//!
//! # Shared Types
//!
//! A shared type is a type that is native to C++ and Rust. Thus, member fields
//! are visible in both languages.
//!
//! This is useful mostly for C-style enums. Less commonly, it's also useful for
//! structs.
//!
//! An example is the [`OptimizationParameters`] type, because the alternative
//! design would entail writing getter and setter methods for all fields.
//!
//! # Opaque Types
//!
//! An "opaque" type is a native C++ type that is exposed to Rust, but whose
//! member fields are not visible in Rust.
//!
//! Opaque types must be either heap-allocated, or passed by reference. CXX
//! provides [`std::unique_ptr`](cxx::UniquePtr) and
//! [`std::shared_ptr`](cxx::SharedPtr) wrappers in Rust, and a
//! [`std::boxed::Box`] wrapper in C++. These are useful to pass object
//! ownership across the FFI.
//!
//! ## Opaque Wrapper Types
//!
//! C++ types need to be wrapped if their methods are called from Rust. The
//! wrapper provides methods that have Rust-compatible argument and return
//! types. However, the wrapper class itself is native to C++ and opaque to
//! Rust.
//!
//! ## Opaque Directly Exposed Types
//!
//! Some C++ types need to be represented in Rust, but Rust does not need to
//! call their methods. In this case, a wrapper is not needed and the type can
//! be directly exposed to Rust.
//!
//! # Constructors
//!
//! C++ class constructors are wrapped in `new()` functions. This follows the Rust design convention,
//! because Rust does not have constructors.
//!
//! # Rust `enum` to C++ `std::variant` Conversion
//!
//! The C++ `std::variant` is semantically equivalent to a Rust `enum`. However,
//! cxxbridge (and also bindgen) does not support exposing `std::variant` types
//! to Rust (at least not in a useful way). Thus, a Rust wrapper is required.
//!
//! On the Rust side, the bindings must destructure the `enum` into a shared
//! C-style enum type, plus a shared struct type for each non-empty enum variant
//! (i.e., an enum variant with data attached).
//!
//! On the C++ side, the function binding must receive two arguments: the
//! C-style enum, and a `void *` representing the struct. The function must then
//! recover the struct's type via the enum. Finally, the function must construct
//! a new `std::variant` from the enum and the struct.
//!
//! For an example, see [`StorageKindType`].
//!
//! # Error Handling
//!
//! If a function on the Rust side of the bindings returns a `Result<T>`, then
//! cxxbridge generate a `try {} catch(std::exception) {}` block on the C++
//! side.
//!
//! All functions should use this mechanism to handle C++ exceptions and avoid
//! the program exiting, unless the C++ function is marked `noexcept`.
//!
//! # References and Sources
//!
//! This crate uses [CXX](https://cxx.rs) to wrap C++ types as Rust types. The
//! library builds GQE using cmake.
//!
//! See the [cxx-juice](https://github.com/JamesHallowell/cxx-juce/tree/main)
//! crate as an example of how to integrate CXX into a larger project.

pub use ffi::*;

use cxx::{ExternType, SharedPtr};

/// A [`LogicalRelation`] inside a [`cxx::SharedPtr`]
///
/// Workaround for non-implemented `CxxVector<SharedPtr<Type>>`. See
/// https://github.com/dtolnay/cxx/issues/774
#[repr(transparent)]
pub struct SharedLogicalRelation {
    pub ptr: SharedPtr<ffi::LogicalRelation>,
}

// Workaround for non-implemented `CxxVector<SharedPtr<Type>>`. See
// https://github.com/dtolnay/cxx/issues/774
unsafe impl ExternType for SharedLogicalRelation {
    type Id = cxx::type_id!("cxx_gqe::shared_logical_relation");
    type Kind = cxx::kind::Trivial;
}

// The FFI cannot easily be split into multiple files. This is a current
// limitation of `cxxbridge`.
//
// See https://cxx.rs/extern-c++.html#safely-unifying-occurrences-of-an-extern-type-across-bridges
#[cxx::bridge(namespace = "cxx_gqe")]
mod ffi {
    /// Optimization parameters
    #[derive(Debug)]
    #[cxx_name = "optimization_parameters"]
    pub struct OptimizationParameters {
        pub max_num_workers: usize,
        pub max_num_partitions: i32,
        pub log_level: String,
        pub join_use_hash_map_cache: bool,
        pub read_zero_copy_enable: bool,
        pub use_customized_io: bool,
        pub io_bounce_buffer_size: i32,
        pub io_auxiliary_threads: usize,
        pub in_memory_table_compression_format: CompressionFormat,
        pub io_block_size: usize,
        pub io_engine: IOEngineType,
        pub io_pipelining: bool,
        pub io_alignment: usize,
    }

    /// Page kind for the memory allocator
    #[derive(Copy, Clone, Debug)]
    #[cxx_name = "page_kind_type"]
    #[repr(i32)]
    pub enum PageKindType {
        #[cxx_name = "system_default"]
        SystemDefault = 0,
        #[cxx_name = "small"]
        Small,
        #[cxx_name = "transparent_huge"]
        TransparentHuge,
        #[cxx_name = "huge2mb"]
        Huge2Mb,
        #[cxx_name = "huge1gb"]
        Huge1Gb,
    }

    /// Storage kind for the table
    #[derive(Copy, Clone, Debug)]
    #[cxx_name = "storage_kind_type"]
    #[repr(i32)]
    pub enum StorageKindType {
        #[cxx_name = "system_memory"]
        SystemMemory,
        #[cxx_name = "numa_memory"]
        NumaMemory,
        #[cxx_name = "pinned_memory"]
        PinnedMemory,
        #[cxx_name = "device_memory"]
        DeviceMemory,
        #[cxx_name = "managed_memory"]
        ManagedMemory,
        #[cxx_name = "parquet_file"]
        ParquetFile,
    }

    /// NUMA memory specification
    #[cxx_name = "numa_memory_info"]
    pub struct NumaMemoryInfo<'n> {
        numa_node_set: &'n [u64],
        numa_node_set_bytes: i32,
        page_kind: PageKindType,
    }

    /// Device memory specification
    #[cxx_name = "device_memory_info"]
    pub struct DeviceMemoryInfo {
        device_id: i32,
    }

    /// Parquet file specification
    #[cxx_name = "parquet_file_info"]
    pub struct ParquetFileInfo {
        file_paths: Vec<String>,
    }

    /// Partitioning schema kind
    #[derive(Copy, Clone, Debug)]
    #[cxx_name = "partitioning_schema_kind_type"]
    #[repr(i32)]
    pub enum PartitioningSchemaKindType {
        #[cxx_name = "automatic"]
        Automatic,
        #[cxx_name = "none"]
        None,
        #[cxx_name = "key"]
        Key,
    }

    /// By-key partitioning schema specification
    #[cxx_name = "key_schema_info"]
    pub struct KeySchemaInfo {
        columns: Vec<String>,
    }

    /// Compression format kind
    #[derive(Copy, Clone, Debug)]
    #[cxx_name = "compression_format"]
    #[repr(i32)]
    pub enum CompressionFormat {
        #[cxx_name = "none"]
        None = 0,
        #[cxx_name = "ans"]
        ANS,
    }

    /// IO engine type
    #[derive(Copy, Clone, Debug)]
    #[cxx_name = "io_engine_type"]
    #[repr(i32)]
    pub enum IOEngineType {
        #[cxx_name = "automatic"]
        Automatic = 0,
        #[cxx_name = "io_uring"]
        IOUring,
        #[cxx_name = "psync"]
        PSync,
    }

    /// cuDF data type
    ///
    /// This definition must be identical to the C++
    /// [`cudf::type_id`](https://docs.rapids.ai/api/libcudf/stable/group__utility__types#gadf077607da617d1dadcc5417e2783539).
    #[derive(Copy, Clone, Debug)]
    #[cxx_name = "type_id"]
    #[repr(i32)]
    pub enum DataType {
        /// Always null with no underlying data
        #[cxx_name = "EMPTY"]
        Empty,
        /// 1 byte signed integer
        #[cxx_name = "INT8"]
        Int8,
        /// 2 byte signed integer
        #[cxx_name = "INT16"]
        Int16,
        /// 4 byte signed integer
        #[cxx_name = "INT32"]
        Int32,
        /// 8 byte signed integer
        #[cxx_name = "INT64"]
        Int64,
        /// 1 byte unsigned integer
        #[cxx_name = "UINT8"]
        Uint8,
        /// 2 byte unsigned integer
        #[cxx_name = "UINT16"]
        Uint16,
        /// 4 byte unsigned integer
        #[cxx_name = "UINT32"]
        Uint32,
        /// 8 byte unsigned integer
        #[cxx_name = "UINT64"]
        Uint64,
        /// 4 byte floating point
        #[cxx_name = "FLOAT32"]
        Float32,
        /// 8 byte floating point
        #[cxx_name = "FLOAT64"]
        Float64,
        /// Boolean using one byte per value, 0 == false, else true
        #[cxx_name = "BOOL8"]
        Bool8,
        /// point in time in days since Unix Epoch in int32
        #[cxx_name = "TIMESTAMP_DAYS"]
        TimestampDays,
        /// point in time in seconds since Unix Epoch in int64
        #[cxx_name = "TIMESTAMP_SECONDS"]
        TimestampSeconds,
        /// point in time in milliseconds since Unix Epoch in int64
        #[cxx_name = "TIMESTAMP_MILLISECONDS"]
        TimestampMilliseconds,
        /// point in time in microseconds since Unix Epoch in int64
        #[cxx_name = "TIMESTAMP_MICROSECONDS"]
        TimestampMicroseconds,
        /// point in time in nanoseconds since Unix Epoch in int64
        #[cxx_name = "TIMESTAMP_NANOSECONDS"]
        TimestampNanoseconds,
        /// time interval of days in int32
        #[cxx_name = "DURATION_DAYS"]
        DurationDays,
        /// time interval of seconds in int64
        #[cxx_name = "DURATION_SECONDS"]
        DurationSeconds,
        /// time interval of milliseconds in int64
        #[cxx_name = "DURATION_MILLISECONDS"]
        DurationMilliseconds,
        /// time interval of microseconds in int64
        #[cxx_name = "DURATION_MICROSECONDS"]
        DurationMicroseconds,
        /// time interval of nanoseconds in int64
        #[cxx_name = "DURATION_NANOSECONDS"]
        DurationNanoseconds,
        /// Dictionary type using int32 indices
        #[cxx_name = "DICTIONARY32"]
        Dictionary32,
        /// String elements
        #[cxx_name = "STRING"]
        String,
        /// List elements
        #[cxx_name = "LIST"]
        List,
        /// Fixed-point type with int32_t
        #[cxx_name = "DECIMAL32"]
        Decimal32,
        /// Fixed-point type with int64_t
        #[cxx_name = "DECIMAL64"]
        Decimal64,
        /// Fixed-point type with __int128_t
        #[cxx_name = "DECIMAL128"]
        Decimal128,
        /// Struct elements
        #[cxx_name = "STRUCT"]
        Struct,
        /// Total number of type ids
        // `NUM_TYPE_IDS` must be last!
        #[cxx_name = "NUM_TYPE_IDS"]
        NumTypeIds,
    }

    /// Column schema helper type
    ///
    /// This type summarizes the column name and column data type, because there
    /// is no shared `std::tuple` type.
    #[derive(Clone, Debug)]
    #[cxx_name = "column_schema"]
    pub struct ColumnSchema {
        pub column_name: String,
        pub data_type: DataType,
    }

    unsafe extern "C++" {
        include!("gqe-sys/include/cxx_gqe.hpp");

        type c_void;

        #[cxx_name = "page_kind_type"]
        pub type PageKindType;

        #[cxx_name = "compression_format"]
        pub type CompressionFormat;

        #[cxx_name = "io_engine_type"]
        pub type IOEngineType;

        #[cxx_name = "catalog"]
        pub type Catalog;

        #[cxx_name = "type_id"]
        pub type DataType;

        pub fn new_catalog() -> UniquePtr<Catalog>;
        pub unsafe fn register_table(
            self: Pin<&mut Catalog>,
            table_name: &str,
            columns: &[ColumnSchema],
            storage_type: StorageKindType,
            storage_info: *const c_void,
            partitioning_schema_type: PartitioningSchemaKindType,
            partitioning_schema_info: *const c_void,
        ) -> Result<()>;
        pub fn column_names(
            self: &Catalog,
            table_name: &str,
        ) -> Result<UniquePtr<CxxVector<CxxString>>>;
        pub fn column_type(self: &Catalog, table_name: &str, column_name: &str)
            -> Result<DataType>;

        pub fn new_optimization_parameters() -> Result<OptimizationParameters>;

        #[cxx_name = "logical_relation"]
        pub type LogicalRelation;

        // Workaround for non-implemented `CxxVector<SharedPtr<Type>>`. See
        // https://github.com/dtolnay/cxx/issues/774
        #[cxx_name = "shared_logical_relation"]
        type SharedLogicalRelation = crate::SharedLogicalRelation;

        #[cxx_name = "query_context"]
        pub type QueryContext;

        pub fn new_query_context(
            parameters: &OptimizationParameters,
        ) -> Result<UniquePtr<QueryContext>>;

        #[cxx_name = "substrait_parser"]
        pub type SubstraitParser<'a>;

        pub fn new_substrait_parser<'a>(
            catalog: Pin<&'a mut Catalog>,
        ) -> Result<UniquePtr<SubstraitParser<'a>>>;
        pub fn from_file(
            self: Pin<&mut SubstraitParser>,
            substrait_file: &str,
        ) -> Result<UniquePtr<CxxVector<SharedLogicalRelation>>>;

        #[cxx_name = "physical_relation"]
        pub type PhysicalRelation;

        #[cxx_name = "physical_plan_builder"]
        pub type PhysicalPlanBuilder<'a>;

        pub fn new_physical_plan_builder<'a>(
            catalog: Pin<&'a mut Catalog>,
        ) -> Result<UniquePtr<PhysicalPlanBuilder<'a>>>;
        pub fn build(
            self: Pin<&mut PhysicalPlanBuilder>,
            logical_relation: &LogicalRelation,
        ) -> Result<SharedPtr<PhysicalRelation>>;

        #[cxx_name = "task_graph"]
        pub type TaskGraph;

        #[cxx_name = "task_graph_builder"]
        pub type TaskGraphBuilder<'a>;

        pub fn new_task_graph_builder<'a, 'b>(
            query_context: Pin<&'a mut QueryContext>,
            catalog: Pin<&'b mut Catalog>,
        ) -> Result<UniquePtr<TaskGraphBuilder<'a>>>;

        #[cxx_name = "build"]
        pub fn build_task_graph(
            self: Pin<&mut TaskGraphBuilder>,
            root_relation: SharedPtr<PhysicalRelation>,
        ) -> Result<UniquePtr<TaskGraph>>;

        pub fn execute_task_graph_single_gpu(
            query_context: Pin<&mut QueryContext>,
            task_graph: &TaskGraph,
        ) -> Result<()>;

        pub fn new_read_relation(
            subquery_relations: &[SharedPtr<LogicalRelation>],
            column_names: &[String],
            column_types: &[DataType],
            table_name: &str,
        ) -> Result<SharedPtr<LogicalRelation>>;

        pub fn new_write_relation(
            input_relation: SharedPtr<LogicalRelation>,
            column_names: &[String],
            column_types: &[DataType],
            table_name: &str,
        ) -> Result<SharedPtr<LogicalRelation>>;
    }

    impl CxxVector<SharedLogicalRelation> {}
    impl SharedPtr<LogicalRelation> {}
    impl UniquePtr<OptimizationParameters> {}
}
