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

#pragma once

#include <gqe/utility/helpers.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace gqe {

/**
 * @brief Enum to specify which engine to use for I/O operations.
 */
enum class io_engine_type { automatic, io_uring, psync };

/**
 * @brief Enum to specify the type of a join.
 * check here for more details: https://substrait.io/relations/logical_relations/#join-types
 */
enum class join_type_type { inner, left, left_semi, left_anti, full, single };

/**
 * @brief Enum to specify policy for enabling the unique keys optimization for inner hash join.
 */
enum class unique_keys_policy : int {
  none,   ///< Disable the unique keys optimization.
  right,  ///< Build on right side, and assume keys are unique.
  left,   ///< Build on left side, and assume keys are unique.
  either  ///< Build on either left or right side, and assume keys are unique.
};

/**
 * @brief Representation of a set of CPUs.
 *
 * C++ wrapper class for the Linux `CPU_SET` type. It can also be used to represent other resource
 * types, such as NUMA node set.
 *
 * The CPU set is compatible with operating system interfaces such as `sched_setaffinity` and
 * `mbind`.
 */
class cpu_set {
 public:
  static constexpr int max_count = 1024; /**< Maximum CPU ID. Can be increased if required. */
  static constexpr unsigned int qword_count = utility::divide_round_up(
    max_count, sizeof(unsigned long) * CHAR_BIT); /**< Number of unsigned longs in the cpu_set. */

  /**
   * @brief Create an empty CPU set.
   */
  explicit cpu_set();

  /**
   * @brief Destroy the CPU set.
   */
  ~cpu_set();

  /**
   * @brief Create a CPU set with CPU ID as a member.
   */
  explicit cpu_set(int cpu_id);

  /**
   * @brief Copy-construct from a `cpu_set_t`.
   */
  explicit cpu_set(const cpu_set_t& other, const int32_t num_cpus);

  /**
   * @brief Copy-construct from the other CPU set.
   */
  cpu_set(const cpu_set& other);

  /**
   * @brief Copy the other CPU set.
   */
  cpu_set& operator=(const cpu_set& other);

  /**
   * @brief Add a CPU ID to the set.
   */
  cpu_set& add(int cpu_id) noexcept;

  /**
   * @brief Return whether CPU ID is a member of the set.
   */
  [[nodiscard]] bool contains(int cpu_id) const noexcept;

  /**
   * @brief Return the current number of CPUs in the set.
   */
  [[nodiscard]] int count() const noexcept;

  /**
   * @brief Return whether the set is empty.
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Return the first (lowest) CPU ID in the set.
   *
   * @throws std::logic_error if the set is empty.
   */
  [[nodiscard]] int front() const;

  /**
   * @brief Return the last (highest) CPU ID in the set.
   *
   * @throws std::logic_error if the set is empty.
   */
  [[nodiscard]] int back() const;

  /**
   * @brief Return the raw bits of the set.
   */
  [[nodiscard]] const unsigned long* bits() const noexcept;

  /**
   * @brief Return the raw bits of the set.
   */
  [[nodiscard]] unsigned long* bits() noexcept;

  /**
   * @brief Visualize the bits of the CPU set.
   *
   * This method is helpful for debugging.
   */
  [[nodiscard]] std::string pretty_print() const;

  /**
   * @brief Compare two CPU sets for equality.
   */
  bool operator==(const cpu_set& other) const noexcept;

 private:
  cpu_set_t* _cpu_set;
};

/**
 * The operating system page kind.
 *
 * Linux supports multiple page kinds. These include regular "small" pages, huge pages,
 * and transparent huge pages. Huge pages come in different sizes. For example, the x86_64
 * architecture supports 2 MB and 1 GB huge pages.
 *
 * == Transparent Huge Pages Configuration ==
 *
 * The recommended configuration for transparent huge pages are:
 * ```sh
 * sudo bash -c 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled'
 * sudo bash -c 'echo madvise > /sys/kernel/mm/transparent_hugepage/defrag'
 * ```
 * With these settings, GQE can control whether or not THP are enabled, and
 * memory allocations will fail if THP are requested but not available.
 *
 * In more detail, the default page kind is configured with
 * `/sys/kernel/mm/transparent_hugepage/enabled`. Valid settings are `never`, `madvise`, and
 * `always`. The setting `madvise` sets the default to small pages, the setting `always` to
 * transparent huge pages. These settings allow GQE to set a custom page size. In contrast, `never`
 * prevents GQE from configuring transparent huge pages.
 *
 * The transparent huge page reclamation behavior is configured with
 * `/sys/kernel/mm/transparent_hugepage/defrag`. Valid settings are `always`, `defer+madvise`,
 * `madvise`, `defer`, and `never`. The first three settings stall on the `madvise` syscall (i.e.,
 * allocation) until enough transparent huge pages are available. In contrast, `defer` and `never`
 * allow small pages to be allocated despite requesing transparent huge pages.
 *
 * == Huge Pages Configuration ==
 *
 * The system administrator must configure huge pages in the operating system before they can be
 * allocated by GQE. Linux supports reserving huge pages ahead-of-time or allocating huge pages
 * on-the-fly (i.e,. overcommit).
 *
 * The number of system-wide reserved and overcommitted 2 MB huge pages are set with:
 * ```
 * echo 20 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
 * echo 20 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages
 * ```
 *
 * The equivalent reservation setting for each NUMA node is:
 * ```
 * echo 20 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
 * ```
 *
 * Substitute the NUMA node and page size as necessary.
 *
 * == References ==
 *
 * More information on huge pages and transparent huge pages is available in the Linux kernel
 * documentation:
 * - https://www.kernel.org/doc/Documentation/vm/transhuge.txt
 * - https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt
 */
class page_kind {
 public:
  enum type {
    system_default, /**< The system default page size. Linux systems typically configure either
                      small pages or transparent huge pages. */
    small, /**< Forces the use of small pages. On x86_64 the page size is 4 KB. PPC64le supports 4
              KB and 64 KB. AARCH64 supports 4 KB, 16 KB, and 64 KB. A single page size is
              configured at boot time. */
    transparent_huge, /**< Forces the use of transparent huge pages, even if they are disabled in
                         the OS default page kind. */
    huge2mb,          /**< Allocate 2 MB huge pages with `mmap`. */
    huge1gb           /**< Allocate 1 GB huge pages with `mmap`. */
  };

  /**
   * Create a page kind of type system default.
   */
  page_kind();

  /**
   * Create a page kind of the specified type.
   */
  explicit page_kind(type type);

  /**
   * @brief Return the page kind type.
   */
  [[nodiscard]] type value() const noexcept;

  /**
   * @brief Return the page size of the type.
   */
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  type _type{system_default};
};

namespace memory_kind {

/**
 * @brief System memory allocated with the default system allocator.
 */
struct system {
  bool operator==(system const& other) const;
};

/**
 * @brief System memory allocated on a specific NUMA node.
 *
 * The memory resource for numa is managed centrally by task_manager_context.
 * Use task_manager_context::get_memory_resource(memory_kind::numa{...}) to access it.
 */
struct numa {
  cpu_set numa_node_set;                                           /**< NUMA node set hint. */
  gqe::page_kind::type page_kind = gqe::page_kind::system_default; /**< Page kind hint */

  /**
   * @brief Construct numa with explicit numa_node_set.
   */
  numa(cpu_set node_set, gqe::page_kind::type pk = gqe::page_kind::system_default);

  /**
   * @brief Construct numa using the memory affinity of the current CUDA device.
   */
  explicit numa(gqe::page_kind::type pk = gqe::page_kind::system_default);

  bool operator==(numa const& other) const;
};

/**
 * @brief Pinned system memory allocated with `cudaMallocHost`.
 */
struct pinned {
  bool operator==(pinned const& other) const;
};

/**
 * @brief Pinned host memory allocated on a specific NUMA node with a specific page kind.
 *
 * The memory resource for numa_pinned is managed centrally by task_manager_context.
 * Use task_manager_context::get_memory_resource(memory_kind::numa_pinned{...}) to access it.
 */
struct numa_pinned {
  cpu_set numa_node_set;                                           /**< NUMA node set hint. */
  gqe::page_kind::type page_kind = gqe::page_kind::system_default; /**< Page kind hint */

  /**
   * @brief Construct numa_pinned with explicit numa_node_set.
   */
  numa_pinned(cpu_set node_set, gqe::page_kind::type pk = gqe::page_kind::system_default);

  /**
   * @brief Construct numa_pinned using the memory affinity of the current CUDA device.
   */
  explicit numa_pinned(gqe::page_kind::type pk = gqe::page_kind::system_default);

  bool operator==(numa_pinned const& other) const;
};
/**
 * @brief CUDA device memory.
 */
struct device {
  rmm::cuda_device_id device_id; /**< CUDA device identifier hint. */

  bool operator==(device const& other) const;
};

/**
 * @brief CUDA managed memory.
 */
struct managed {
  bool operator==(managed const& other) const;
};

/**
 * @brief Boost shared memory.
 *
 * The memory resource for boost_shared is managed centrally by task_manager_context.
 * Use task_manager_context::get_memory_resource(memory_kind::boost_shared{}) to access it.
 */
struct boost_shared {
  bool operator==(boost_shared const&) const;
};

/**
 * @brief Memory kind of an in-memory table.
 */
using type = std::variant<memory_kind::system,
                          memory_kind::numa,
                          memory_kind::pinned,
                          memory_kind::numa_pinned,
                          memory_kind::device,
                          memory_kind::managed,
                          memory_kind::boost_shared>;

/**
 * @brief Return whether the GPU can directly access the memory kind
 */
bool is_gpu_accessible(memory_kind::type type);

/**
 * @brief Return whether the CPU can directly access the memory kind
 */
bool is_cpu_accessible(memory_kind::type type);

/**
 * @brief Hash function for memory_kind::type.
 */
struct type_hash {
  std::size_t operator()(memory_kind::type const& type) const;
};

}  // namespace memory_kind

namespace storage_kind {

/**
 * @copydoc gqe::memory_kind::system_memory
 */
using system_memory = memory_kind::system;

/**
 * @copydoc gqe::memory_kind::numa_memory
 */
using numa_memory = memory_kind::numa;

/**
 * @copydoc gqe::memory_kind::pinned_memory
 */
using pinned_memory = memory_kind::pinned;

/**
 * @copydoc gqe::memory_kind::device_memory
 */
using device_memory = memory_kind::device;

/**
 * @copydoc gqe::memory_kind::managed_memory
 */
using managed_memory = memory_kind::managed;

/**
 * @copydoc gqe::memory_kind::numa_pinned_memory
 */
using numa_pinned_memory = memory_kind::numa_pinned;

/**
 * @copydoc gqe::memory_kind::boost_shared_memory
 */
using boost_shared_memory = memory_kind::boost_shared;

/**
 * @brief Parquet file format, optionally Hive partitioned.
 */
struct parquet_file {
  std::vector<std::string> file_paths; /**< File paths. */
};

/**
 * @brief Storage kind of a table.
 *
 * The storage kind declares the physical representation of a table. For
 * example, the storage kind can be in-memory or a file. Some storage kinds also
 * take a location hint that specifies where the table should be stored, e.g.,
 * on which NUMA node.
 */
using type = std::variant<storage_kind::system_memory,
                          storage_kind::numa_memory,
                          storage_kind::pinned_memory,
                          storage_kind::device_memory,
                          storage_kind::managed_memory,
                          storage_kind::numa_pinned_memory,
                          storage_kind::boost_shared_memory,
                          storage_kind::parquet_file>;

}  // namespace storage_kind

namespace partitioning_schema_kind {

/**
 * @brief Automatic partitioning that infers the schema from the data.
 *
 * Inferrence of the partitioning schema is only possible for files. In-memory
 * storage defaults to `none`.
 */
struct automatic {};

/**
 * @brief Do not partition data.
 */
struct none {};

/**
 * @brief Key-by partitioning schema.
 *
 * The data are partitioned by key. Each unique key defines a partition.
 */
struct key {
  std::vector<std::string> columns; /**< Declare the key attributes. */
};

/**
 * @brief Partitioning schema kind of a table.
 *
 * The partitioning schema kind declares how records are assigned to table
 * partitions.
 */
using type = std::variant<partitioning_schema_kind::automatic,
                          partitioning_schema_kind::none,
                          partitioning_schema_kind::key>;

};  // namespace partitioning_schema_kind

/**
 * @brief Statistics of a column.
 */
struct column_statistics {
  size_t column_id          = 0;  /// Column ID.
  int64_t compressed_size   = 0;  /// Compressed size in bytes of the column.
  int64_t uncompressed_size = 0;  /// Uncompressed size in bytes of the column.
};

/**
 * @brief Statistics of a table.
 */
struct table_statistics {
  int64_t num_rows      = 0;  /// Number of rows in the table.
  size_t num_row_groups = 0;  /// Number of row groups in the table.
  size_t num_columns    = 0;  /// Number of columns in the table.
  std::vector<size_t> compressed_num_row_groups =
    {};  /// Number of compressed row groups in the table for each column ID.
  std::vector<int64_t> compressed_size_per_column = {};  /// Compressed size in bytes per column ID.
  std::vector<int64_t> uncompressed_size_per_column =
    {};  /// Uncompressed size in bytes per column ID.

  table_statistics() = default;

  /**
   * @brief Initialize the table statistics with the given number of rows and columns.
   *
   * The initialization is performed along with `table_statistics_manager`.
   * We don't initialize the num_row_groups here because we don't know the number of row groups
   * during construction. And row groups are atomically updated later via `append_table_statistics`.
   *
   * @param num_rows_ number of rows
   * @param num_columns_ number of columns
   */
  table_statistics(int64_t num_rows_, size_t num_columns_);

  /**
   * @brief Add the column statistics for a given column to the current table statistics.
   *
   * @param col_stats column statistics to append
   */
  void add_column_statistics(const column_statistics& col_stats);

  /**
   * @brief Append a table statistics to the current one.
   *
   * Accumulate the number of rows, number of row groups, compressed and uncompressed sizes in
   * bytes, and number of compressed row groups.
   *
   * @param new_stats table statistics to append
   */
  void append_table_statistics(const table_statistics& new_stats);
};

class query_context;
class task_manager_context;
class context_reference;
class task;

/**
 * @brief Functor for generating tasks for user-defined relations.
 *
 * The first argument of the functor is a vector with the same length as the number of child
 * relations, where the `i`th element is a vector of tasks holding the output of the `i`th child
 * relation. These tasks serve as the input to the functor.
 *
 * The second argument is the context reference.
 *
 * The third argument is the task ID both as the input and as the output. When the functor
 * constructs a new task, it should use this argument as the task ID, and also increment this
 * argument.
 *
 * The fourth argument is the stage ID of the tasks generated by the functor.
 *
 * The return value of the functor is a vector of the output tasks.
 */
using user_defined_task_functor = std::function<std::vector<std::shared_ptr<task>>(
  std::vector<std::vector<std::shared_ptr<task>>>, context_reference ctx_ref, int32_t&, int32_t)>;

namespace window_frame_bound {

/**
 * @brief Windows which do not have finite bounds but instead extend to the end of the partition.
 */
struct unbounded {
  bool operator==(const unbounded& other) const { return true; }

  bool operator!=(const unbounded& other) const { return false; }
};

/**
 * @brief Windows which extend a finite number of rows before or after the current row.
 */
struct bounded {
  bounded(int64_t bound) : _bound(bound) {}

  [[nodiscard]] int64_t get_bound() const noexcept { return _bound; }

  bool operator==(const bounded& other) const { return _bound == other.get_bound(); }

  bool operator!=(const bounded& other) const { return _bound != other.get_bound(); }

 private:
  int64_t _bound;
};

using type = std::variant<unbounded, bounded>;

}  // namespace window_frame_bound

/**
 * @brief Indicate the algorithm used for compression.
 */
enum class compression_format : int8_t {
  none,                    ///< Uncompressed
  ans,                     ///< ANS compression
  lz4,                     ///< LZ4 compression
  snappy,                  ///< snappy compression
  gdeflate,                ///< GDeflate compression
  deflate,                 ///< Deflate compression
  cascaded,                ///< Cascaded compression
  zstd,                    ///< ZSTD compression
  gzip,                    ///< GZIP compression
  bitcomp,                 ///< Bitcomp compression
  best_compression_ratio,  ///< Choose the best compression algorithm based on column type to yield
                           ///< best compression ratio
  best_decompression_speed,  ///< Choose the best compression algorithm based on column type to
                             ///< yield best decompression speed
};

/**
 * @brief Construct compression_format from string representation
 *
 * @param format_str String representation of compression format
 *
 * @return compression_format from string representation
 *
 * @throw std::invalid_argument if format_str is not recognized
 */
compression_format compression_format_from_string(std::string const& format_str);

}  // namespace gqe
