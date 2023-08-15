/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task.hpp>

#include <rmm/cuda_device.hpp>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace gqe {

/**
 * @brief Enum to specify the type of a join.
 */
enum class join_type_type { inner, left, left_semi, left_anti, full, single };

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
   * @brief Return the raw bits of the set.
   */
  [[nodiscard]] const unsigned long* bits() const noexcept;

  /**
   * @brief Visualize the bits of the CPU set.
   *
   * This method is helpful for debugging.
   */
  [[nodiscard]] std::string pretty_print() const;

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
};

/**
 * @brief System memory allocated on a specific NUMA node.
 */
struct numa {
  cpu_set numa_node_set; /**< NUMA node identifier hint. */
  gqe::page_kind::type page_kind = gqe::page_kind::system_default; /**< Page kind hint */
};

/**
 * @brief Pinned system memory allocated with `cudaMallocHost`.
 */
struct pinned {
};

/**
 * @brief CUDA device memory.
 */
struct device {
  rmm::cuda_device_id device_id; /**< CUDA device identifier hint. */
};

/**
 * @brief CUDA managed memory.
 */
struct managed {
};

/**
 * @brief Memory kind of an in-memory table.
 */
using type = std::variant<memory_kind::system,
                          memory_kind::numa,
                          memory_kind::pinned,
                          memory_kind::device,
                          memory_kind::managed>;

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
                          storage_kind::parquet_file>;

}  // namespace storage_kind

namespace partitioning_schema_kind {

/**
 * @brief Automatic partitioning that infers the schema from the data.
 *
 * Inferrence of the partitioning schema is only possible for files. In-memory
 * storage defaults to `none`.
 */
struct automatic {
};

/**
 * @brief Do not partition data.
 */
struct none {
};

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
 * @brief Statistics of a table.
 */
struct table_statistics {
  int64_t num_rows;
};

/**
 * @brief Functor for generating tasks for user-defined relations.
 *
 * The first argument of the functor is a vector with the same length as the number of child
 * relations, where the `i`th element is a vector of tasks holding the output of the `i`th child
 * relation. These tasks serve as the input to the functor.
 *
 * The second argument is the query context.
 *
 * The third argument is the task ID both as the input and as the output. When the functor
 * constructs a new task, it should use this argument as the task ID, and also increment this
 * argument.
 *
 * The fourth argument is the stage ID of the tasks generated by the functor.
 *
 * The return value of the functor is a vector of the output tasks.
 */
using user_defined_task_functor =
  std::function<std::vector<std::shared_ptr<task>>(std::vector<std::vector<std::shared_ptr<task>>>,
                                                   query_context* query_context,
                                                   int32_t&,
                                                   int32_t)>;

namespace window_frame_bound {

/**
 * @brief Windows which do not have finite bounds but instead extend to the end of the partition.
 */
struct unbounded {
};

/**
 * @brief Windows which extend a finite number of rows before or after the current row.
 */
struct bounded {
  bounded(int64_t bound) : _bound(bound) {}

  [[nodiscard]] int64_t get_bound() const noexcept { return _bound; }

 private:
  int64_t _bound;
};

using type = std::variant<unbounded, bounded>;

}  // namespace window_frame_bound
}  // namespace gqe
