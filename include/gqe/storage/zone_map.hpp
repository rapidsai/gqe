/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/storage/buffer.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/storage/storage_handle.hpp>
#include <gqe/types.hpp>

#include <cuda/stream>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <arrow/compute/expression.h>

#include <filesystem>
#include <optional>
#include <unordered_map>

namespace arrow {
class RecordBatch;
}  // namespace arrow

namespace gqe {

/**
 * @brief An expression visitor which transform an expression on a base table to an equivalent
 * expression that can be evaluated on the zone map for the base table.
 *
 * Basic transformation rules (COLUMN is a column of the base table, MIN and MAX are the
 * corresponding columns of the zone map):
 *
 * - COLUMN < value → MIN < value
 * - COLUMN <= value → MIN <= value
 * - COLUMN > value → MAX > value
 * - COLUMN >= value → MAX >= value
 * - COLUMN == value → MIN <= value AND MAX >= value
 *
 * If COLUMN is on the RHS, MIN and MAX are switched:
 *
 * - value < COLUMN → value < MAX
 * - value > COLUMN → value > MIN
 *
 * If a comparison is used inside a negation, MIN and MAX are switched:
 *
 * - NOT (COLUMN < value) → NOT (MAX < value)
 *
 * Support is currently limited to the expressions encountered in TPC-H queries. If an expression
 * encountered in TPC-H queries cannot be transformed (e.g., LIKE), it is ignored. Boolean AND and
 * OR correctly handle these unsupported expressions:
 *
 * - supported_expr AND unsupported_expr is replaced just with supported_expr
 * - supported_expr OR unsupported_expr is entirely ignored
 *
 * Expressions that are not encountered in TPC-H queries call the visitor of the base class
 * gqe::expression_visitor, which throws an exception.
 */
class zone_map_expression_transformer : public gqe::expression_visitor {
 public:
  /**
   * Transform an expression on a base table so that it can be evaluated on the corresponding zone
   * map.
   * @param input_expression The expression on a base table.
   * @return A transformed expression where columns references of the base table are replaced with
   * the corresponding MIN and MAX columns of the zone map; or nullptr if the expression could
   * not be transformed.
   */
  static std::optional<arrow::compute::Expression> transform(
    const gqe::expression& input_expression);

  void visit(const binary_op_expression* expression) override;
  void visit(const column_reference_expression* expression) override;
  void visit(const literal_expression<int8_t>* expression) override;
  void visit(const literal_expression<int32_t>* expression) override;
  void visit(const literal_expression<double>* expression) override;
  void visit(const literal_expression<cudf::timestamp_D>* expression) override;
  void visit(const literal_expression<std::string>* expression) override;
  void visit(const scalar_function_expression* expression) override;
  void visit(const unary_op_expression* expression) override;

  // The following visit methods are not tested. They are not needed for TPC-H and are therefore
  // ignored.
  void visit(const literal_expression<int64_t>* expression) override;
  void visit(const literal_expression<float>* expression) override;
  void visit(const literal_expression<numeric::decimal32>* expression) override;
  void visit(const literal_expression<numeric::decimal64>* expression) override;
  void visit(const literal_expression<numeric::decimal128>* expression) override;
  void visit(const subquery_expression* expression) override;
  void visit(const if_then_else_expression* expression) override;
  void visit(const cast_expression* expression) override;
  void visit(const is_null_expression* expression) override;

  ~zone_map_expression_transformer() override = default;

 private:
  // Make the constructor private, so that it can only be created by the method transform.
  // This ensures that each transformer is only used for the transformation of a single expression.
  zone_map_expression_transformer() {};

  // The following methods handle different binary operators to reduce the complexity of the visit
  // method.

  void visitEqualExpression(const binary_op_expression* expression,
                            const gqe::expression* lhs,
                            const gqe::expression* rhs);
  void visitNotEqualExpression(const binary_op_expression* expression);
  void visitComparisonExpression(const binary_op_expression* expression,
                                 const cudf::binary_operator op,
                                 const gqe::expression* lhs,
                                 const gqe::expression* rhs);
  void visitAndExpression(const binary_op_expression* expression,
                          const gqe::expression* lhs,
                          const gqe::expression* rhs);
  void visitOrExpression(const binary_op_expression* expression,
                         const gqe::expression* lhs,
                         const gqe::expression* rhs);

  /// Track the currently used comparison operator.
  void transform_comparisons(cudf::binary_operator op, const std::function<void()>& function);

  /// Switch between MIN and MAX columns for comparisons inside negations.
  void negate_comparisons(const std::function<void()>& function);

  /// Map expressions contained inside the input expression on the base table to their
  /// corresponding transformed expressions on the zone map.
  std::unordered_map<const gqe::expression*, std::optional<arrow::compute::Expression>>
    _transformation_map{};

  /// Encode the position of a column reference inside a comparison expression.
  enum class side { lhs, rhs };

  /// Track whether a column is used on the LHS or RHS of a comparison to determine whether the MIN
  /// or MAX column is selected.
  side _side = side::lhs;

  /// Track the comparison operator to determine whether the MIN or MAX column is selected for a
  /// column reference. This field should not be manipulated directly; instead wrap transformation
  /// code inside the method transform_comparisons.
  std::optional<cudf::binary_operator> _current_comparison;

  /// Negations influence whether the MIN or MAX column is selected for a column reference.
  /// A simple flag is enough to track negations, because nested negations cancel each other out.
  /// This field should not be manipulated directly; instead wrap negated transformations
  /// inside the method negate_comparisons to ensure that this flag is restored after the evaluation
  /// of a negated expression.
  bool _inside_negation = false;

  /// Some transformation insert new operators into the expression tree, for example ==.
  /// These inserted operators need to be kept valid during the lifetime of the transformer.
  std::vector<std::shared_ptr<gqe::expression>> _expanded_expressions;
};

/**
 * @brief Allocator-aware persistent zone-map storage.
 *
 * A zone map contains min/max values for every column in fixed-size partitions of an input
 * `cudf::table_view`. The partitions are virtual: this storage records partition boundaries,
 * per-column null counts, and a serialized Arrow IPC record batch for the min/max table. Callers
 * can reconstruct runtime Arrow objects from this persistent representation when evaluating
 * filters.
 *
 * This type is allocator-aware and contains no virtual functions so it can be constructed in the
 * shared memory segment.
 *
 * @tparam Allocator Allocator used for persistent host-side metadata containers.
 */
template <class Allocator = std::allocator<std::byte>>
class zone_map_storage {
 public:
  using allocator_type = gqe::storage_allocator_t<Allocator>;
  template <class T>
  using vector_type       = gqe::storage_vector_t<Allocator, T>;
  using null_count_vector = vector_type<cudf::size_type>;
  using null_counts_type  = vector_type<null_count_vector>;

  /**
   * @brief Create zone-map storage for an input table.
   *
   * @param table Input table view used to compute min/max values and null counts.
   * @param partition_size Number of rows per zone-map partition.
   * @param stream CUDA stream used for cuDF work and storage copies.
   * @param compute_mr Device memory resource used while computing the zone map.
   * @param storage_mr Memory resource used for backing serialized storage buffers.
   * @param alloc Allocator for persistent host-side metadata containers.
   */
  zone_map_storage(cudf::table_view const& table,
                   cudf::size_type partition_size,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref compute_mr,
                   rmm::device_async_resource_ref storage_mr,
                   allocator_type const& alloc = allocator_type{});

  /**
   * @brief Reconstruct zone-map storage from deserialized data.
   *
   * @param partition_size Rows per zone-map partition.
   * @param num_rows Row count of the source table.
   * @param null_counts Per-partition, per-column null counts, i.e.,
   * `null_counts[partition][column]`.
   * @param ipc_record_batch Serialized Arrow IPC min/max record-batch bytes to copy into storage.
   * @param stream CUDA stream for the storage buffer copy.
   * @param storage_mr Memory resource backing the serialized IPC storage buffer.
   * @param alloc Allocator for persistent host-side metadata containers.
   */
  zone_map_storage(cudf::size_type partition_size,
                   cudf::size_type num_rows,
                   std::vector<std::vector<cudf::size_type>> const& null_counts,
                   std::span<std::byte const> ipc_record_batch,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref storage_mr,
                   allocator_type const& alloc = allocator_type{});

  zone_map_storage(zone_map_storage const&)                = delete;
  zone_map_storage& operator=(zone_map_storage const&)     = delete;
  zone_map_storage(zone_map_storage&&) noexcept            = default;
  zone_map_storage& operator=(zone_map_storage&&) noexcept = default;

  /**
   * @brief Explicitly release buffers owned by this storage.
   *
   * This is a terminal-use API. Callers must not evaluate the zone map through wrappers referencing
   * this storage after deallocation.
   */
  void deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Return the serialized Arrow IPC record batch containing zone-map min/max values.
   */
  [[nodiscard]] std::span<std::byte const> ipc_record_batch() const
  {
    return {static_cast<std::byte const*>(_ipc_record_batch.data()), _ipc_record_batch.size()};
  }

  /**
   * @brief Return the number of rows per zone-map partition.
   */
  [[nodiscard]] cudf::size_type partition_size() const noexcept { return _partition_size; }

  /**
   * @brief Return the number of rows in the source table.
   */
  [[nodiscard]] cudf::size_type num_rows() const noexcept { return _num_rows; }

  /**
   * @brief Return per-partition null counts for each source column.
   */
  [[nodiscard]] null_counts_type const& null_counts() const noexcept { return _null_counts; }

 private:
  /// Serialized Arrow IPC record batch containing the zone-map min/max columns.
  gqe::storage::buffer _ipc_record_batch;

  /// The size of each zone-map partition.
  cudf::size_type _partition_size{};

  /// The number of rows of the input table.
  cudf::size_type _num_rows{};

  /// Per-partition null counts for each column. The first vector is over the partitions of the zone
  /// map, the second vector is over the columns of the input table. I.e., _null_counts[i][j]
  /// contains the null count for column j in partition i.
  null_counts_type _null_counts;
};

using shared_zone_map_storage = zone_map_storage<gqe::storage::shared_memory_allocator>;

/**
 * @brief Runtime interface for zone maps.
 *
 * Given a transformed filter expression, a zone map indicates which partitions of the original
 * table may contain rows that satisfy the filter. A partition marked as pruned can be skipped; an
 * unpruned partition may still contain rows that fail the full predicate and must be filtered
 * later.
 *
 * The partitions are virtual, i.e., only start offset, end offset, and null counts are returned.
 * To materialize a partition, callers slice the original row-group columns using those offsets.
 */
class zone_map {
 public:
  /**
   * @brief Structure indicating the start offset, end offset, and null counts of a partition and
   * whether this partition was pruned by the filter expression.
   */
  struct partition {
    /// True, if the partition was pruned by a filter expression; false, otherwise.
    bool pruned = false;

    /// The start offset (inclusive) of a virtual zone map partition.
    cudf::size_type start;

    /// The end offset (exclusive) of a virtual zone map partition.
    cudf::size_type end;

    /// Null counts for each column of the virtual zone map partition.
    std::vector<cudf::size_type> null_counts{};

    bool operator==(const partition& other) const;
  };

  zone_map(const zone_map& other)      = delete;
  zone_map& operator=(const zone_map&) = delete;
  zone_map(zone_map&&)                 = default;
  zone_map& operator=(zone_map&&)      = default;
  virtual ~zone_map()                  = default;
  zone_map()                           = default;

  /**
   * @brief Explicitly release backing storage associated with this zone map.
   */
  virtual void deallocate(cuda::stream_ref, rmm::device_async_resource_ref) {}

  /**
   * Evaluate a filter expression on the zone map and indicate for each partition whether it was
   * pruned.
   *
   * The result contains information about all partitions of the zone map without gaps. This vector
   * can be reduced using zone_map::consolidate_partitions.
   *
   * For example, if `partition_size` is 10 and the underlying table has 70 rows, the result may
   * look like this:
   * <pre>
   * | 0    9 | 10    19 | 20    29 | 30  39 | 40  49 | 50    59 | 60  69 |
   * | pruned | unpruned | unpruned | pruned | pruned | unpruned | pruned |
   * </pre>
   *
   * @param partial_filter Filter expression transformed for zone-map evaluation.
   * @return Information about every partition, including start and end offsets, null counts, and
   * pruning status.
   */
  [[nodiscard]] virtual std::vector<partition> evaluate(
    const arrow::compute::Expression& partial_filter) const = 0;

  /**
   * @brief Serialize this zone map to `<zone_map_dir>/zone_map.bin`.
   *
   * Writes the partition size, row count, per-column/partition null counts, and the Arrow IPC
   * min/max record batch. This is host-only work.
   */
  virtual void serialize_to_disk(std::filesystem::path const& zone_map_dir) const = 0;

  /**
   * @brief Consolidate maximal runs of partitions that are either pruned or not pruned.
   *
   * For example, if zone_map::evaluate returns:
   * <pre>
   * | 0    9 | 10    19 | 20    29 | 30  39 | 40  49 | 50    59 | 60  69 |
   * | pruned | unpruned | unpruned | pruned | pruned | unpruned | pruned |
   * </pre>
   *
   * Then this method consolidates it as follows:
   * <pre>
   * | 0    9 | 10    29 | 30  49 | 50    59 | 60  69 |
   * | pruned | unpruned | pruned | unpruned | pruned |
   * </pre>
   *
   * The result spans the same row range as the input and merges partitions only when they are
   * contiguous and have the same pruning status.
   *
   * @param partitions Partitions produced by evaluating a filter on a zone map.
   * @return A vector where consecutive partitions with the same pruning status are consolidated.
   * @note Consolidation can also be broken by a gap in row coverage between two consecutive,
   * equally-pruned partitions (e.g. rows removed upstream); such a gap ends a group even though the
   * pruning status on both sides is the same.
   */
  static std::vector<partition> consolidate_partitions(const std::vector<partition>& partitions);

  /**
   * @brief Produce a single partition covering all unpruned partitions, including pruned gaps.
   *
   * For example, if zone_map::evaluate returns:
   * <pre>
   * | 0    9 | 10    19 | 20    29 | 30  39 | 40  49 | 50    59 | 60  69 |
   * | pruned | unpruned | unpruned | pruned | pruned | unpruned | pruned |
   * </pre>
   *
   * Then this method consolidates it as follows:
   * <pre>
   * | 10    59 |
   * | unpruned |
   * </pre>
   *
   * The result is a single unpruned partition spanning from the first unpruned partition to the
   * last unpruned partition, including any pruned gaps between them.
   *
   * @param partitions Partitions produced by evaluating a filter on a zone map.
   * @return A single partition that begins at the first unpruned partition and ends at the last
   * unpruned partition.
   */
  static partition consolidate_maximally_covering_partition(
    const std::vector<partition>& partitions);

#ifndef NDEBUG
  /// Output zone map to a Parquet file for debugging
  virtual void write_to_parquet_file(std::string_view filename) const = 0;
#endif

 private:
  /// Consolidate consecutive runs of pruned/unpruned partitions.
  /// There must be no gaps between the partitions, otherwise the computed null counts are off.
  static partition consolidate_partitions(const std::vector<partition>::const_iterator begin,
                                          const std::vector<partition>::const_iterator end);
};

/**
 * @brief Shared zone-map behavior for local and shared storage handles.
 *
 * This wrapper owns the runtime cache used for evaluation. The storage handle decides whether the
 * persistent zone-map storage is owned locally or resolved from the shared memory segment.
 *
 * @tparam StorageHandle Handle type that exposes the concrete zone-map storage.
 */
template <class StorageHandle>
class basic_zone_map : public zone_map {
 public:
  using storage_type = typename StorageHandle::storage_type;

  [[nodiscard]] std::vector<partition> evaluate(
    const arrow::compute::Expression& partial_filter) const override;

  void serialize_to_disk(std::filesystem::path const& zone_map_dir) const override;

  void deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr) override;

#ifndef NDEBUG
  void write_to_parquet_file(std::string_view filename) const override;
#endif

 protected:
  // Forward construction args to storage handle to hide details about storage from zone-map
  // wrapper.
  template <class... Args>
  explicit basic_zone_map(Args&&... args);

  StorageHandle _storage_handle;

 private:
  /// Runtime Arrow RecordBatch reconstructed from persistent zone-map storage.
  std::shared_ptr<arrow::RecordBatch> _zone_map_cache;
};

/**
 * @brief Process-local zone-map wrapper.
 */
class local_zone_map final
  : public basic_zone_map<gqe::storage::local_storage_handle<zone_map_storage<>>> {
 public:
  using storage_type        = zone_map_storage<>;
  using storage_handle_type = gqe::storage::local_storage_handle<storage_type>;
  using base_type           = basic_zone_map<storage_handle_type>;

  /**
   * @brief Create a process-local zone map for an input table.
   *
   * @param table Input table view used to compute min/max values and null counts.
   * @param partition_size Number of rows per zone-map partition.
   * @param stream CUDA stream used for cuDF work and storage copies.
   * @param compute_mr Device memory resource used while computing the zone map.
   */
  explicit local_zone_map(
    cudf::table_view const& table,
    cudf::size_type partition_size,
    cuda::stream_ref stream                   = cudf::get_default_stream(),
    rmm::device_async_resource_ref compute_mr = rmm::mr::get_current_device_resource_ref());

  /// Reconstruct a process-local zone map from deserialized data.
  local_zone_map(cudf::size_type partition_size,
                 cudf::size_type num_rows,
                 std::vector<std::vector<cudf::size_type>> const& null_counts,
                 std::span<std::byte const> ipc_record_batch,
                 cuda::stream_ref stream);

  /**
   * @brief Reconstruct a zone map previously written by @ref zone_map::serialize_to_disk.
   *
   * @return The reconstructed zone map, or nullptr if `<zone_map_dir>/zone_map.bin` does not exist
   *         (e.g. the row group was serialized without a zone map).
   */
  [[nodiscard]] static std::unique_ptr<zone_map> deserialize_from_disk(
    std::filesystem::path const& zone_map_dir, cuda::stream_ref stream);

  void deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr) override;
};

/**
 * @brief Process-local wrapper over zone-map storage in the shared memory segment.
 *
 * The wrapper resolves an existing shared-memory storage object by handle. Destroying the wrapper
 * does not destroy the shared storage; call deallocate() through the row-group/table teardown path
 * when the shared storage should be released.
 */
class shared_zone_map final
  : public basic_zone_map<gqe::storage::shared_storage_handle<shared_zone_map_storage>> {
 public:
  using storage_type        = shared_zone_map_storage;
  using storage_handle_type = gqe::storage::shared_storage_handle<storage_type>;
  using base_type           = basic_zone_map<storage_handle_type>;

  shared_zone_map(shared_segment& segment, typename storage_handle_type::handle_type handle);
};

extern template class zone_map_storage<std::allocator<std::byte>>;
extern template class zone_map_storage<gqe::storage::shared_memory_allocator>;
extern template class basic_zone_map<gqe::storage::local_storage_handle<zone_map_storage<>>>;
extern template class basic_zone_map<gqe::storage::shared_storage_handle<shared_zone_map_storage>>;

/**
 * @brief Reconstruct a serialized row-group zone map directly into the shared memory segment.
 *
 * Parses `<zone_map_dir>/zone_map.bin` and constructs a `shared_zone_map_storage` in @p segment via
 * `segment.construct`, returning a handle to it. The min/max (IPC) buffers are backed by @p
 * storage_mr, which must be the segment-backed resource so the zone map lives in the segment.
 *
 * @return The segment handle of the constructed storage, or nullopt if the file is absent (the row
 * group was serialized without a zone map).
 */
[[nodiscard]] std::optional<shared_memory_handle_t> deserialize_zone_map_into_segment(
  shared_segment& segment,
  gqe::storage::shared_memory_allocator const& alloc,
  std::filesystem::path const& zone_map_dir,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref storage_mr);

}  // namespace gqe
