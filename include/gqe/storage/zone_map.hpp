/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/expression.hpp>

#include <boost/container/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/offset_ptr.hpp>

// Forward declarations
namespace gqe {
namespace storage {

class shared_table;
}  // namespace storage
}  // namespace gqe

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
  static std::unique_ptr<gqe::expression> transform(const gqe::expression& input_expression);

  void visit(const binary_op_expression* expression) override;
  void visit(const column_reference_expression* expression) override;
  void visit(const literal_expression<int8_t>* expression) override;
  void visit(const literal_expression<int32_t>* expression) override;
  void visit(const literal_expression<double>* expression) override;
  void visit(const literal_expression<cudf::timestamp_D>* expression) override;
  void visit(const literal_expression<std::string>* expression) override;
  void visit(const scalar_function_expression* expression) override;
  void visit(const unary_op_expression* expression) override;

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
  std::unordered_map<const gqe::expression*, std::shared_ptr<gqe::expression>>
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
 * @brief A zone map consisting of min/max values created for all columns in partitions of
 * a single `cudf::table_view`.
 *
 * Given a (transformed) filter expression, the zone map indicates which partitions of the
 * original table_view contain rows that satisfy the filter expression.
 *
 * The partitions/zones of the zone map are virtual, i.e., only the start offset, end offset, and
 * null_counts of the partitions of the original contiguous table view are stored. To actually
 * construct the partition, users have to create a table_view representing a slice of the original
 * input table.
 */
class zone_map {
  friend class shared_zone_map;
  friend class shared_zone_map_table;

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

  /**
   * @brief Create a zone map of a cudf::table_view with a given partition size.
   * @param table The input cudf::table_view.
   * @param partition_size The number of rows per partition.
   */
  explicit zone_map(const cudf::table_view& table, cudf::size_type partition_size);

  zone_map(const zone_map& other)      = delete;
  zone_map& operator=(const zone_map&) = delete;
  zone_map(zone_map&&)                 = default;
  zone_map& operator=(zone_map&&)      = default;
  virtual ~zone_map()                  = default;
  zone_map()                           = default;

  /**
   * Evaluate a filter expression on the zone map and indicate for each partition of the zone map if
   * it was pruned or not.
   *
   * The result is a vector containing information about all partitions of the zone map without any
   * gaps. This vector can be reduced using the method zone_map::consolidate_partitions.
   *
   * For example, if `partition_size` is 10, and the underlying cudf::table_view has 70 rows, the
   * result may look like this:
   * <pre>
   * | 0    9 | 10    19 | 20    29 | 30  39 | 40  49 | 50    59 | 60  69 |
   * | pruned | unpruned | unpruned | pruned | pruned | unpruned | pruned |
   * </pre>
   *
   * @param partial_filter The filter expression that should be evaluated.
   * @param[in] use_like_shift_and If `true`, use shift_and kernel for computing like filter.
   * @return A vector containing information about all partitions of the zone map, including start
   * and end offsets, and if the partitions was pruned or not.
   */
  [[nodiscard]] virtual std::vector<partition> evaluate(const gqe::expression& partial_filter,
                                                        bool use_like_shift_and);

  /**
   * @brief Consolidate maximal runs of partitions that are either pruned or not pruned.
   *
   * For example, let's assume that zone_map::evaluate returns the following vector of partitions:
   * <pre>
   * | 0    9 | 10    19 | 20    29 | 30  39 | 40  49 | 50    59 | 60  69 |
   * | pruned | unpruned | unpruned | pruned | pruned | unpruned | pruned |
   * </pre>
   *
   * Then this method will consolidate this vector as follows:
   * <pre>
   * | 0    9 | 10    29 | 30  49 | 50    59 | 60  69 |
   * | pruned | unpruned | pruned | unpruned | pruned |
   * </pre>
   *
   * Notice that (a) the result spans the same range of rows as the input, (b) that pruned and
   * unpruned partitions alternate, and (c) that the second and third partition both span two
   * partitions in the input vector that are either both unpruned or pruned.
   *
   * @param partitions A vector of pruned/unpruned partitions produced by evaluating a filter on a
   * zone map. Specifically, this vector can contain consecutive partitions with the same pruning
   * status.
   * @return A vector of pruned/unpruned partitions where multiple consecutive partitions with the
   * same pruning status are consolidated into a single partition.
   */
  static std::vector<partition> consolidate_partitions(const std::vector<partition>& partitions);

  /**
   * @brief Produce a single partition that covers all unpruned partitions, including any pruned
   * gaps.
   *
   * For example, let's assume that zone_map::evaluate returns the following vector of partitions:
   * <pre>
   * | 0    9 | 10    19 | 20    29 | 30  39 | 40  49 | 50    59 | 60  69 |
   * | pruned | unpruned | unpruned | pruned | pruned | unpruned | pruned |
   * </pre>
   *
   * Then this method will consolidate this vector as follows:
   * <pre>
   * | 10    59 |
   * | unpruned |
   * </pre>
   *
   * Notice that the result is a single unpruned partition that spans all unpruned partitions
   * in the input and includes the pruned partitions from 30 to 49 (pruned gaps).
   *
   * @param partitions A vector of pruned/unpruned partitions produced by evaluating a filter on a
   * zone map. Specifically, this vector can contain consecutive partitions with the same pruning
   * status.
   * @return A single partition that begins at the first unpruned partition of the input and ends at
   * the last unpruned partition.
   */
  static partition consolidate_maximally_covering_partition(
    const std::vector<partition>& partitions);

#ifndef NDEBUG
  /// Output zone map to a Parquet file for debugging
  void write_to_parquet_file(std::string_view filename) const;
#endif

 private:
  /// Consolidate consecutive runs of pruned/unpruned partitions.
  /// There must be no gaps between the partitions, otherwise the computed null counts are off.
  static partition consolidate_partitions(const std::vector<partition>::const_iterator begin,
                                          const std::vector<partition>::const_iterator end);

  /// Compute the cudf::table representing the zone map
  [[nodiscard]] static std::unique_ptr<cudf::table> compute_zone_map(
    const cudf::table_view& table,
    cudf::size_type partition_size,
    rmm::mr::device_memory_resource* mr);

  /// Compute the null counts for each zone map partitions.
  [[nodiscard]] static std::vector<std::vector<cudf::size_type>> compute_null_counts(
    const ::cudf::table_view& table, cudf::size_type partition_size);

  /// The size of each zone map partition.
  cudf::size_type _partition_size;

  /// The number of rows of the input table.
  cudf::size_type _num_rows;

  /// Memory resource used to allocate the cudf::table storing the zone map.
  std::unique_ptr<rmm::mr::device_memory_resource> _memory_resource;

  /// The actual zone map table.
  std::unique_ptr<cudf::table> _zone_map;

  /// Per-partition null counts for each column. The first vector is over the columns of the input
  /// table, the second vector is over the partitions of the zone map. I.e., _null_counts[i][j]
  /// contains the null counts for column i in partition j.
  std::vector<std::vector<cudf::size_type>> _null_counts;
};

/**
 * @brief A derived class of zone_map that can additionally find the shared zone map table on
 * CPU shared memory and copy it to device memory.
 */
class shared_zone_map : public zone_map {
 public:
  explicit shared_zone_map(cudf::size_type partition_size,
                           std::string table_name,
                           boost::interprocess::managed_shared_memory* segment);

  void copy_to_device(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::vector<partition> evaluate(const gqe::expression& partial_filter,
                                  bool use_like_shift_and) override;

  ~shared_zone_map();

 private:
  /// The name of the shared zone map table
  std::string _table_name;

  /// The managed shared memory segment.
  boost::interprocess::managed_shared_memory* _segment;
};

/**
 * @brief Zone map table with boost data structures that can be safely allocated and
 * accessed on inter-process CPU shared memory.
 */
class shared_zone_map_table {
 public:
  explicit shared_zone_map_table(
    const cudf::table_view& table,
    cudf::size_type partition_size,
    boost::interprocess::managed_shared_memory* segment,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

  ~shared_zone_map_table();
  /// The actual zone map table.
  boost::interprocess::offset_ptr<gqe::storage::shared_table> _zone_map;

  using SharedSizeTypeAllocator =
    boost::interprocess::allocator<cudf::size_type,
                                   boost::interprocess::managed_shared_memory::segment_manager>;
  using SharedVectorSizeTypeAllocator = boost::interprocess::allocator<
    boost::container::vector<cudf::size_type, SharedSizeTypeAllocator>,
    boost::interprocess::managed_shared_memory::segment_manager>;

  /// The number of rows of the input table.
  cudf::size_type _num_rows;

  /// The managed shared memory segment.
  boost::interprocess::managed_shared_memory* _segment;

  /// Per-partition null counts for each column. The first vector is over the columns of the input
  /// table, the second vector is over the partitions of the zone map. I.e., _null_counts[i][j]
  /// contains the null counts for column i in partition j.
  boost::container::vector<boost::container::vector<cudf::size_type, SharedSizeTypeAllocator>,
                           SharedVectorSizeTypeAllocator>
    _null_counts;
};

}  // namespace gqe
