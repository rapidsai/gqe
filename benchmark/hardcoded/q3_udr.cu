/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/catalog.hpp>
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/user_defined.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/utility/helpers.hpp>

#include <cuco/static_map.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>

#include <thrust/for_each.h>
#include <thrust/pair.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

void print_usage()
{
  std::cout << "Run TPC-DS Q3 benchmark with customized kernels" << std::endl
            << "./q3_udr <path-to-dataset>" << std::endl;
}

std::shared_ptr<gqe::logical::read_relation> read_table(
  std::string table_name,
  std::vector<std::string> column_names,
  gqe::catalog const* tpcds_catalog,
  std::shared_ptr<gqe::logical::project_relation> partial_filter_haystack = nullptr,
  std::unique_ptr<gqe::expression> partial_filter                         = nullptr)
{
  std::vector<cudf::data_type> column_types;
  column_types.reserve(column_names.size());
  for (auto const& column_name : column_names)
    column_types.push_back(tpcds_catalog->column_type(table_name, column_name));
  return std::make_shared<gqe::logical::read_relation>(
    partial_filter
      ? std::vector<std::shared_ptr<gqe::logical::relation>>{std::move(partial_filter_haystack)}
      : std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(column_names),
    std::move(column_types),
    std::move(table_name),
    std::move(partial_filter));  // partial_filter
}

// Use static_map instead of static_multimap because the join key is a primary key
// The hash map stores (join_key, row_idx) pairs.
using join_hash_map_type = cuco::static_map<int64_t, int64_t>;
// Assume the join keys are never max int64_t.
// This is likely okay since Spark uses 4-byte signed integers to represent `d_date_sk` and
// `i_item_sk`.
constexpr cuco::sentinel::empty_key<int64_t> empty_key_sentinel(
  std::numeric_limits<int64_t>::max());
// Row indices are non-negative in nature.
constexpr cuco::sentinel::empty_value<int64_t> empty_value_sentinel(-1);

// A customized task to use a fused kernel for the three-way join between "date_dim", "item" and
// "store_sales" tables.
class custom_task : public gqe::task {
 public:
  custom_task(gqe::query_context* query_context,
              int32_t task_id,
              int32_t stage_id,
              std::shared_ptr<gqe::task> date_dim_table,
              std::shared_ptr<gqe::task> item_table,
              std::shared_ptr<gqe::task> store_sales_table);

  void execute() override;
};

custom_task::custom_task(gqe::query_context* query_context,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<gqe::task> date_dim_table,
                         std::shared_ptr<gqe::task> item_table,
                         std::shared_ptr<gqe::task> store_sales_table)
  : gqe::task(query_context,
              task_id,
              stage_id,
              {std::move(date_dim_table), std::move(item_table), std::move(store_sales_table)},
              {})
{
}

// Number of threads in a warp
constexpr int warp_size = 32;

/*
 * Fused kernel for the three-way join between "date_dim", "item" and "store_sales" tables.
 *
 * Before running the kernel, we need to build hash maps for the item and the date_dim tables. Then,
 * warp `idx` is assigned to part of the store_sales table with row indices in
 * [idx * in_rows_per_warp, (idx + 1) * in_rows_per_warp). The warp writes the result to
 * *item_indices*, *date_dim_indices*, *ss_indices* starting at row index `idx * in_rows_per_warp`.
 * The warp will not write beyond `(idx + 1) * in_rows_per_warp` because the static map can generate
 * at most one output. The number of output rows for each warp is stored in *out_rows_per_warp*.
 */
template <int block_size>
__global__ void probe_hash_maps(join_hash_map_type::device_view item_map,
                                join_hash_map_type::device_view date_dim_map,
                                cudf::column_device_view ss_item_sk_column,
                                cudf::column_device_view ss_sold_date_sk_column,
                                cudf::size_type const in_rows_per_warp,
                                int64_t* item_indices,
                                int64_t* date_dim_indices,
                                int64_t* ss_indices,
                                cudf::size_type* out_rows_per_warp)
{
  __shared__ cudf::size_type warp_out_row_idx[block_size / warp_size];
  __shared__
    typename cub::WarpScan<int8_t>::TempStorage warp_scan_temp_storage[block_size / warp_size];

  int const global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int const global_warp_id   = global_thread_id / warp_size;
  int const local_warp_id    = threadIdx.x / warp_size;
  int const warp_lane        = global_thread_id % warp_size;

  // The current warp is responsible for processing rows of the store_sales table in [start_idx,
  // end_idx)
  cudf::size_type const start_idx = global_warp_id * in_rows_per_warp;
  cudf::size_type const end_idx =
    min((global_warp_id + 1) * in_rows_per_warp, ss_item_sk_column.size());

  if (warp_lane == 0) warp_out_row_idx[local_warp_id] = start_idx;
  __syncwarp();

  for (cudf::size_type warp_row_idx = start_idx; warp_row_idx < end_idx;
       warp_row_idx += warp_size) {
    // Note: Since this code block contains whole-warp scan, it is important that all threads in the
    // warp reach here to avoid deadlock.

    // Row index in the store_sales table to be processed by the current thread
    auto const thread_row_idx = warp_row_idx + warp_lane;
    int8_t num_matches        = 0;  // 0 for no matches, 1 for a match
    int64_t item_idx          = -1;
    int64_t date_dim_idx      = -1;

    if (thread_row_idx < end_idx) {
      // We first probe the item table because it is a more selective filter
      auto const item_tuple = item_map.find(ss_item_sk_column.element<int64_t>(thread_row_idx));

      // If we find a match in the item table, we continue to probe the date_dim table
      if (item_tuple != item_map.end() && ss_sold_date_sk_column.is_valid(thread_row_idx)) {
        auto const date_dim_tuple =
          date_dim_map.find(ss_sold_date_sk_column.element<int64_t>(thread_row_idx));
        if (date_dim_tuple != date_dim_map.end()) {
          item_idx     = item_tuple->second;
          date_dim_idx = date_dim_tuple->second;
          num_matches  = 1;
        }
      }
    }

    // Use a whole-warp scan to calculate the output location
    int8_t out_offset;
    cub::WarpScan<int8_t>(warp_scan_temp_storage[local_warp_id])
      .ExclusiveSum(num_matches, out_offset);

    if (num_matches == 1) {
      const auto thread_out_row_idx        = warp_out_row_idx[local_warp_id] + out_offset;
      item_indices[thread_out_row_idx]     = item_idx;
      date_dim_indices[thread_out_row_idx] = date_dim_idx;
      ss_indices[thread_out_row_idx]       = thread_row_idx;
    }
    __syncwarp();

    if (warp_lane == warp_size - 1) warp_out_row_idx[local_warp_id] += (out_offset + num_matches);
    __syncwarp();
  }

  if (warp_lane == 0)
    out_rows_per_warp[global_warp_id] = warp_out_row_idx[local_warp_id] - start_idx;
}

/*
 * Kernel for copying multiple chunks simultaneously. For chunk `idx`, the kernel copies memory from
 * `in_array + in_elements_per_chunk * idx` to `out_array + out_chunk_offsets[idx - 1]`, with size
 * `out_chunk_offsets[idx] - out_chunk_offsets[idx - 1]`.
 */
__global__ void stream_compaction(int64_t const* in_array,
                                  int64_t* out_array,
                                  cudf::size_type const in_elements_per_chunk,
                                  cudf::size_type const* out_chunk_offsets,
                                  int num_chunks)
{
  for (int chunk_idx = blockIdx.x; chunk_idx < num_chunks; chunk_idx += gridDim.x) {
    auto const out_offset_start = chunk_idx == 0 ? 0 : out_chunk_offsets[chunk_idx - 1];
    auto const out_offset_end   = out_chunk_offsets[chunk_idx];
    auto const chunk_in_offset  = in_elements_per_chunk * chunk_idx;

    for (cudf::size_type out_offset = out_offset_start + threadIdx.x; out_offset < out_offset_end;
         out_offset += blockDim.x) {
      out_array[out_offset] = in_array[chunk_in_offset + out_offset - out_offset_start];
    }
  }
}

void custom_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();

  auto date_dim_table    = dependent_tasks[0]->result().value();
  auto item_table        = dependent_tasks[1]->result().value();
  auto store_sales_table = dependent_tasks[2]->result().value();

  auto constexpr load_factor          = 0.5;
  std::size_t const date_dim_capacity = std::ceil(date_dim_table.num_rows() / load_factor);
  std::size_t const item_capacity     = std::ceil(item_table.num_rows() / load_factor);

  // Construct a hash map for the date_dim table
  join_hash_map_type date_dim_map(date_dim_capacity, empty_key_sentinel, empty_value_sentinel);

  auto d_date_sk_column = cudf::column_device_view::create(date_dim_table.column(0));

  thrust::for_each(
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(d_date_sk_column->size()),
    [map  = date_dim_map.get_device_mutable_view(),
     keys = *d_date_sk_column] __device__(auto row_idx) mutable {
      // We don't need to check for NULLs because "d_date_sk" is not NULL per TPC-DS specification.
      map.insert(thrust::pair<int64_t, int64_t>(keys.element<int64_t>(row_idx), row_idx));
    });

  // Similarly, construct a hash map for the item table
  join_hash_map_type item_map(item_capacity, empty_key_sentinel, empty_value_sentinel);

  auto i_item_sk_column = cudf::column_device_view::create(item_table.column(0));

  thrust::for_each(
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(i_item_sk_column->size()),
    [map  = item_map.get_device_mutable_view(),
     keys = *i_item_sk_column] __device__(auto row_idx) mutable {
      // We don't need to check for NULLs because "i_item_sk" is not NULL per TPC-DS specification.
      map.insert(thrust::pair<int64_t, int64_t>(keys.element<int64_t>(row_idx), row_idx));
    });

  // Probe the two hash maps in a fused kernel
  // We don't know exactly how many rows will be produced a priori, but we know it is capped by the
  // number of rows in the store_sales table.
  auto const ss_num_rows = store_sales_table.num_rows();
  rmm::device_uvector<int64_t> item_indices(ss_num_rows, rmm::cuda_stream_default);
  rmm::device_uvector<int64_t> date_dim_indices(ss_num_rows, rmm::cuda_stream_default);
  rmm::device_uvector<int64_t> ss_indices(ss_num_rows, rmm::cuda_stream_default);

  auto ss_item_sk_column      = cudf::column_device_view::create(store_sales_table.column(0));
  auto ss_sold_date_sk_column = cudf::column_device_view::create(store_sales_table.column(1));

  // Thread block size of the "probe_hash_maps" kernel, must be a multiple of `warp_size`
  constexpr int block_size = 128;
  // Number of threadblocks for the "probe_hash_maps" kernel, chosen based on experiments
  constexpr int grid_size     = 1600;
  constexpr auto num_warps    = grid_size * (block_size / warp_size);
  auto const in_rows_per_warp = (ss_num_rows + num_warps - 1) / num_warps;
  rmm::device_uvector<cudf::size_type> out_rows_per_warp(num_warps, rmm::cuda_stream_default);

  probe_hash_maps<block_size><<<grid_size, block_size>>>(item_map.get_device_view(),
                                                         date_dim_map.get_device_view(),
                                                         *ss_item_sk_column,
                                                         *ss_sold_date_sk_column,
                                                         in_rows_per_warp,
                                                         item_indices.data(),
                                                         date_dim_indices.data(),
                                                         ss_indices.data(),
                                                         out_rows_per_warp.data());
  rmm::cuda_stream_default.synchronize();

  // Turn the number of rows into output offsets for each warp
  thrust::inclusive_scan(
    thrust::device, out_rows_per_warp.begin(), out_rows_per_warp.end(), out_rows_per_warp.begin());

  // Allocate vectors for the compacted row indices
  auto const out_rows_total = out_rows_per_warp.back_element(rmm::cuda_stream_default);
  rmm::device_uvector<int64_t> compact_item_indices(out_rows_total, rmm::cuda_stream_default);
  rmm::device_uvector<int64_t> compact_date_dim_indices(out_rows_total, rmm::cuda_stream_default);
  rmm::device_uvector<int64_t> compact_ss_indices(out_rows_total, rmm::cuda_stream_default);

  // FIXME: Use DeviceBatchMemcpy from CUB instead of customized kernel after 2.1.0
  // https://github.com/NVIDIA/cub/pull/359
  stream_compaction<<<grid_size, block_size>>>(item_indices.data(),
                                               compact_item_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);

  stream_compaction<<<grid_size, block_size>>>(date_dim_indices.data(),
                                               compact_date_dim_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);

  stream_compaction<<<grid_size, block_size>>>(ss_indices.data(),
                                               compact_ss_indices.data(),
                                               in_rows_per_warp,
                                               out_rows_per_warp.data(),
                                               num_warps);

  rmm::cuda_stream_default.synchronize();

  // Convert indices from vectors to columns
  auto item_indices_column =
    std::make_unique<cudf::column>(std::move(compact_item_indices), rmm::device_buffer{}, 0);
  auto date_dim_indices_column =
    std::make_unique<cudf::column>(std::move(compact_date_dim_indices), rmm::device_buffer{}, 0);
  auto ss_indices_column =
    std::make_unique<cudf::column>(std::move(compact_ss_indices), rmm::device_buffer{}, 0);

  // Materialize the join output
  auto materialize_column =
    [](cudf::table_view input_table, cudf::size_type column_idx, cudf::column_view gather_map) {
      auto gathered_column = cudf::gather(input_table.select({column_idx}), gather_map)->release();
      return std::move(gathered_column[0]);
    };

  std::vector<std::unique_ptr<cudf::column>> out_columns;
  out_columns.push_back(materialize_column(store_sales_table, 2, ss_indices_column->view()));
  out_columns.push_back(materialize_column(date_dim_table, 1, date_dim_indices_column->view()));
  out_columns.push_back(materialize_column(item_table, 1, item_indices_column->view()));
  out_columns.push_back(materialize_column(item_table, 2, item_indices_column->view()));

  emit_result(std::make_unique<cudf::table>(std::move(out_columns)));
  remove_dependencies();
}

// Functor for generating the output tasks from input tasks
std::vector<std::shared_ptr<gqe::task>> custom_relation_generate_tasks(
  std::vector<std::vector<std::shared_ptr<gqe::task>>> children_tasks,
  gqe::query_context* query_context,
  int32_t& task_id,
  int32_t stage_id)
{
  auto date_dim_table =
    std::make_shared<gqe::concatenate_task>(query_context, task_id, stage_id, children_tasks[0]);
  task_id++;

  auto item_table =
    std::make_shared<gqe::concatenate_task>(query_context, task_id, stage_id, children_tasks[1]);
  task_id++;

  std::vector<std::shared_ptr<gqe::task>> pipeline_results;
  for (auto const& sales_table : children_tasks[2]) {
    pipeline_results.push_back(std::make_shared<custom_task>(
      query_context, task_id, stage_id, date_dim_table, item_table, sales_table));
    task_id++;
  }

  return pipeline_results;
}

int main(int argc, char* argv[])
{
  // Parse the command line arguments to get the dataset location
  if (argc != 2) {
    print_usage();
    return EXIT_FAILURE;
  }
  std::string const dataset_location(argv[1]);

  // Configure the memory pool
  // FIXME: For multi-GPU, we need to construct a memory pool for each device
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
  rmm::mr::set_current_device_resource(&pool_mr);

  // Register the input tables
  gqe::catalog tpcds_catalog;
  tpcds_catalog.register_table("store_sales",
                               {{"ss_item_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_sold_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"ss_ext_sales_price", cudf::data_type(cudf::type_id::FLOAT64)}},
                               gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(
                                 dataset_location + "/store_sales")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table("date_dim",
                               {{"d_date_sk", cudf::data_type(cudf::type_id::INT64)},
                                {"d_year", cudf::data_type(cudf::type_id::INT64)},
                                {"d_moy", cudf::data_type(cudf::type_id::INT64)}},
                               gqe::storage_kind::parquet_file{
                                 gqe::utility::get_parquet_files(dataset_location + "/date_dim")},
                               gqe::partitioning_schema_kind::automatic{});
  tpcds_catalog.register_table(
    "item",
    {{"i_item_sk", cudf::data_type(cudf::type_id::INT64)},
     {"i_brand_id", cudf::data_type(cudf::type_id::INT64)},
     {"i_brand", cudf::data_type(cudf::type_id::STRING)},
     {"i_manufact_id", cudf::data_type(cudf::type_id::INT64)}},
    gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(dataset_location + "/item")},
    gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  std::shared_ptr<gqe::logical::relation> date_dim_table =
    read_table("date_dim", {"d_date_sk", "d_year", "d_moy"}, &tpcds_catalog);
  date_dim_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(date_dim_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<int64_t>>(11)));

  std::shared_ptr<gqe::logical::relation> item_table =
    read_table("item", {"i_item_sk", "i_brand_id", "i_brand", "i_manufact_id"}, &tpcds_catalog);
  item_table = std::make_shared<gqe::logical::filter_relation>(
    std::move(item_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(3),
      std::make_shared<gqe::literal_expression<int64_t>>(128)));

  // predicate pushdown via partial filter
  std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
  col_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

  auto const partial_filter_haystack = std::make_shared<gqe::logical::project_relation>(
    date_dim_table,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery relations
    std::move(col_0_exprs));

  auto partial_filter = std::make_unique<gqe::in_predicate_expression>(
    std::vector<std::shared_ptr<gqe::expression>>{
      std::make_shared<gqe::column_reference_expression>(1)},  // ss_sold_date_sk
    0);

  std::shared_ptr<gqe::logical::relation> store_sales_table =
    read_table("store_sales",
               {"ss_item_sk", "ss_sold_date_sk", "ss_ext_sales_price"},
               &tpcds_catalog,
               std::move(partial_filter_haystack),
               std::move(partial_filter));

  // Use the user-defined relation for the three-way join between "date_dim", "item" and
  // "store_sales" tables.
  // After this operation, store_sales_table contains columns
  // ["ss_ext_sales_price", "d_year", "i_brand_id", "i_brand"]
  store_sales_table = std::make_shared<gqe::logical::user_defined_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(
      {std::move(date_dim_table), std::move(item_table), std::move(store_sales_table)}),
    custom_relation_generate_tasks,
    std::vector<cudf::data_type>({cudf::data_type(cudf::type_id::FLOAT64),
                                  cudf::data_type(cudf::type_id::INT64),
                                  cudf::data_type(cudf::type_id::INT64),
                                  cudf::data_type(cudf::type_id::STRING)}),
    false);

  // Groupby on d_year, i_brand, i_brand_id
  // After this operation, store_sales_table contains columns
  // ["d_year", "i_brand_id", "i_brand", SUM(ss_ext_sales_price)]
  std::vector<std::unique_ptr<gqe::expression>> groupby_keys;
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(2));
  groupby_keys.push_back(std::make_unique<gqe::column_reference_expression>(3));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> groupby_values;
  groupby_values.push_back(
    std::make_pair(cudf::aggregation::SUM, std::make_unique<gqe::column_reference_expression>(0)));

  store_sales_table = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::move(groupby_keys),
    std::move(groupby_values));

  // Sort on d_year, SUM(ss_ext_sales_price) desc, brand_id
  // After this operation, store_sales_table contains columns
  // ["d_year", "i_brand_id", "i_brand", SUM(ss_ext_sales_price)]
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(3));
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));

  store_sales_table = std::make_shared<gqe::logical::sort_relation>(
    std::move(store_sales_table),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<cudf::order>(
      {cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING}),
    std::vector<cudf::null_order>(
      {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}),
    std::move(sort_exprs));

  // LIMIT 100
  store_sales_table =
    std::make_shared<gqe::logical::fetch_relation>(std::move(store_sales_table), 0, 100);

  auto logical_plan = std::move(store_sales_table);

  gqe::physical_plan_builder plan_builder(&tpcds_catalog);
  auto physical_plan = plan_builder.build(logical_plan.get());

  gqe::optimization_parameters opms{};
  gqe::query_context qctx(&opms);

  gqe::task_graph_builder graph_builder(&qctx, &tpcds_catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::utility::time_function(gqe::execute_task_graph_single_gpu, &qctx, task_graph.get());

  // Output the result to disk
  assert(task_graph->root_tasks.size() == 1);
  auto destination = cudf::io::sink_info("output.parquet");
  auto options     = cudf::io::parquet_writer_options::builder(
    destination, task_graph->root_tasks[0]->result().value());
  cudf::io::write_parquet(options);

  return 0;
}
