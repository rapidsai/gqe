# Engine facts

## In-memory tables

- In-memory `COPY` stages the table uncompressed in shared memory before write
  tasks compress it. The loadable table size is therefore limited by the
  uncompressed size and `/dev/shm`, not the compressed size.
- For large inputs, load per-query-group projections. `COPY` projects Parquet
  columns onto a narrower table schema by name, and `COPY FROM` accepts a
  directory of symlinks.
- Flight SQL tables must use `shared_segment_memory` storage because
  `local_memory_descriptor` cannot be serialized for multi-process execution.
  Containers commonly limit `/dev/shm` to 2 GiB, so large loads may require a
  larger host mount.

## Logging and diagnostics

- `GQE_LOG_LEVEL=debug` enables debug logs. Task-manager debug logs are forwarded
  to the node manager's standard output. Pruning reports per-cause counts at
  debug level.
- Capture memory footprint from the live server's `_gqe_column_stats` and
  `_gqe_table_stats` system tables. Use `gqe-cli --parquet` when a snapshot is
  needed.

## Known limitations

- Compressed output materialization does not support nullable columns. It
  constructs columns without null masks and hard-codes the null count to zero,
  and therefore throws when the zone map reports nulls in the selected
  partitions. This behavior is implemented by
  `compressed_sliced_output_column_builder::make_cudf_column()` and
  `string_compressed_sliced_output_column_builder::make_cudf_column()`. The
  uncompressed path in `contiguous_output_column_builder::make_cudf_column()`
  reconstructs the mask from the zone map and does carry nulls.
- `output_table_builder::build()` generally selects the first row group to
  determine each output column's storage layout, so corresponding columns must
  use compatible layouts across row groups. Compressed-sliced string columns
  are the exception: the builder examines every selected row group, chooses
  64-bit output offsets when any source uses them or the combined character
  data requires them, and widens 32-bit source offsets during materialization.
