# GPU Query Executor (GQE) project

`GQE` is a C++ proof-of-concept SQL query engine for running data analytics queries on GPU.

## Build GQE

Please see the [contributing guide section](CONTRIBUTING.md#setting-up-your-build-environment) on setting up the build environment and building GQE.

GQE has the following CMake flags to control what to build.
- GQE implements a prototype Parquet reader that is limited in functionality but more performant. By default, this reader will not be built and GQE will rely on cuDF's Parquet reader. Specify `-DGQE_ENABLE_CUSTOMIZED_PARQUET=ON` to build the prototype Parquet reader.

## Generate Query Plan

`GQE` does not have a SQL parser. Instead, it can import a logical plan from a [Substrait](https://substrait.io) file.

To generate a Substrait query plan, `GQE` uses the producer from [datafusion-substrait](https://github.com/datafusion-contrib/datafusion-substrait). The `datafusion-substrait` project relies on the `SQL` parser and optimizer from [DataFusion](https://github.com/apache/datafusion). The code is written in Rust and all dependencies managed by Cargo.

### Pre-requisites
#### Install Rust
Before running the producer, please install [Rust](https://www.rust-lang.org/tools/install).

For Linux and macOS, run:
```
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

For Windows, please visit this [installation guide](https://forge.rust-lang.org/infra/other-installation-methods.html).

#### Data in `parquet` format
The producer register the schema by leveraging the `SessionContext::register_parquet()` function from the `datafusion` crate.

Please make sure you have a data file per table in `parquet` format for schema registration.


### Running DataFusion-Substrait producer from GQE

Build this as part of GQE by specifying `GQE_ENABLE_SUBSTRAIT_PRODUCER`

```
conda activate gqe
mkdir build
cd build
cmake -G Ninja .. -DGQE_ENABLE_SUBSTRAIT_PRODUCER=ON
ninja
```

This will install the producer in the active conda environment

To generate a `Substrait` plan from a `SQL` query string, first, modify the `config/template.yaml` file.
The config file should contain the following sections:
```
sql:
output:
tables:
```
The `sql` section specifies the `SQL` string to be parsed and translated into a `Substrait` plan.

The `output` section specifies the path where the output `Substrait` binary plan should be stored.

The `tables` section should contain the list of tables to register and their data location.
The information for each table should be stored in a subsection of the `tables` section. The name of each subsection should correspond to the table name. Fill each `table` subsection with either `directory` OR `file` information. An exception will be thrown if both are defined.

To generate the `Substrait` plan file, run:
```
producer <path_to_config_file>
```
The `.bin` file we just generated can be translated into `GQE` logical plan using the `GQE` consumer.

> **_NOTE:_** You can use the config file `tpcds_q3.yaml` as an example. You will need to provide your own `parquet` files for `date_dim`, `item` and `store_sales`. Once all information in the brackets are filled, the file should be ready for `cargo run`.

### Inspecting Substrait Plan
To decode a `Substrait` plan, `cd` to the directory which contains `plan.proto` file. And run `proto --decode`
```bash
$ cd <path_to_subtrait_def>/substrait/proto
$ protoc --decode substrait.Plan substrait/plan.proto < your_substrait_plan.bin
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| MAX_NUM_WORKERS | 1 | Max number of worker threads per stage |
| MAX_NUM_PARTITIONS | 8 | The maximum number of read tasks that can be generated for a single table |
| GQE_LOG_LEVEL | info | Enable log messages for this level or higher |
| GQE_LOG_FILE | info | Enable log messages written to a file path, the value should be a valid file path |
| GQE_INITIAL_QUERY_MEMORY | 10GB | Initial memory pool size for queries (in bytes). This memory is used only for a query, and is not shared between queries. |
| GQE_MAX_QUERY_MEMORY | 90% of available GPU memory | Maximum memory pool size for queries (in bytes). |
| GQE_INITIAL_TASK_MANAGER_MEMORY | 10GB | Initial memory pool size for task manager memory resources (in bytes). This memory is used across queries, e.g., for in-memory tables. |
| GQE_MAX_TASK_MANAGER_MEMORY | unlimited | Maximum memory pool size for task manager memory resources (in bytes). |
| GQE_JOIN_USE_HASH_MAP_CACHE | false | Allow multiple join tasks to reuse the same hash map. Enabling this option may increase device-memory usage in some circumstances. |
| GQE_JOIN_USE_PERFECT_HASH | true | Enable perfect hashing for join operations. |
| GQE_JOIN_USE_MARK_JOIN | true | Uses the mark join implementation for left semi and anti joins. Disable to fall back to cuDF. |
| GQE_READ_USE_ZERO_COPY | true | Enable zero-copy reads for in-memory table. When disabled, read tasks copy input data to a temporary output buffer. |
| GQE_USE_CUSTOMIZED_IO | false | Whether to use the customized Parquet reader if supported. |
| GQE_IO_BOUNCE_BUFFER_SIZE | 4 | Size in GB per worker of the page-locked CPU memory bounce buffer used for the customized Parquet reader. |
| GQE_IO_AUXILIARY_THREADS | 8 | Number of auxiliary threads per worker launched by the customized Parquet reader. |
| GQE_USE_OPT_TYPE_FOR_SINGLE_CHAR_COL | true | Use optimized char type instead of string type to store single-char columns (currently only for TPC-H dataset) |
| GQE_IN_MEMORY_TABLE_COMP_FORMAT | none | Compression format for the in-memory table. Supported values are :`none`,`ans`,`lz4`,`snappy`,`cascaded`,`gdeflate`,`deflate`,`zstd`,`gzip`,`bitcomp`,`best_compression_ratio`,`best_decompression_speed`. `best_compression_ratio`, and `best_decompression_speed` autoselects the compression algorithm based on the column type to yield the best compression ratio or decompression speed respectively. |
| GQE_COMPRESSION_CHUNK_SIZE | 16 | n in 2^n -- Size in Bytes used for chunking of data for compression. |
| GQE_IO_BLOCK_SIZE | 2048 | Size in KiB of the block used by the customized Parquet reader when io_uring is enabled. |
| GQE_IO_ENGINE | IO_URING | The I/O engine used by the customized Parquet reader. Supported values are `IO_URING`, `PSYNC` and `AUTO`. |
| GQE_IO_PIPELINING_ENABLE | true | Enable pipelining disk read and H2D when using customized Parquet reader. |
| GQE_IO_ALIGNMENT | 4096 | The alignment of the memory buffer used by the customized Parquet reader. |
| GQE_USE_OVERLAP_MTX | true | Improve overlap and pipelining by using locks in memory read task. Keeping this option enabled when running with compressed columns is not optimal. |
| GQE_USE_PARTITION_PRUNING | false | Enable partition pruning. |
| GQE_ZONE_MAP_PARTITION_SIZE | 100'000 | Number of rows per zone map row group. Setting this to 0 disables the creation of zone maps and therefore partition pruning, even if `GQE_USE_PARTITION_PRUNING` is `true`. |
| GQE_AGGREGATE_USE_PERFECT_HASH | true | Enable perfect hashing for groupby operations. |
| GQE_COMPRESSION_RATIO_THRESHOLD | 1.0 | Compression ratio threshold to decide whether to compress the columns. Below this threshold, columns are stored uncompressed. Higher ratio means better compression. |

Note that in order to achieve overlapping, libcudf has to be compiled with per-thread default stream, which can be enabled by passing `--ptds` to [`build.sh`](https://github.com/rapidsai/cudf/blob/branch-25.10/CONTRIBUTING.md#build-cudf-from-source).

### CUDA Device and NUMA Affinity

The CUDA device to execute on can be selected by setting [the `CUDA_VISIBLE_DEVICES` environment variable](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-visible-devices). This variable remaps the device ordinals. GQE selects device 0.

The NUMA affinity of a device depends on the system topology. When memory kind is specified as `numa` or `numa_pinned`, GQE allocates tables on the NUMA node affine to the device used for table registration. In multi-process execution, NUMA node selection occurs per process.

### Partition pruning

If a table is split into multiple Parquet files, the lexicographical sort order of the file names has to correspond to the sort oder of the rows in the table, for pruning to be effective. This means that numbers in the file names have to be padded with leading zeros.

Good: `lineitem01.parquet`, ... , `lineitem09.parquet`, `lineitem10.parquet`

Bad: `lineitem1.parquet`, ... , `lineitem9.parquet`, `lineitem10.parquet`
