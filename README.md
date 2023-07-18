# GPU Query Executor (GQE) project

`GQE` is a C++ proof-of-concept SQL query engine for running data analytics queries on the GPUs.

## Substrait Producer
`GQE` uses the producer from [datafusion-substrait](https://github.com/datafusion-contrib/datafusion-substrait). The `datafusion-substrait` project relies on the `SQL` parser and optimizer from [DataFusion](https://github.com/apache/arrow-datafusion). The code is written in Rust and all dependencies managed by Cargo.

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
First, go to the `rust/producer/` directory:
```
$ cd <project_root>/rust/producer/
```

To build the producer, run:
```
$ cargo build
```

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
cargo run <path_to_config_file>
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
| GQE_JOIN_USE_HASH_MAP_CACHE | true | Allow multiple join tasks to reuse the same hash map. Enabling this option may increase device-memory usage in some circumstances. If `MAX_NUM_WORKERS` is set to more than 1, this option is disabled. |

Note that in order to achieve overlapping, libcudf has to be compiled with per-thread default stream, which can be enabled by passing `--ptds` to [`build.sh`](https://github.com/rapidsai/cudf/blob/branch-23.06/CONTRIBUTING.md#build-cudf-from-source).
