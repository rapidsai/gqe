# Substrait Producer

In addition to accepting SQL directly through the [client-server mode](../../README.md#run-gqe), `GQE` can also import a pre-generated logical plan from a [Substrait](https://substrait.io) file.

To produce such plans from SQL offline, `GQE` ships a producer based on [datafusion-substrait](https://github.com/datafusion-contrib/datafusion-substrait), which relies on the `SQL` parser and optimizer from [DataFusion](https://github.com/apache/datafusion). The code is written in Rust and all dependencies managed by Cargo.

## Pre-requisites

### Rust toolchain
Rust is included in the `gqe` conda environment shipped with the Docker image described in [CONTRIBUTING.md](../../CONTRIBUTING.md#setting-up-your-build-environment); no separate install is needed inside the container. If you build outside the container, install Rust via [rustup](https://rustup.rs).

### Data in `parquet` format
The producer registers table schemas by calling `SessionContext::register_parquet()` from the `datafusion` crate, so you need one `parquet` file per table.

## Running DataFusion-Substrait producer from GQE

Extend the base build (see [CONTRIBUTING.md](../../CONTRIBUTING.md#setting-up-your-build-environment)) by reconfiguring CMake with `-DGQE_ENABLE_SUBSTRAIT_PRODUCER=ON`:

```bash
cd /gqe/build
cmake -G Ninja -DGQE_ENABLE_SUBSTRAIT_PRODUCER=ON ..
cmake --build .
```

This installs the `producer` binary into the active `gqe` conda environment.

To generate a `Substrait` plan from a `SQL` query string, first, modify the `rust/producer/config/template.yaml` file.
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

> **_NOTE:_** You can use the config file `rust/producer/config/tpcds_q3.yaml` as an example. You will need to provide your own `parquet` files for `date_dim`, `item` and `store_sales`. Once all information in the brackets are filled, the file should be ready for `cargo run`.

## Executing a Substrait plan

The `tpc` benchmark binary loads a TPC-H or TPC-DS dataset into a pre-registered catalog and executes one or more Substrait plans against it:

```bash
./build/benchmark/tpc <h|ds> <path-to-substrait> <path-to-dataset> [storage-kind]
```

Arguments:

- `<h|ds>` — selects the hardcoded schema: `h` for TPC-H, `ds` for TPC-DS.
- `<path-to-substrait>` — either a single `.bin` plan file or a directory of `.bin` files; each plan in the directory is executed in turn.
- `<path-to-dataset>` — a directory with one subdirectory per table, each holding Parquet files (same layout expected by `scripts/load_tpch.py`).
- `[storage-kind]` — optional, defaults to `parquet_file`. One of `system_memory`, `numa_memory`, `numa_pinned_memory`, `pinned_memory`, `device_memory`, `managed_memory`, `parquet_file`. In-memory storage kinds pre-load the tables once, avoiding Parquet read overhead per query.

Example — run a single TPC-H query plan against a Scale-Factor 1 dataset, keeping tables on the GPU:

```bash
./build/benchmark/tpc h /path/to/tpch/plans/q1.bin /path/to/tpch/sf1 device_memory
```

Each query writes its result to `output_<plan_stem>.parquet` in the current working directory.

## Inspecting Substrait Plan
To decode a `Substrait` plan, `cd` to the directory which contains `plan.proto` file. And run `proto --decode`
```bash
$ cd <path_to_subtrait_def>/substrait/proto
$ protoc --decode substrait.Plan substrait/plan.proto < your_substrait_plan.bin
```
