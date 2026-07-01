# Customized Parquet Reader

GQE ships a prototype Parquet reader that is limited in functionality but more performant than the default cuDF reader. It is disabled by default; the sections below cover how to build it and the environment variables that tune its behavior.

## Building

Add `-DGQE_ENABLE_CUSTOMIZED_PARQUET=ON` to the CMake configure step of the base build (see [CONTRIBUTING.md](../../CONTRIBUTING.md#setting-up-your-build-environment)):

```bash
cd /gqe/build
cmake -G Ninja -DGQE_ENABLE_CUSTOMIZED_PARQUET=ON ..
cmake --build .
```

## Environment variables

Setting `GQE_USE_CUSTOMIZED_IO=true` at runtime routes reads through the customized reader; all other `GQE_IO_*` variables tune it.

| Variable | Default | Description |
|---|---|---|
| GQE_USE_CUSTOMIZED_IO | false | Whether to use the customized Parquet reader if supported. |
| GQE_IO_BOUNCE_BUFFER_SIZE | 4 | Size in GB per worker of the page-locked CPU memory bounce buffer used for the customized Parquet reader. |
| GQE_IO_AUXILIARY_THREADS | 8 | Number of auxiliary threads per worker launched by the customized Parquet reader. |
| GQE_IO_BLOCK_SIZE | 2048 | Size in KiB of the block used by the customized Parquet reader when io_uring is enabled. |
| GQE_IO_ENGINE | IO_URING | The I/O engine used by the customized Parquet reader. Supported values are `IO_URING`, `PSYNC` and `AUTO`. |
| GQE_IO_PIPELINING_ENABLE | true | Enable pipelining disk read and H2D when using customized Parquet reader. |
| GQE_IO_ALIGNMENT | 4096 | The alignment of the memory buffer used by the customized Parquet reader. |
