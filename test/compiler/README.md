# MLIR Query Compiler Tests

To build the query compiler and these tests, see
[Building the query compiler](../../docs/compiler/README.md#building-the-query-compiler).

## Benchmarking

Build the test in a Release build directory with the query compiler on:

```bash
cmake --build build --target mlir_tpch_q6
```

Then run the test with:

```bash
TPCH_DATA_DIR=/scratch/local/sf1_id32 ./build/test/compiler/mlir_tpch_q6
```

## Debugging

For debugging, use a GQE build directory configured against an LLVM built with
assertions, as described in
[Building with assertions locally](../../docs/compiler/README.md#building-with-assertions-locally).
The dev container's LLVM is built without assertions, which also compiles out
the debug output used below.

There are two options for running the test, with CTest or manually. CTest is intended for the CI, as it configures `TPCH_DATA_DIR`.

To run the test with CTest:

```bash
ctest -R mlir_tpch_q6 -V
```

To run the test manually and print debug info for a selected conversion or transform pass:

```bash
TPCH_DATA_DIR=/scratch/local/sf1_id32 ./build-assert/test/compiler/mlir_tpch_q6 -debug-only declarative-to-imperative-conversion-pass
```

Passing `-debug` instead of `-debug-only my-pass` prints all MLIR debug info, including the MLIR library's output.
