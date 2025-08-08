# MLIR Query Compiler Tests

## Benchmarking

Compile the test with:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DGQE_ENABLE_QUERY_COMPILER=On ..
make -j mlir_tpch_q6
```

Then run the test with:

```bash
TPCH_DATA_DIR=/scratch/local/sf1_id32 ./test/compiler/mlir_tpch_q6
```

## Debugging

Compile the test with:

```bash
cmake -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DGQE_ENABLE_QUERY_COMPILER=On -DLLVM_ENABLE_ASSERTIONS=ON ..
make -j mlir_tpch_q6
```

There are two options for running the test, with CTest or manually. CTest is intended for the CI, as it configures `TPCH_DATA_DIR`.

To run the test with CTest:

```bash
ctest -R mlir_tpch_q6 -V
```

To run the test manually and print debug info for a selected conversion or transform pass:

```bash
TPCH_DATA_DIR=/scratch/local/sf1_id32 ./test/compiler/mlir_tpch_q6 -debug-only declarative-to-imperative-conversion-pass
```

Passing `-debug` instead of `-debug-only my-pass` prints all MLIR debug info, including the MLIR library's output.
