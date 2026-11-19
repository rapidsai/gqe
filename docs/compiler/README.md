# GQE Query Compiler

The query compiler converts relational algebra operators encoded as MLIR
operations into executable cubin bytecode. It integrates with GQE on the
physical query plan level by transforming a GQE physical plan into an MLIR query
plan.

The intention is to generate efficient bytecode by using just-in-time
compilation. This allows the compiler to specialize the code of complex SQL
expressions, and fuse relational algegra operators into operator pipelines that
pass data in registers or CUDA shared memory.

Low compile times are a target but not an immediate priority. Observed compile
times are typically below 100 ms.

## Design documents

Please find information on the query compiler's design here:

 - [The declarative to imperative conversion pass (RelAlgToSCF).](RelAlgToSCFDialectConversion.md)

## Building the query compiler

Set `-DGQE_ENABLE_QUERY_COMPILER=ON` when configuring GQE. The compiler needs
LLVM and MLIR, which the dev container installs in `/usr/local` (the version is
set by `LLVM_VERSION` in the `Dockerfile`). The container's LLVM is built
**without assertions**.

GQE takes its LLVM compile definitions from the LLVM installation it finds, so
it follows that installation's assertion setting: with an assertions build, all
of GQE is compiled with `-UNDEBUG`, `-D_DEBUG`, and `-D_GLIBCXX_ASSERTIONS`.
Do not change those flags in GQE to switch between the two.

### Building with assertions locally

Assertions in LLVM and in the compiler code catch many mistakes with a clear
message instead of a crash, so use them while developing compiler code. Build
a second LLVM into a prefix of your own and use a separate GQE build directory
for it. Use the same version as the `Dockerfile` (`LLVM_VERSION`) and the same
options:

```bash
source /conda/bin/activate gqe
LLVM_VERSION=$(sed -n 's/^ARG LLVM_VERSION=//p' Dockerfile)
git clone --branch llvmorg-${LLVM_VERSION} --depth 1 \
    https://github.com/llvm/llvm-project.git ~/src/llvm-project
cmake -S ~/src/llvm-project/llvm -B ~/src/llvm-project/build-assert -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/llvm-assert \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_CUDA_RUNNER=ON
cmake --build ~/src/llvm-project/build-assert --target install
```

Then point GQE at it from a separate build directory:

```bash
cmake -S . -B build-assert -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DGQE_ENABLE_QUERY_COMPILER=ON \
    -DMLIR_DIR=$HOME/llvm-assert/lib/cmake/mlir \
    -DLLVM_DIR=$HOME/llvm-assert/lib/cmake/llvm
cmake --build build-assert
```

The build takes a while; use all cores. If `/tmp` is small, set `TMPDIR` (and
`CCACHE_DIR`) to a directory with more space.

Without assertions, code that is only used by an `assert` triggers
`-Wunused-variable` errors and must be marked `[[maybe_unused]]`. Never put
code with side effects inside an `assert`: it does not run in the container's
build.
