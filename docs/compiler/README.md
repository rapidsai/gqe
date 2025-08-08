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
