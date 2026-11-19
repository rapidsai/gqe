# Relational Algebra to Structured Control Flow Dialect Conversion

The pass converts RelAlg to SCF (and a handful of other base dialects). As
RelAlg expresses a declarative query plan, the pass effectively lowers this
declarative IR to an imperative IR.

The pass involves four main tasks:

  - Declarative to imperative conversion of the query plan. In the declarative
    form, the query plan declares dependencies along the data flow edges between
    ops. The root op depends on its parent ops, etc. However, in declarative
    form, the dependencies are represented abstractly, by a tuple stream. The
    conversion instantiates these dependencies by following the data flow
    "bottom up". As the imperative form consists of control flow ops, the ops
    are nested into the bodies of the control flow ops.

  - Materialization of IUs. "IU" stands for "information unit", is defined as "a
    single IU corresponds to a variable that can be bound to a value" by Guido
    Moerkotte in his book ["Building Query
    Compilers"](https://pi3.informatik.uni-mannheim.de/~moer/querycompiler.pdf).
    The declarative IR contains IU references. However, the IUs need to be
    materialized into values by converting them into load (or store) ops on
    MemRefs.

  - Kernel outlining. RelAlg ops can be grouped into operator pipelines, [as
    defined by Neumann](https://dl.acm.org/doi/10.14778/2002938.2002940). Ops
    that materialize their result break a pipeline. In GPU code, this terminates
    a kernel, as pipeline breakers require a global sychronization. In effect, a
    pipeline is converted into a kernel. However, as RelAlg ops can participate
    in multiple pipelines (e.g., a hash join converts to build and probe
    phases), and dependency information is lost after converting to imperative
    form, this pass outlines the kernels.

  - Op conversion. Each op is converted from the source dialect (i.e., RelAlg)
    into the target dialects (declared in `dependentDialects` below). Op
    conversions are documented alongside the respective ops.

Note: In future, if RelAlg is converted to an intermediate declarative form
instead of imperative, kernel outlining could be done in a dedicated MLIR
transform pass.

The first three tasks are described in more detail below.

## Declarative to Imperative Conversion

RelAlg ops are combined to a query plan by producing and consuming a
`relalg.tuplestream`. Therefore, the tuple stream declares that the consuming op
depends on the producing op. The result is a dependency graph, that we restrict
to a DAG.

Each tuple stream has an implicit schema, defined by the result of the tuple
stream producer. The conversion instantiates the tuple stream schema as an IU
tuple. In order to construct each IU tuple before it is accessed, the conversion
pass walks the query plan IR via the [op dominance
graph](https://en.wikipedia.org/wiki/Dominator_(graph_theory)) in post-order.

For each op, we define a conversion by implementing a
`DeclarativeConversionPattern`.

During the IR walk, the pass converts each op with the
`DeclarativeConversionPattern::matchAndRewrite` conversion pattern. The pattern
is expected to set the rewriter's insert position to the current tuple stream.
The op then forwards the tuple stream by saving a new insert position, by
creating a `relalg.forwardts` op (ForwardTupleStreamOp).

Sidenote: The design technically allows an op to create multiple
ForwardTupleStreamOps. It is not yet clear if this is actually needed. Potential
use-cases are binary RelAlg ops (e.g., joins) that would create multiple tuple
streams if exlicitly lowered into their components (e.g., build and probe
phases), and optimizations for tuple stream producers (e.g., initializing and
flushing a memory write buffer outside the producer loop).

The IR walk occurs on an immutable graph without recursion to simplify the
implementation and avoid bugs in op conversions. The pass enforces immutability
by creating an explicit traversal list before performing the rewrites. However,
an exception is required for ForwardTupleStreamOp, which is created by the
rewrites. However, these are only placeholders that need to be erased, without a
substantial rewrite.

During the conversion, each rewrite is expected to erase or replace the op it
visits. The pass keeps dependencies alive during the conversion performing two
phases. In a first phase, the pass performs the rewrites. In a second phase, the
pass relinks dependent ops in the target IR, and then erases the RelAlg IR op.
Therefore, erasing an op in a rewrite only marks it as erased, but does not
erase it. The erasures are applied in the second phase.

The pass is assisted by `DeclarativeConversionPatternRewriter`, which provides a
builder API to the `matchAndRewrite` op conversions. The rewriter facilitates
conversion. It saves and restores insert points of tuple streams. The rewriter
also provices methods for manipulating the tuple stream's IU tuple.

## Materialization of IUs

IU references are dereferenced using a GetIUOp. The information needed to
convert a GetIUOp is:

  - The IU's operand index, which identifies the IU among an op's operands. An
    IU is represented in the IR by an `mlir::Value`.

  - The IU tuple of a tuple stream, which maps the IU index to the IU. The IU
    tuple optionally associates a name with an IU, making it a named IU tuple.

Therefore, a tuple stream producer constructs an IU tuple containing the IUs it
produces. `DeclarativeConversionPatternRewriter` provides the `getIUTuple()`
method to get the IU tuple of the op currently being converted. The op
conversion pattern is expected to append its IUs to its IU tuple. For
convenience, `DeclarativeConversionPatternRewriter` also provides the
`appendIUTuple()` method, that appends the IUs from another IU tuple (e.g., the
parent) to the current op's IU tuple.

`getIUTuple()` is overloaded with a variant that takes a tuple stream argument.
This allows a conversion pattern to lookup IUs on an input tuple stream.

The rewriter conversion pass owns all IU tuples, and retains them in a
`NamedIUTupleCollection`. The collection provides methods that operate on
multiple IU tuples. For example, it implements the underlying
`append(srcTupleStream, dstTupleStream)` used by `appendIUTuple()`. The design
is inspired by `mlir::SymbolTableCollection`.

GetIUOps conversion is tied to the declarative to imperative conversion pass. It
cannot be lowered before the pass, because the IU tuple cannot be constructed
without the IUs. It cannot be lowered after the pass, because the underlying
tuple stream no longer exists.

GetIUOp does not directly take a tuple stream argument. Instead, because a
GetIUOp is always located within a region of a tuple stream consumer op (e.g., a
FilterOp), the conversion looks up the tuple stream of the consumer op.

Tuple stream consumer ops are expected to declare the `TupleStreamConsumer`
trait. The GetIUOp conversion identifies the parent consumer by searching for
this trait.

## Design Choices

  - ForwardTupleStreamOp is an explicit placeholder that is not lowered to SCF.
    Alternatively, the insert point could be stored implicitly using a method
    call. However, the explicit Op enables IR analysis and debugging via
    printing the insert point to screen.

  - Threads are assumed to participate in a cooperative group (e.g., a subwarp
    or warp). For this reason, a thread cannot be assumed to be "active". As a
    result, executing GetIUOp can result in undefined behavior, such as an
    invalid memory access. To handle active/inactive threads,
    ForwardTupleStreamOp contains an `isActive` Boolean flag that indicates the
    current thread's status.

    A thread can become inactive for three reasons: (1) In the scan loop, the
    thread's row index is out-of-bounds. (2) "Filter divergence", i.e., an Op
    (e.g., a filter or join)filters out the thread's row. (3) "Expansion
    divergence", i.e., an Op (e.g., a join) expands one row to multiple rows. In
    case (3), the neighboring threads become inactive.

## Invariants that Op Rewrites must uphold

  - If an Op produces a tuple stream, the Op's corresponding `matchAndRewrite()`
    method must create a ForwardTupleStreamOp.

  - Threads must not exit early, as they are participating in a cooperative
    group.

  - Threads must check the ForwardTupleStreamOp's `isActive` flag before
    executing a `GetIUOp`.

# Kernel Outlining

TODO: Not yet implemented.

Currently, for TPC-H Q6, the kernel is explicitly instantiated by the ScanOp.
Kernel arguments for the query state are appended by the AggregateOp.

Design requirements:

  - Materialize Ops (e.g., CudfTableMaterialize) should be able to declare kernel arguments for
    their output.

  - The kernel launch and GPU memory allocations / frees should be created
    together with the kernel.

  - Outlining should encompass multiple op pipelines by generating multiple
    kernels.

  - A single op should be able to split into multiple kernels. E.g., a hash join
    op should split into the build and probe pipelines. E.g., an aggregate op
    should split into the hash table build, and a hash table scan that consumes
    the hash table and produces a new tuple stream.
