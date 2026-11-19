# C++ and CUDA guidance

Read [`REVIEW_GUIDELINES_CPP.md`](../../REVIEW_GUIDELINES_CPP.md) before writing
C++ or CUDA code. It defines the correctness, resource-management, stream,
performance, and test review bar. Formatting is enforced by the configured
pre-commit hooks.

## Documentation

Use `include/gqe/qep/task.hpp` and `include/gqe/qep/algorithms.hpp` as
documentation-style exemplars.

- Write Doxygen prose in full sentences. Use short paragraphs of two to four
  sentences and bullets for enumerations. Do not join sentences with semicolons
  or colons. Separate documented struct fields with blank lines.
- Put large contracts in class- or file-level comments. Organize them with
  Markdown headings such as `# Design`, `## Thread Safety`, and `# Use Case
  Examples`. Define important terms before using them. Keep per-method
  documentation to a short `@brief` and parameter descriptions when the
  enclosing contract already states the semantics.
- Explain why a design was chosen, including relevant rejected alternatives.
  State contracts as component responsibilities. Keep header contracts
  self-contained; external links may supplement but must not replace them.
- Document algorithm complexity and define every variable in the bound. Include
  pathological bounds and explain failure behavior. State how each `@pre`
  condition is enforced, and give assertions message strings.
- Document every template type parameter. Include callable signatures and the
  reason for constraints. Add a short comment to each clause of a `requires`
  expression.
- Group related free functions under `// ----- Section -----` separators. Put
  identifiers in backticks in prose.
- Document repeated parameters such as `ctx_ref`, `stream`, and `mr` with
  consistent wording. Put defaulted `stream` and `mr` parameters last.
- Comment on constraints that the code cannot express. Do not add comments about
  code provenance or review history.

## GQE conventions

- GQE's memory-resource contract differs from the cuDF convention. The passed
  `mr` controls all allocations made by an API, including temporary allocations.
  Do not use `cudf::get_current_device_resource_ref()` for scratch storage. If
  persisted output and compute scratch require different resources, expose both
  explicitly using the `storage_mr` and `compute_mr` pattern. This rule tracks
  issue #416.
- When a default CUDA stream is permitted, use `cudf::get_default_stream()`.
  Do not use the RMM default stream.
- Use American English in code, comments, documentation, and merge request text.
- Name mutex guard variables `guard`, not `lock`. Boolean names must read as
  predicates, such as `is_negated`. Implementations share the interface noun,
  such as `dictionary_interner` and `dictionary_column_interner`.
- Defaults belong in `gqe::optimization_parameters`. Plumbing structures such as
  `compression::configuration` contain bare fields, and their field order stays
  consistent with the parameters header.
- Pass `cudf::column_view` by value.
