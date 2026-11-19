<!--
Before merging, remove:
- Every HTML comment block in this description.
- The entire `## Draft MR submission checklist` section.
- The entire `## Review checklist` section.

For additional guidance, see `docs/agent-guidance/merge-requests.md`.
-->

## Goal and reason

<!--
*What is the high-level goal of this MR, and _why_ does GQE need this change?*
-->

## Problem statement

<!--
*What is the concrete problem this MR solves towards achieving the stated goal?*
-->

## Solution approach

<!--
*What is the solution proposed by this MR?*
-->

## Alternative approaches

<!--
*What alternatives have you considered (if any)?*

We are not looking for straw man arguments here but real alternatives that you
considered, especially if you put work into it only to discover it was a
dead-end or poor approach for some non-obvious reason.
-->

## Impact (feature / bug / performance)

<!--
*What impact does this MR have?*
E.g., the new API features added. An example of the buggy query result fixed.
The measured speedup achieved. Support with profiles, plots, tables, or logs
(if relevant). Changes that make performance claims or may affect performance
should include runtime evidence from
[gqe-bench](https://gitlab-master.nvidia.com/Devtech-Compute/gqe-bench).
-->

## Issues closed

<!--
*The list of GQE issues this MR closes (if any).*
Using "Closes #<issue number>" will close the issue automatically in gitlab.
-->

## Tests

<!--
*How was this MR tested?*
Consider relevant queries and scale factors for the
current phase of GQE development (e.g., TPC-H SF100, SF1k).
-->

## References

<!--
*The list of relevant papers, external links, etc.*
-->

## Draft MR submission checklist

(to be removed before merging)

*Only for draft MRs: The list of points to be completed by author before the submitting MR for review.*

## Review checklist

(to be removed before merging)

 - [ ] MR submitter is the assignee in GitLab sidebar.
 - [ ] Reviewer is different from the assignee and has assigned themselves as
       reviewer in GitLab sidebar.
 - [ ] MR branch is in the `Devtech-Compute/gqe` repository (i.e., not a fork),
       and is prefixed with `$user-`.
   - Exception: Author has not been granted write permissions.
 - [ ] MR title is ["syntactically correct"](https://cbea.ms/git-commit). E.g.,
       "Optimize the aggregate operator for high group cardinality".
   - 50 characters or fewer.
   - Capitalized.
   - Imperative mood.
   - Descriptive of the change.
 - [ ] MR goal, problem, and approach make sense to me (as described above).
 - [ ] MR is self-contained. I.e., doesn't rely on follow-up MRs to be
       minimally functional.
 - [ ] MR solves *one* (sub-)issue. Big issue is split into multiple
       self-contained MRs.
 - [ ] Code style adheres to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) and [libcudf developer guidelines](https://github.com/rapidsai/cudf/blob/main/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md).
   - MLIR code instead adheres to the [LLVM coding standards](https://llvm.org/docs/CodingStandards.html).
 - [ ] Code is DRY. I.e., uses existing GQE / cuDF / MLIR infrastructure and
       utilities.
 - [ ] Code is documented.
   - [Doxygen API comments](https://github.com/rapidsai/cudf/blob/main/cpp/doxygen/developer_guide/DOCUMENTATION.md).
   - Algorithm / data structure description (if relevant).
 - [ ] Code includes CI unit and/or integration tests (if relevant).
 - [ ] MR is tested (see section Tests) and passes the applicable GQE tests.
 - [ ] Stale MR branches are cleaned up. E.g., `this-MR-branch-v1`.
