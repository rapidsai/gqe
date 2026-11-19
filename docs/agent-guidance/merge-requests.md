# Merge request guidance

Follow the repository's
[GitLab merge request template](../../.gitlab/merge_request_templates/default.md)
and the signing requirements in [`CONTRIBUTING.md`](../../CONTRIBUTING.md#signing-your-work).

- Keep all substantive template sections, including those that remain empty.
  Treat HTML comments as drafting guidance. Leave the review checklist unchecked
  when preparing the description for review. Before merging, remove every HTML
  comment block and the entire Draft MR submission checklist and Review checklist
  sections.
- The MR submitter is the assignee in the GitLab sidebar. The reviewer must be
  a different person and must assign themselves as reviewer; the assignee should
  not assign the reviewer.
- Put the MR branch in the main repository and prefix it with the author's
  username. Authors without write permission may use a fork.
- For stacked MRs, use one branch per stack level and target each MR at its
  predecessor so that each diff contains only that level's changes.
- Keep the MR self-contained and focused on one issue or sub-issue.
- In Alternative approaches, describe real alternatives considered, including
  abandoned work and why it failed or proved unsuitable.
- In the Tests section, record the commands, environment, branch, and commit
  hashes used for testing. Use the relevant query and scale-factor details.
- When the change makes a performance claim or may affect performance, include
  benchmark runtime evidence from
  [gqe-bench](https://gitlab-master.nvidia.com/Devtech-Compute/gqe-bench).
- Use tables for measured results when they make comparisons clearer.
- Use ASCII in text intended for GitLab copy-paste. In particular, use `x`,
  `->`, and `-` instead of typographic equivalents that terminal clipboard paths
  may corrupt.
