/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gqe/qep/pipeline_normalization.hpp>

#include <gqe/qep/algorithms.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/intermediate_state_read.hpp>
#include <gqe/qep/operator/intermediate_state_write.hpp>
#include <gqe/utility/error.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

/** @brief Semantic QEP task families used to select pipeline-boundary normalization rules. */
enum class task_family { fold, iterate, stateless_transform, stateful_transform };

/** @brief Classify a QEP task through its public visitor interface. */
class task_family_visitor final : public qep_visitor {
 public:
  void visit(fold_task const&) override { family = task_family::fold; }
  void visit(iterate_task const&) override { family = task_family::iterate; }
  void visit(stateless_transform_task const&) override
  {
    family = task_family::stateless_transform;
  }
  void visit(stateful_transform_task const&) override { family = task_family::stateful_transform; }

  std::optional<task_family> family;
};

/** @brief Return the semantic family of `value`. */
[[nodiscard]] task_family classify(task const& value)
{
  task_family_visitor visitor;
  value.accept(visitor);
  GQE_EXPECTS(visitor.family.has_value(),
              "pipeline_normalization: task visitor did not classify task",
              std::logic_error);
  return *visitor.family;
}

/** @brief Results from `derive_ordering` */
struct derived_order_result {
  std::unordered_map<task const*, bool> ordered;
  std::vector<task const*> order;
};

/**
 * @brief Derive complete-result ordering from task semantics in dependency order.
 *
 * A preserving task inherits order only from one relational input. Multiple relational inputs
 * have no shared sequence contract, so combining them on a preserving ordered path is rejected.
 * Materialized inputs to normalized stateful transforms are task-private initialization state.
 * In a source plan, physical-relation provenance distinguishes these from external relations.
 *
 * Source intermediate readers are analyzed as preserving tasks so normalization can upgrade
 * bounded readers of ordered input before cloning them. Other tasks keep their intrinsic policy.
 * Analysis takes expected `O(V + E)` time and `O(V + E)` auxiliary space for `V` tasks and `E`
 * edges.
 *
 * @return Per-task complete-result ordering and the plan's topological task order.
 */
[[nodiscard]] derived_order_result derive_ordering(
  query_execution_plan const& plan, task_physical_relations const* physical_relations = nullptr)
{
  auto const tasks = plan.tasks();
  auto order       = sort_topologically<task const*>(
    std::span<task const* const>(tasks),
    [&](task const* current) { return plan.predecessors(current); },
    [&](task const* current) { return plan.successors(current); });
  GQE_EXPECTS(
    order.has_value(), "pipeline_normalization: source QEP contains a cycle", std::logic_error);

  std::unordered_map<task const*, bool> ordered;
  ordered.reserve(tasks.size());
  for (auto const* current : *order) {
    auto const policy             = physical_relations != nullptr &&
                            dynamic_cast<intermediate_state_read_task const*>(current) != nullptr
                                      ? task_ordering::preserve
                                      : current->ordering_policy();
    bool is_ordered               = policy == task_ordering::establish;
    std::size_t relational_inputs = 0;
    // Determine whether to preserve an ordered input.
    if (policy == task_ordering::preserve) {
      for (auto const* predecessor : plan.predecessors(current)) {
        if (classify(*current) == task_family::stateful_transform &&
            predecessor->is_pipeline_breaker()) {
          if (physical_relations == nullptr) { continue; }
          auto const predecessor_relation = physical_relations->find(predecessor);
          auto const current_relation     = physical_relations->find(current);
          GQE_EXPECTS(predecessor_relation != physical_relations->end() &&
                        predecessor_relation->second != nullptr &&
                        current_relation != physical_relations->end() &&
                        current_relation->second != nullptr,
                      "pipeline_normalization: missing physical relation for "
                      "pipeline-breaker input to stateful transform",
                      std::logic_error);
          if (predecessor_relation->second == current_relation->second) { continue; }
        }
        ++relational_inputs;
        is_ordered = is_ordered || ordered.at(predecessor);
      }
      GQE_EXPECTS(!is_ordered || relational_inputs == 1,
                  "pipeline_normalization: preserving ordered output with multiple relational "
                  "inputs is currently unsupported and requires an explicit row-alignment contract",
                  std::logic_error);
    }
    ordered.emplace(current, is_ordered);
  }
  return {std::move(ordered), std::move(*order)};
}

/** @brief Create a reader whose chunking respects the producer's complete-result ordering. */
[[nodiscard]] std::unique_ptr<intermediate_state_read_task> make_reader(
  bool is_ordered, std::optional<cudf::size_type> morsel_max_rows)
{
  if (!is_ordered && morsel_max_rows.has_value()) {
    return std::make_unique<intermediate_state_read_task>(*morsel_max_rows);
  }
  return std::make_unique<intermediate_state_read_task>(
    intermediate_state_read_task::single_output);
}

/**
 * @brief Compute executor-neutral graph metadata for a normalized QEP.
 *
 * The analysis consists of:
 *
 * - Task topological sorting in `O(V + E)` time.
 * - Pipeline partitioning in `O(K * (V + E) * α(V))` time.
 * - Pipeline topological sorting in `O(P + E_p)` time.
 *
 * Where:
 *
 * - `V` is the number of tasks.
 * - `E` is the number of task edges.
 * - `K` is the number of pipeline breakers.
 * - `P` is the number of pipelines.
 * - `E_p` is the number of pipeline edges.
 * - `α` is the inverse Ackermann function from union-find.
 *
 * Pipeline partitioning dominates the worst case because its validation walks each breaker's
 * descendants. The returned metadata occupies `O(V + P + E_p)` space. Peak auxiliary working
 * storage is `O(V + E)`, excluding `plan`, because a descendant traversal may queue one entry per
 * edge. See `partition_into_pipelines` for details.
 */
[[nodiscard]] qep_metadata analyze(query_execution_plan const& plan,
                                   task_stable_keys const& stable_keys)
{
  auto const tasks = plan.tasks();
  auto task_order  = sort_topologically<task const*>(
    std::span<task const* const>(tasks),
    [&](task const* current) { return plan.predecessors(current); },
    [&](task const* current) { return plan.successors(current); });
  GQE_EXPECTS(task_order.has_value(),
              "pipeline_normalization: normalized QEP contains a cycle",
              std::logic_error);

  auto pipelines        = partition_into_pipelines(plan, stable_keys);
  auto pipeline_id_view = std::views::iota(std::size_t{0}, pipelines.num_pipelines) |
                          std::views::transform([](std::size_t i) { return pipeline_id(i); });
  std::vector<pipeline_id> pipeline_ids(pipeline_id_view.begin(), pipeline_id_view.end());
  auto pipeline_order = sort_topologically<pipeline_id>(
    std::span<pipeline_id const>(pipeline_ids),
    [&](pipeline_id id) -> auto const& { return pipelines.predecessors.at(id); },
    [&](pipeline_id id) -> auto const& { return pipelines.successors.at(id); });
  GQE_EXPECTS(pipeline_order.has_value(),
              "pipeline_normalization: pipeline graph contains a cycle",
              std::logic_error);

  return qep_metadata{.topological_order          = std::move(*task_order),
                      .pipelines                  = std::move(pipelines),
                      .pipeline_topological_order = std::move(*pipeline_order)};
}

/**
 * @brief Insert materialization boundaries to prevent pipeline dependency cycles.
 *
 * Start with one component per task, where each component represents tasks that can belong to
 * the same pipeline. First merge streaming edges whose successor or any of its descendants has
 * multiple streaming inputs. Inserting materialization boundaries on these edges could change
 * partition boundaries and row order, breaking row correspondence between those inputs. These
 * connections must therefore remain within one pipeline. Reject required components that
 * contain a breaker edge or form a cycle.
 *
 * Then consider remaining streaming edges in task topological order. The components at an
 * edge's endpoints can be merged only if both conditions hold:
 *
 * - No alternative directed path connects them through another component. Merging the endpoints
 *   of such a path would create a cycle in the graph of dependencies between components.
 * - No pipeline-breaker edge or streaming edge already selected for materialization connects
 *   them directly. Merging would remove that required pipeline boundary.
 *
 * If either condition fails, select the streaming edge for an intermediate writer and reader
 * instead of merging its components. Dependencies between components remain acyclic throughout.
 * Diamonds without a pipeline-breaker dependency remain pipelined.
 *
 * Selected edges from the same producer reuse one writer, but each receives a separate reader.
 * Sharing a reader could reconnect pipelines that must execute separately. Predecessor order
 * and the distinction between materialized and pipelined task inputs are preserved.
 *
 * Before inserting boundaries, reject any selected edge whose successor or descendants have
 * multiple pipelined predecessors. Independent writers can concatenate partitions in different
 * orders, so their outputs cannot guarantee matching rows for tasks that combine columns from
 * multiple pipelined inputs. The conservative traversal includes descendants across pipeline
 * breakers because task families do not establish whether those tasks require row correspondence.
 *
 * For `V` tasks and `E` edges in `source`, analysis takes expected `O(E * (V + E))` time and
 * `O(V + E)` auxiliary space. Rebuilding adds `O(E)` tasks in the worst case. Missing output
 * column types cause an error only when an inserted writer needs them. A topological sort
 * rejects cyclic input before grouping tasks into components.
 */
[[nodiscard]] query_execution_plan repair_bypasses(query_execution_plan source,
                                                   task_output_data_types const& output_types,
                                                   task_stable_keys& stable_keys,
                                                   std::optional<cudf::size_type> morsel_max_rows)
{
  auto const [ordered, order] = derive_ordering(source);
  auto const tasks            = source.tasks();

  std::unordered_map<task const*, task const*> components;
  std::unordered_map<task const*, std::unordered_set<task const*>> cuts;
  for (auto const* current : tasks) {
    components.emplace(current, current);
  }
  auto const merge_components = [&](task const* from, task const* to) {
    for (auto& [current, component] : components) {
      if (component == to) { component = from; }
    }
  };

  // Propagate the replay restriction across breakers as well as streaming edges.
  std::unordered_set<task const*> requires_streaming;
  for (auto const* current : std::views::reverse(order)) {
    auto const predecessors     = source.predecessors(current);
    auto const streaming_inputs = std::ranges::count_if(
      predecessors, [](task const* predecessor) { return !predecessor->is_pipeline_breaker(); });
    if (streaming_inputs > 1 ||
        std::ranges::any_of(source.successors(current), [&](task const* successor) {
          return requires_streaming.contains(successor);
        })) {
      requires_streaming.insert(current);
    }
  }
  for (auto const* successor : order) {
    if (!requires_streaming.contains(successor)) { continue; }
    for (auto const* predecessor : source.predecessors(successor)) {
      if (!predecessor->is_pipeline_breaker()) {
        merge_components(components.at(predecessor), components.at(successor));
      }
    }
  }

  // If mandatory streaming connections already conflict with dependencies, no safe cut exists.
  std::vector<task const*> component_ids;
  std::unordered_map<task const*, std::vector<task const*>> component_predecessors;
  std::unordered_map<task const*, std::vector<task const*>> component_successors;
  for (auto const* current : tasks) {
    auto const* from = components.at(current);
    if (from == current) { component_ids.push_back(from); }
    for (auto const* successor : source.successors(current)) {
      auto const* to = components.at(successor);
      GQE_EXPECTS(from != to || !current->is_pipeline_breaker(),
                  "pipeline_normalization: required streaming component crosses a pipeline breaker",
                  std::logic_error);
      if (from != to) {
        component_successors[from].push_back(to);
        component_predecessors[to].push_back(from);
      }
    }
  }
  auto const component_order = sort_topologically<task const*>(
    std::span<task const* const>(component_ids),
    [&](task const* current) -> auto const& { return component_predecessors[current]; },
    [&](task const* current) -> auto const& { return component_successors[current]; });
  GQE_EXPECTS(component_order.has_value(),
              "pipeline_normalization: required streaming components form a dependency cycle",
              std::logic_error);

  for (auto const* successor : order) {
    for (auto const* predecessor : source.predecessors(successor)) {
      if (predecessor->is_pipeline_breaker()) { continue; }
      auto const* from = components.at(predecessor);
      auto const* to   = components.at(successor);
      if (from == to) { continue; }

      bool has_fence = false;
      std::unordered_map<task const*, std::vector<task const*>> edges;
      for (auto const* current : tasks) {
        for (auto const* next : source.successors(current)) {
          auto const* a = components.at(current);
          auto const* b = components.at(next);
          if (a != b) { edges[a].push_back(b); }
          auto const prior_cuts = cuts.find(next);
          bool const is_cut     = prior_cuts != cuts.end() && prior_cuts->second.contains(current);
          if (a == from && b == to && (current->is_pipeline_breaker() || is_cut)) {
            has_fence = true;
          }
        }
      }
      std::vector<task const*> pending;
      for (auto const* next : edges[from]) {
        if (next != to) { pending.push_back(next); }
      }
      std::unordered_set<task const*> visited;
      bool has_alternate_path = false;
      while (!pending.empty()) {
        auto const* current = pending.back();
        pending.pop_back();
        if (current == to) {
          has_alternate_path = true;
          break;
        }
        if (!visited.insert(current).second) { continue; }
        for (auto const* next : edges[current]) {
          pending.push_back(next);
        }
      }
      if (has_fence || has_alternate_path) {
        cuts[successor].insert(predecessor);
      } else {
        merge_components(from, to);
      }
    }
  }
  if (cuts.empty()) { return source; }

  // Replay does not preserve partition correspondence with other streams. Check every
  // descendant, not only cut endpoints, before constructing the repaired graph.
  // Conservatively continue across breakers because task families do not establish whether
  // a descendant relies on the original row correspondence.
  std::vector<task const*> pending;
  for (auto const& [successor, predecessors] : cuts) {
    pending.push_back(successor);
  }
  std::unordered_set<task const*> visited;
  while (!pending.empty()) {
    auto const* current = pending.back();
    pending.pop_back();
    if (!visited.insert(current).second) { continue; }
    auto const predecessors     = source.predecessors(current);
    auto const streaming_inputs = std::ranges::count_if(
      predecessors, [](task const* predecessor) { return !predecessor->is_pipeline_breaker(); });
    GQE_EXPECTS(streaming_inputs <= 1,
                "pipeline_normalization: bypass replay reaching multiple streaming inputs "
                "is unsupported because partition and row correspondence cannot be preserved",
                std::logic_error);
    for (auto const* successor : source.successors(current)) {
      pending.push_back(successor);
    }
  }

  query_execution_plan_builder builder;
  std::unordered_map<task const*, task const*> clones;
  task_stable_keys repaired_stable_keys;
  repaired_stable_keys.reserve(stable_keys.size());
  std::unordered_map<task const*, task const*> writers;
  for (auto const* current : tasks) {
    auto clone = current->clone();
    if (auto const key = stable_keys.find(current); key != stable_keys.end()) {
      repaired_stable_keys.emplace(clone.get(), key->second);
    }
    clones.emplace(current, clone.get());
    builder.add_task(std::move(clone));
  }
  for (auto const* successor : order) {
    for (auto const* predecessor : source.predecessors(successor)) {
      auto const cut = cuts.find(successor);
      if (cut == cuts.end() || !cut->second.contains(predecessor)) {
        builder.add_successor(clones.at(predecessor), clones.at(successor));
        continue;
      }
      if (!writers.contains(predecessor)) {
        auto const types = output_types.find(predecessor);
        GQE_EXPECTS(types != output_types.end(),
                    "pipeline_normalization: missing output data types for bypass writer",
                    std::logic_error);
        auto const* input = clones.at(predecessor);
        if (dynamic_cast<gather_task const*>(predecessor) == nullptr) {
          auto gather            = std::make_unique<gather_task>();
          auto const* gather_ptr = gather.get();
          builder.add_task(std::move(gather));
          builder.add_successor(input, gather_ptr);
          input = gather_ptr;
        }
        auto writer            = std::make_unique<intermediate_state_write_task>(types->second);
        auto const* writer_ptr = writer.get();
        builder.add_task(std::move(writer));
        builder.add_successor(input, writer_ptr);
        writers.emplace(predecessor, writer_ptr);
      }
      auto reader            = make_reader(ordered.at(predecessor), morsel_max_rows);
      auto const* reader_ptr = reader.get();
      builder.add_task(std::move(reader));
      builder.add_successor(writers.at(predecessor), reader_ptr);
      builder.add_successor(reader_ptr, clones.at(successor));
    }
  }
  stable_keys = std::move(repaired_stable_keys);
  return builder.build();
}

/**
 * @brief Rebuild a QEP with complete pipeline boundaries and compute metadata for the result.
 *
 * Local boundary rewriting takes expected `O(V + E)` time and auxiliary space.
 * Bypass repair takes expected `O(E * (V + E))` time, as documented by `repair_bypasses`.
 * The subsequent analysis has the complexity documented by `analyze`.
 */
class normalizer {
 public:
  normalizer(query_execution_plan const* source,
             task_output_data_types const* output_data_types,
             task_physical_relations const* physical_relations,
             task_stable_keys const* stable_keys,
             std::optional<cudf::size_type> morsel_max_rows)
    : _source(source),
      _output_data_types(output_data_types),
      _physical_relations(physical_relations),
      _stable_keys(stable_keys),
      _morsel_max_rows(morsel_max_rows)
  {
    GQE_EXPECTS(!morsel_max_rows.has_value() || *morsel_max_rows > 0,
                "pipeline_normalization: morsel_max_rows must be positive when specified",
                std::invalid_argument);
  }

  /** @brief Return an independently owned normalized plan and its graph metadata. */
  [[nodiscard]] std::pair<query_execution_plan, qep_metadata> normalize()
  {
    auto const tasks = _source->tasks();
    _ordered         = derive_ordering(*_source, _physical_relations).ordered;
    if (_morsel_max_rows.has_value() &&
        std::ranges::any_of(_ordered, [](auto const& entry) { return entry.second; })) {
      GQE_LOG_DEBUG(
        "morsel_max_rows={} (GQE_MORSEL_MAX_ROWS) may be overridden for ordered "
        "intermediate results: readers emit the complete input in one chunk "
        "to preserve row order.",
        *_morsel_max_rows);
    }
    clone_tasks(tasks);
    rewire_edges(tasks);
    complete_terminals(tasks);

    auto result = repair_bypasses(
      _builder.build(), _normalized_output_types, _normalized_stable_keys, _morsel_max_rows);
    auto metadata = analyze(result, _normalized_stable_keys);
    return std::make_pair(std::move(result), std::move(metadata));
  }

 private:
  /** @brief Clone all source tasks into the normalized plan. */
  void clone_tasks(std::vector<task const*> const& tasks)
  {
    _clones.reserve(tasks.size());
    _normalized_stable_keys.reserve(tasks.size());
    for (auto const* original : tasks) {
      bool const is_ordered_reader =
        dynamic_cast<intermediate_state_read_task const*>(original) != nullptr &&
        _ordered.at(original);
      std::unique_ptr<task> clone = is_ordered_reader
                                      ? std::make_unique<intermediate_state_read_task>(
                                          intermediate_state_read_task::single_output)
                                      : original->clone();
      auto* clone_ptr             = clone.get();
      _clones[original]           = clone_ptr;
      if (auto const types = _output_data_types->find(original);
          types != _output_data_types->end()) {
        _normalized_output_types.emplace(clone_ptr, types->second);
      }
      _normalized_stable_keys.emplace(clone_ptr, _stable_keys->at(original));
      _builder.add_task(std::move(clone));
    }
  }

  /** @brief Recreate source edges and insert required pipeline boundaries. */
  void rewire_edges(std::vector<task const*> const& tasks)
  {
    for (auto const* successor : tasks) {
      auto const predecessors = _source->predecessors(successor);
      auto const family       = classify(*successor);
      if (predecessors.empty()) {
        GQE_EXPECTS(family == task_family::iterate,
                    "pipeline_normalization: every task without predecessors must be an iterate "
                    "task",
                    std::logic_error);
        continue;
      }

      bool const has_breaker = std::ranges::any_of(
        predecessors, [](task const* predecessor) { return predecessor->is_pipeline_breaker(); });
      bool const read_all_predecessors =
        (family == task_family::stateless_transform || family == task_family::fold) && has_breaker;
      std::size_t pipelined_predecessors = 0;
      for (auto const* predecessor : predecessors) {
        bool const is_pipelined      = !predecessor->is_pipeline_breaker();
        bool normalized_is_pipelined = is_pipelined;

        task const* normalized_predecessor = _clones.at(predecessor);
        switch (family) {
          case task_family::iterate:
            if (is_pipelined) { normalized_predecessor = ensure_writer_after(predecessor); }
            break;
          case task_family::stateless_transform:
          case task_family::fold:
            if (!is_pipelined || read_all_predecessors) {
              normalized_predecessor = ensure_reader_after(predecessor);
            }
            break;
          case task_family::stateful_transform:
            if (!is_pipelined && have_different_physical_relations(predecessor, successor)) {
              normalized_predecessor  = ensure_reader_after(predecessor);
              normalized_is_pipelined = true;
            }
            break;
        }
        pipelined_predecessors += normalized_is_pipelined ? 1 : 0;
        _builder.add_successor(normalized_predecessor, _clones.at(successor));
      }

      if (family == task_family::stateful_transform) {
        GQE_EXPECTS(pipelined_predecessors > 0,
                    "pipeline_normalization: stateful transform must have at least one pipelined "
                    "predecessor",
                    std::logic_error);
      }
    }
  }

  /** @brief Add a materializing write boundary after each non-breaker terminal task. */
  void complete_terminals(std::vector<task const*> const& tasks)
  {
    for (auto const* terminal : terminals(*_source, std::span<task const* const>(tasks))) {
      if (!terminal->is_pipeline_breaker()) { std::ignore = ensure_writer_after(terminal); }
    }
  }

  /** @brief Whether the tasks have different required physical-relation provenance. */
  [[nodiscard]] bool have_different_physical_relations(task const* predecessor,
                                                       task const* successor) const
  {
    auto const predecessor_relation = _physical_relations->find(predecessor);
    auto const successor_relation   = _physical_relations->find(successor);
    GQE_EXPECTS(
      predecessor_relation != _physical_relations->end() && predecessor_relation->second != nullptr,
      "pipeline_normalization: missing physical relation for pipeline-breaker "
      "predecessor of stateful transform",
      std::logic_error);
    GQE_EXPECTS(
      successor_relation != _physical_relations->end() && successor_relation->second != nullptr,
      "pipeline_normalization: missing physical relation for stateful transform",
      std::logic_error);
    return predecessor_relation->second != successor_relation->second;
  }

  /** @brief Return a shared inserted reader after `predecessor`, creating it if necessary. */
  [[nodiscard]] task const* ensure_reader_after(task const* predecessor)
  {
    if (auto const found = _readers.find(predecessor); found != _readers.end()) {
      return found->second;
    }

    auto reader      = make_reader(_ordered.at(predecessor), _morsel_max_rows);
    auto* reader_ptr = reader.get();
    _builder.add_task(std::move(reader));
    auto const* reader_input = predecessor->is_pipeline_breaker()
                                 ? _clones.at(predecessor)
                                 : ensure_writer_after(predecessor);
    _builder.add_successor(reader_input, reader_ptr);
    _readers[predecessor] = reader_ptr;
    if (auto const types = _output_data_types->find(predecessor);
        types != _output_data_types->end()) {
      _normalized_output_types.emplace(reader_ptr, types->second);
    }
    return reader_ptr;
  }

  /** @brief Return a shared inserted writer after `predecessor`, creating it if necessary. */
  [[nodiscard]] task const* ensure_writer_after(task const* predecessor)
  {
    if (auto const found = _writers.find(predecessor); found != _writers.end()) {
      return found->second;
    }

    auto const output_types = _output_data_types->find(predecessor);
    GQE_EXPECTS(output_types != _output_data_types->end(),
                "pipeline_normalization: missing output data types for inserted writer",
                std::logic_error);

    auto gather      = std::make_unique<gather_task>();
    auto writer      = std::make_unique<intermediate_state_write_task>(output_types->second);
    auto* gather_ptr = gather.get();
    auto* writer_ptr = writer.get();
    _builder.add_task(std::move(gather));
    _builder.add_task(std::move(writer));
    _builder.add_successor(_clones.at(predecessor), gather_ptr);
    _builder.add_successor(gather_ptr, writer_ptr);
    _writers[predecessor] = writer_ptr;
    _normalized_output_types.emplace(gather_ptr, output_types->second);
    _normalized_output_types.emplace(writer_ptr, output_types->second);
    return writer_ptr;
  }

  query_execution_plan const* _source;
  task_output_data_types const* _output_data_types;
  task_physical_relations const* _physical_relations;
  task_stable_keys const* _stable_keys;
  std::optional<cudf::size_type> _morsel_max_rows;
  std::unordered_map<task const*, bool> _ordered;
  task_output_data_types _normalized_output_types;
  query_execution_plan_builder _builder;
  std::unordered_map<task const*, task const*> _clones;
  /** Stable keys for cloned source tasks; normalization-inserted tasks are not identified. */
  task_stable_keys _normalized_stable_keys;
  std::unordered_map<task const*, task const*> _readers;
  std::unordered_map<task const*, task const*> _writers;
};

}  // namespace

pipeline_normalization_result::pipeline_normalization_result(query_execution_plan plan,
                                                             qep_metadata metadata)
  : _plan(std::move(plan)), _metadata(std::move(metadata))
{
}

pipeline_normalization_result pipeline_normalization(
  query_execution_plan const& source,
  task_output_data_types const& output_data_types,
  task_stable_keys const& stable_keys,
  task_physical_relations const& physical_relations,
  std::optional<cudf::size_type> morsel_max_rows)
{
  auto const tasks = source.tasks();
  GQE_EXPECTS(stable_keys.size() == tasks.size(),
              "pipeline_normalization: stable task key count differs from QEP task count",
              std::logic_error);
  std::unordered_set<stable_task_key, stable_task_key_hash> unique_stable_keys;
  unique_stable_keys.reserve(stable_keys.size());
  for (auto const* task : tasks) {
    auto const stable_key = stable_keys.find(task);
    GQE_EXPECTS(stable_key != stable_keys.end(),
                "pipeline_normalization: QEP task has no stable task key",
                std::logic_error);
    auto const inserted = unique_stable_keys.insert(stable_key->second).second;
    GQE_EXPECTS(inserted, "pipeline_normalization: duplicate stable task key", std::logic_error);
  }

  auto [plan, metadata] =
    normalizer(&source, &output_data_types, &physical_relations, &stable_keys, morsel_max_rows)
      .normalize();
  return pipeline_normalization_result(std::move(plan), std::move(metadata));
}

}  // namespace qep
}  // namespace gqe
