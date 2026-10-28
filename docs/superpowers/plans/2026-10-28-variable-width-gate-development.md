# Variable-width gate development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Parent owns repository edits; separate reviewer challenges major claims. Steps use checkbox syntax for tracking.

**Goal:** Build a separately reviewed observations-only experimental gate-candidate search; no full-function or universal-reconstruction claim.

**Architecture:** A strict adapter binds the existing public observation/choice seals to opaque jobs. A pure numerical kernel searches candidates under a generic r/2 hypothesis, and a bounded controller records every decision. Independent review and clean freeze precede the single development attempt.

**Tech Stack:** Existing Python3.12.15, NumPy2.5.4, pytest, strict mypy, Ruff; no dependencies or native rebuilds.

**Spec:** [Design](../../glu-variable-width-gate-development-design.md). Read it in full; design approval is not execution approval.

## Global Constraints

- Existing source31 pins/rank selector/markers/data stay unchanged. New modules only; no private assets, teacher factors, heldout, forward calls or paid services.
- Existing data are PUBLIC DEVELOPMENT ONLY. No new data/fitting/RNG until independent full-source approval and source/resource gates.
- 180s shared process+children,512MiB sampled TREE, BLAS1, stdout+stderr2KiB, actual free disk>=2GiB, combined artifact bound<1GiB.
- Six jobs job00..job05 in canonical order. Stable choice fixed; full original12seal/54public/24estimator inventory verified; basis[:r] only, even0<r<p and r<=192/n32 required.
- Seed mapping, all numerical constants and worst-case logging follow the design exactly. No seed retry, truncation, extra round, or post-hoc threshold change.
- All results are candidates, certificate=false/privategateclosed. Source/fixture approval, terminal checkpoint and independent saved audit are distinct gates.

## Review Focus

- A valid hash on the wrong full q-row basis must not substitute for its sealed first r rows (Task1).
- Original case/channel names or teacher metadata must never enter optimizer job inputs (Task1).
- Over-target counts and repeated sign directions must remain visible, not silently clipped (Task2).
- Partial IO/monitor failure must leave unknown/not_started cases and consume the marker (Task3).
- Worst-case logging must include rejected starts/mixtures/merges for all6*4rounds, not expected successful exits (Task3).

### Task1: Strict sealed public adapter

**Files:** Create `examples/isolated_glu/variable_gate_view.py`; test `tests/test_variable_gate_view.py`.

**Interfaces:** `load_jobs(public_root: Path, estimator_root: Path, commitments: dict[str, str]) -> tuple[GateJob, ...]`, with frozen `GateJob(job_id: str, basis: NDArray[np.float64])`; `GateJob` exposes derived n/r through checked geometry, no original ID/channel/width. Separate adapter provenance report retains mapping and input hashes. Reuse pinned schema/seal validators through explicitly reviewed calls; never import evaluator/collection/inventory into optimizer.

- [x] Write tests `test_sealed_rank_slices_full_basis`, `test_all_inputs_bound_before_jobs`, `test_optimizer_input_has_no_channel_metadata`: a tiny literal fixture selects rank2 from a3row orthobasis; mutation of any sealed input fails before jobs are returned; job fields contain only opaque ID and basis.
- [x] Run the three tests and record RED from missing module/interfaces; fixture data only.
- [x] Implement strict job validation, complete12/54/24 binding, canonical opaque mapping and first-r selection; factor small-fixture validation from registered6job guard. No current actual arrays loaded during tests.
- [x] Run focused tests, strict mypy and Ruff; independently review adapter source and fixtures, commit passing component. This is not full experiment approval.

### Task2: Pure candidate kernel with complete decision trace

**Files:** Create `examples/isolated_glu/variable_gate_search.py`; test `tests/test_variable_gate_search.py`.

**Interfaces:** `search_job(job: GateJob, initial_rng: np.random.Generator, rescue_rng: np.random.Generator, sink: TraceSink) -> GateResult`; `TraceSink.write(name: str, arrays: dict[str, NDArray[np.float64]]) -> None`; `GateResult(status: str, candidates: NDArray[np.float64], report: dict[str, object])`. Kernel accesses no paths, manifests, truth, callbacks or original IDs. Production constants are fixed; small fixture tests inject primitive outputs through a separate explicitly named internal test seam without weakening registered configuration.

- [ ] Write RED tests for odd/full/null abstention, zero/overshoot stop, exact-target skip, sign duplicates, no truncation, independent seeded streams, source-span loss recomputation and quartic geometry identity. The already established mathematical limitation is a fixture assertion, not a repeated research experiment.
- [ ] Implement documented dimension-adaptive initial Adam and refinement; copy/adapt rescue with generator injection and trace sink. Pin exact differences from fixed kernel. Persist initial/final RNG states and all generated starts; same numerical constants as spec.
- [ ] Enforce original-basis checks, finite arithmetic and scientific-vs-operational statuses. Recompute reports from candidates, never trust a search-side success flag.
- [ ] Run fixture tests, strict mypy/Ruff and independent kernel review. Commit passing component; do not run actual six jobs yet.

### Task3: Exact inventory, bounded controller and preregistration

**Files:** Create `examples/isolated_glu/variable_gate_inventory.py`, `variable_gate_controller.py`, `variable_gate_common.py`; `tests/test_variable_gate_controller.py`; `experiments/preregistrations/variable-width-gate-development-v1.json`; `docs/glu-variable-width-gate-development-protocol.md`.

**Interfaces:** `inventory() -> dict[str, object]` is pure integer/shape planning for all6 maximum-rank192 jobs and4rescue rounds. Controller `run(config_path: Path, output_root: Path) -> dict[str, object]`; default CLI outer monitor uses existing bounded launch primitives with pinned source digests. Distinct global marker `variable-width-gate-development-v1-attempt.json` before RNG/fit. No actual dataset generation.

- [ ] Write RED fixture tests for exact retained/rejected trace byte inventory, seed-domain mapping, changed sources/input hashes, duplicate/extra shards, overshoot, late nonfinite, failed final receipt, exhausted resources and incomplete case table. Confirm failures precede RNG where applicable.
- [ ] Implement exact schemas/shard caps and combined upper bound including maximum literal transport plus metadata reserve. The target96 is not an archive count cap: initial merge may retain256 starts, and a rescue overshoot may retain up to95+512=607 rows before scientific failure; inventory these full outputs plus all512 raw starts. Split large traces into inventoried bounded shards; never omit evidence to fit quota. If bound>=1GiB, redesign/review before actual execution.
- [ ] Implement clean-source/input verification, preflight TREE/transport/disk, exclusive marker, source checks across stages, six canonical statuses, partial counters and failure-safe final receipt. Scientific case failures continue, operational failures stop. Monitor covers controller+children, sampled RSS limitation explicit.
- [ ] Write preregistration with exact source hashes, public inputs/seals, opaque mapping, fixed constants/seeds, full inventory, no teacher-angle evaluation and no numerical success certificate. Run meaningful fixture integration/types/lint once; independent full-source/resource review closes all findings before commit/freeze.

### Task4: One attempt and independent saved-output audit

**Files:** Evidence under `experiments/evidence/variable-width-gate-development-v1/`; outcome doc plus RESUME/RESEARCH_LOG/ROADMAP and atomic ResearchMemory state/ledger updates.

- [ ] Verify full signed approval, unchanged clean committed source, inventory/disk/TREE preflight and absent marker; only then invoke default controller once. No background job or automatic retry.
- [ ] Preserve terminal result/partial status/resources/input and output commitments, checkpoint before audit. Do not classify target count as identified coverage.
- [ ] Separate verifier independently recomputes saved initialization/refinement/merge geometry and summaries without refit/RNG/query/teacher access under its own bounded monitor. Persist approval and limitations; any failure remains recorded.
- [ ] Update atomic queue and next discriminating reconstruction task, commit/push public evidence and validate actual resume. No gate opening or production support until separately justified science.
