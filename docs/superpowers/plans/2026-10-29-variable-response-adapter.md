# Variable response adapter implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Parent edits repository; the existing independent reviewer challenges the complete component. ResearchMemory is the atomic persistent ledger required by AGENTS.md.

**Goal:** Build a strict observation adapter for the six-case chosen-input/full-response calibration contract, using fixtures only before full read-protocol approval.

**Architecture:** A public-metadata compiler produces an immutable exact transcript whitelist without opening the truth directory. A separate bounded streaming view validates all permitted files before exposing opaque jobs, revalidates each pair before delivery and rehashes the whitelist at completion. There is no solver, target callback, CLI experiment or new data generation in this component.

**Tech Stack:** Python3.12, stdlib, existing NumPy, pytest/Ruff/strict mypy.

**Spec:** docs/glu-variable-width-response-view-design.md (independently approved design only).

## Global constraints

- Default6cases/2steps/17probes/2112x32 input and response;204pairs/408archives/430848vectorrows.
- Encoded archive<=544768bytes; NPY header<=4096; one exact member, native finite float64.
- Public commitments<=2MiB, other public metadata/report<=128KiB; no truth manifest/directory enumeration/context/seed/reference/evaluator access.
- No actual raw transcript access, RNG/query/fit/teacher/private/heldout; production source unchanged until complete later read-protocol review.
- Validation keeps one pair only; immutable bytes-backed delivered arrays; distinct validation/delivery/completion reads and delivery counters.
- Late failure is terminal; partial iterator remains partial; no restart/cache/retry. Completion rehashes all408+metadata without decoding arrays.

## Review focus

- Pinned public metadata can be changed late: initial and completion checks must reject it.
- A late invalid archive must prevent exposure of every job, not just its own case.
- Streaming delivery failure must preserve lower-bound read counters and exact delivered counts.
- Encoded ZIP header tricks must fail before large array allocation.
- A consumer may retain arrays: adapter working-set promises do not bound that consumer or replace future TREE monitoring.

### Task1: Public commitment compiler

Create examples/isolated_glu/variable_response_manifest.py and tests/test_variable_response_view.py.

Interfaces: `load_manifest(repository: Path) -> TranscriptManifest`; immutable profile, metadata pins and pair descriptors. An internal explicit literal profile/pins seam exercises the same compiler with tiny synthetic files; production defaults are fixed constants, no fixture CLI option.

- [x] Write fixtures with literal metadata/transcript files and forbidden context/manifest sentinels. Assert canonical exact whitelist/counts/pinned counters/opaque case map, missing commitment/duplicate JSON key/late public hash/native-bool counter rejection; production function never opens a raw transcript.
- [x] Run tests; observe missing-module failure before implementation.
- [x] Implement bounded strict JSON/metadata pins; exact expected names/hashes from prior public commitments and source lineage. Never read hashes from the truth manifest or accept arbitrary basename extraction.
- [x] Run compiler tests; expected PASS. No actual transcript source opened.

### Task2: Bounded streamed response view

Create examples/isolated_glu/variable_response_view.py; extend tests/test_variable_response_view.py.

Interfaces: `open_view(root: Path, manifest: TranscriptManifest) -> ResponseView`; `ResponseView.jobs` returns only opaque `ResponseJob` metadata plus single-use iterators of immutable `ResponseBatch(inputs,responses)`; `finish() -> dict` requires every batch delivered and terminal completion rehash. Parent `report()` retains provenance and separate counters. No source paths/channel/width exposed in batch fields.

- [x] Write failing tests for all-before-exposure validation, canonical immutable pairs, single-use/partial/exhaustion semantics, late pair/source mutation, hash/schema/dtype/nonfinite/oversize/duplicate/member/header/symlink rejection and forbidden read/enumeration sentinels.
- [x] Run tests and observe expected missing-view failures.
- [x] Implement strict ZIP/NPY headers before data allocation, pair streaming, frozen array copies, counters including failed physical reads and explicit state transitions. Do not cache scientific arrays or offer retry.
- [x] Run all changed-path tests, Ruff and strict mypy on both modules. Do not repeat old experiments or broad completed tests.

### Task3: Independent component review and durable checkpoint

Create docs/glu-variable-width-response-adapter.md; preserve review evidence under experiments/evidence/variable-response-adapter-v1/. No read runner yet.

- [ ] Review final source/tests/descriptor arithmetic independently using only small literal controls; full actual-read/source protocol approval remains FALSE.
- [ ] Close significant findings with failing regression tests then minimal fixes. Preserve preliminary failures and exact signatures.
- [ ] Update ResearchMemory/docs atomically, verify old source pins unchanged, commit/push discovery-lab and run actual report-only resume once.
- [ ] Next complete a separate bounded read-only validation protocol/controller/inventory and review before any actual transcript access; do not substitute view validation for reconstruction or full-map scientific evidence.
