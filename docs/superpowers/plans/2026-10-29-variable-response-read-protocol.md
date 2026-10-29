# Read-only response protocol implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Parent edits repository; existing independent reviewer checks the whole protocol. Atomic ResearchMemory is the durable ledger required by AGENTS.md.

**Goal:** Validate the approved full-response calibration view once under source, resource and one-shot gates; no solver/model claim.

**Architecture:** A pure scalar inventory and bounded protocol IO support a distinct controller/default outer launcher. The controller performs literal preflights, commits a marker before actual transcript opening, records ordered hash-only delivery receipts, and checks final adapter counters and output layout. The default uses the signed adapter unchanged; tiny literal seams are Python-only tests.

**Tech Stack:** Python3.12/NumPy/psutil, existing bounded run_json, pytest/strict mypy/Ruff.

**Spec:** docs/glu-variable-width-response-read-protocol-draft.md and independently approved response view design/component.

## Global constraints

- All408whitelisted files before any delivery;204orderedpairs/430848rows; three passes1224archive reads/816decodes in a terminal default run. No actual data before full-source approval/clean commit/preflights/exclusive marker.
- Source data read-only, no truth manifest/context/seed/reference/evaluator/directory inventory. No RNG/model/query/fit/private/heldout.
- Shared180s/512MiB sampled process+children/BLAS1/2KiB; actual>=2GiBfree disk, combinedplannedupper679665861bytes<1GiB.
- Persist preflight/outer receipts before testing success; unavailable failure resources remain unknown. Retain failures and never retry marker/output.
- Exact bounded descriptors/delivery chain/reports/output layout; partial counters/unknown uncommitted delivery explicitly preserved.
- Signed prior adapter/source bytes unchanged. A read-only engineering pass is not reconstruction science or permission for a solver.

## Review focus

- A preflight reports failure after returning a receipt: preserve it before raising.
- A model-free view fails after partial delivery: distinguish adapter-delivered pairs from controller-committed chain prefix.
- A default outer launcher sees a completed controller but bad TREE: outer result must remain failed with receipt retained.
- Source/config/marker changes late: do not label delivery completed merely because iteration ended.
- Encoded receipt bounds and empty engine directories: no hidden output family or source-data copy may escape the inventory.

### Task1: Inventory and bounded protocol IO

Create examples/isolated_glu/variable_response_inventory.py, variable_response_protocol.py; tests/test_variable_response_protocol.py.

- [x] Write failing tests for exact scalar bounds, caps, exclusive fsynced marker, ordered chain/native counters and malformed terminal reports; observe missing module.
- [x] Implement scalar profile/read/storage arithmetic, strict JSON/atomic bounded writes, tree receipt validation, source/config/review gate and exact delivery/counter validation.
- [x] Run focused tests and inspect output.

### Task2: Controller/default outer and literal fixtures

Create examples/isolated_glu/variable_response_controller.py; extend tests/test_variable_response_protocol.py; create formal protocol and preregistration after source stabilizes.

- [x] Write failing tests for complete tiny synthetic pipeline, no premarker data, existingmarker/output refusal, bad disk/TREE/transport, late/partial view failure, source/marker mutation, exact layout and failed outer after completed child; run before implementation.
- [x] Implement source verification/committed default paths, literal transport, monitored preflights, marker, strict adapter delivery loop/chain/progress, final receipts and outer failure persistence. Only stdout small status.
- [x] Run new protocol tests plus adapter integration tests (changed path only), strict mypy new modules and Ruff. No old scientific reruns/native builds/broad already-completed suite.

### Task3: Independent full source and actual read gate

- [ ] Freeze exact runtime/source/prereg/docs/tests/public inputs and complete inventory; independent full review must approve registered defaultCLI and close blockers using RED→GREEN controls.
- [ ] Atomic checkpoint/source commit/push. Actual one-shot read is allowed only after clean committed freeze and all actual gates; no retry after marker.
- [ ] Persist terminal checkpoint and independently audit saved receipts without reopening the transcript merely to repeat successful reads. Keep scientific reconstruction/private gates closed.
