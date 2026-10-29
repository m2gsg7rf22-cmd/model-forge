# Joint response development implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Parent edits repository; the existing independent reviewer challenges each completed scientific boundary. Persist decisions/results atomically with ResearchMemory.

**Goal:** Implement and then test the all-six-case joint full-response development hypothesis, preserving the full reconstruction mission and closed private gate.

**Architecture:** A pure NumPy SwiGLU kernel differentiates and updates all three branches with an externally supplied fixed schedule and trace observer. A distinct parent join combines the frozen rank/response observation contracts and passes only opaque width-hypothesis jobs. Exact inventory/controller/preregistration/full-source review precede actual observation loading, registered RNG or fitting.

**Tech Stack:** Python3.12, NumPy, existing source/IO/run_json infrastructure, pytest/strict mypy/Ruff.

**Spec:** docs/glu-joint-response-development-design.md (DESIGN review must be complete before code).

## Global constraints

- Six cases; width hypothesis stable-p17rank/2, proper evenrank<=192, n/output32; no true width/basis/directions/channel/path/teacher/reference input to learner.
- Training probe centers exclude3/7/11/15; guard includes them. Guard is consumed calibration, also used by old17probe rank; no generalization/heldout claim.
- Joint G/U/D, one4000update run/case, batch256, Adam.9/.999/1e-8, cosine lr.01→.0001, no clipping/decay/restarts,17checkpoints, earliest finite minimum guard MSE.
- Train-only RMS normalization; same paired optimizer streams after all gates, fixed public SHA256domains and4000x256schedules saved before updates. Equal102paired input archive commitments avoid replay/cacheall6.
- No actual observations/RNG/fit before complete new protocol/source/inventory approval/cleanfreeze/resourcegates/newmarker. No oldtruth/context/angles/private-reference/fullfunctioneval/heldout/private-model access.
- Shared180s/512MiBTREE/BLAS1/2KiB, actualdisk2GiB/combinedartifacts1GiB. Exact worst-case logging/source/budgets must be reviewed before use.

## Review focus

- New width metadata could silently disclose truewidth or override rank: require strict join and hypothesis label.
- Guard might affect gradients/normalization: derive scales from training only; guard only selection and never unseen-test label.
- Last update/checkpoint or a callback may fail: preserve completed prefix and report unknown intermediate updates, no prefix promoted as successful failed fit.
- Scale/gauge transformations may change prediction: independently check physical parameter folding and gradient signs/factors.
- Shared paired RNG could accidentally reset/restart or differ shapes: bind groups, widths, initial arrays and schedules before fitting; count cases/steps exactly.

### Task1: Pure full-map numeric kernel

Create examples/isolated_glu/joint_response_kernel.py and tests/test_joint_response_kernel.py.

Interfaces: immutable `Parameters(g,u,d)` with bounded native FP64 arrays; `predict(parameters,x)` in bounded chunks; `loss_gradient(parameters,x,y)` returns `.5mean` loss and G/U/D derivatives; train-only normalization and folding helpers. Then `fit(initial,train_x,train_y,guard_x,guard_y,schedule,observer)` implements the frozen policy without filesystem/RNG/target access. Caller injects all initial arrays/schedule.

- [x] Write literal tests against a separate scalar sigmoid forward and central finite differences for every G/U/D entry, exact loss normalization and physical scaling invariance; reject nonfinite/native-schema/alias errors. Observe missing-module RED first.
- [x] Implement the minimal pure kernel and run the new numeric tests; no actual observation or registered seed.
- [x] Test fixed Adam first update and scheduling/17checkpoint/tie/step0 selection, train/guard separation, complete schedule consumption and ordinary callback/numeric failure handling using tiny literal data. Observe RED, implement, then run changed-path tests/mypy/Ruff.
- [x] Independent kernel/component review before adding actual-data protocol. No scientific result from gradient fixtures.

### Task2: Strict joined observation and exact trace/inventory

Create separate joint-response adapter/inventory/trace modules and tests. Do not modify any existing signed adapter.

- [x] Test canonical six mappings/rankhypothesis and paired input commitment equality, whole-source validation before RNG, no forbidden metadata/files, immutable jobs and bounded single-case assembly on tiny literal stores.
- [x] Implement fixed train/guard indices, training-only scale validation, shared paired initial/schedule construction AFTERgates, precommitted initial/schedule/RNG trace and bounded snapshot/prediction storage.
- [x] Compute exact ALLsixmaxwidth96record/shard/encoded byte bounds including all4000losses/fullschedule/17weights+moments/predictions/failureprogress/transport/reserves; require combined<1GiB. Review before actual data; redesign if bounds fail.

### Task3: Full protocol/source review and one development attempt

Create preregistration, controller/default outer launcher, literal source/transport/failure tests and formal protocol.

- [ ] Freeze source/dependencies/runtime/observation pins/seeds/optimizer/selection/inventory and output contract; full independent approval and clean commit before any actual source observations or registered RNG/fit.
- [ ] One shared-budget defaultCLI attempt after actual preflights/disk and new exclusive global marker. Record ALLsixoutcomes/partial/unknowns, no retry/restarts/threshold changes.
- [ ] Terminal checkpoint and independent saved candidate/metric/resource audit within the approved calibration boundary. Candidate forward verification is allowed in that future contract; original teacher/RNG/optimization replay is not.
- [ ] Preserve actual results/failures and define a DISTINCT fresh-owned sealed full-function/invariant study only after its complete query/storage/source approval. Development fit/guard loss never opens private/universal/full-platform gates.


2026-10-29 component boundary: case/stream/trace sources independently reviewed; shared changed-path integration59tests1.01s, strictmypy6/Ruff12pass. The remaining Task2 checkbox includes integrating precommitted metadata/initial/schedule/checkpoint observer into a case driver; it is deliberately unchecked. Draft full protocol: docs/glu-joint-response-development-protocol.md. No actual observations/registered RNG/fitting/sourcefreeze/resource approval yet.


2026-10-29 integrateddriver component approved; Task2 complete as source components, actual gating/PCGfactory/globalaccounting/defaultCLI still Task3. Sixnew driver tests1.30s/mypy1/Ruff2; zeroactualobservations/registeredRNG/fit. Journal counts within1MiBcase allowance.
