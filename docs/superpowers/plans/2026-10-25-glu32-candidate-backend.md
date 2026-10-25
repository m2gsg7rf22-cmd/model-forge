# Deliver the verified GLU32 numerical profile to agents

Goal remains universal reconstruction research. This delivery exposes the actual
verified fixed isolated-block method; it is not a universal/full-model tool.
Current preregistered evidence: docs/isolated-glu-v4-result.md (three fresh passes).
Keep frozen examples/preregistrations/results and private final-model gate intact.

## Design/contract

- Profile: isolated bias-free SwiGLU, float64 observations, input32/hidden64/output32.
  Unsupported architectures/dimensions/activation/precision fail explicitly.
- Input: strict finite float64 NPZs for Hessians(64,32,32,32) and exact documented
  small/large/even arrays, matching frozen v4 observation shapes/query protocol.
  Cap ZIP allocation/file size, reject symlinks/extra keys/object dtypes/nonfinite.
  Agent-provided metadata is an assumption, never a scientific certificate.
- No new oracle/model/provider calls, teacher truth, pretrained downloads or training.
- Result: candidate gates(64,32), rank-one couplings(64,32,32), inference/reload API,
  calibration diagnostics, source/input/artifact commitments, sampled RSS/time.
  original_weights_recovered/all_inputs_equivalence/heldout_certificate=false.
- Ship the numerical code inside the Python package; no dependency on examples/,
  Git checkout/readiness, temporary venv or unstated developer filesystem paths.
  Experiment readiness gates continue to guard SECRET TEACHER GENERATION only;
  a user-data candidate operation does not create teachers or grant scientific truth.
- Use existing run_json whole-worker process-tree180s/512MiB bounds. Preserve
  fixed256initial starts/1600Adam/12GN; multi-rescue512starts/24GN/up to4rounds,
  no truncation; require64, originalV/rank/gap/norm/duplicate/multiscale guards.

## Implementation and meaningful verification

1. Add packaged numerical kernel src/modelforge/glu32.py and strict artifact/io
   module src/modelforge/glu32_io.py, with tests before source. Port the frozen
   math without changing old experiment bytes. Compare against original profile
   on the one already PUBLIC development fixture (seed572041), not old sealed data.
2. Route runtime_worker operation glu32_reconstruct and service method
   reconstruct_glu32 through existing bounded runtime machinery. Unique project
   output per invocation; structured failure; no incomplete success artifact.
3. Register actual CLI/MCP operation reconstruct_glu32 and discover its real schema.
   Extend artifact export allowlist for candidate model + safe manifest/diagnostics;
   raw private/teacher/heldout files must never enter export implicitly.
4. Test malformed/allocation/nonfinite/rank/gap/count/duplicate/excess/stall inputs,
   budget/failure cleanup, artifact tamper/reload and actual known-fixture inference.
   End-to-end CLI and MCP must create/read findings/export/reload real artifacts.
   No test may replace the learner with copied teacher weights.
5. Verify installed wheel in fresh external cwd, NumPy/core only (no source path
   masking or optional Torch/PySide). No native desktop rebuild. Independent agent
   reviews math equivalence, output semantics, IO/export and actual agent flow.
6. Target changed-path tests/types/lint and3OS integration CI. Update bundled and
   installed skill observation routes with real commands/schema and restrictions
   only after verified delivery. Never advertise a nonexistent full-model route.
7. Atomically checkpoint results/failures and queue precision/access/scaling
   research. No old consumed labels reused and no private final-model work.

## Subsequent science (separate, not executed by this plan)

Investigate finite-precision Hessian observability and signed-pair conditioning,
biased/metadata-negative controls, dimension/width/HVP scaling, and the missing
final-output-to-internal-block reduction. The primary source
https://arxiv.org/html/2608.06631v1 also leaves token-API reduction open and
uses HVP input extensions; these are research leads, not implemented guarantees.
No new fresh/sealed run until its distinct protocol/source review is committed.
