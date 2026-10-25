# Isolated bias-free GLU v4 — registered fresh-teacher result

Classification: engineering verified and registered finite-sample criteria PASSED
under the stated isolated FP64 query assumptions. Universal/full-Transformer
reconstruction, all-input equivalence and original stored weights are NOT proven.

The source/protocol/review/readiness were committed at fad2ffe before fresh
entropy. CI40163909186 passed17tests/0failures/0errors/0skips per OS on Linux,
macOS and Windows; changed types/lint passed. No native rebuild. The17tests
cover the new v4 wrapper and multi-coverage controls, not the whole platform.

V4 preserves v1/v2/v3 outcomes: v1 transport failure, v2 all-three FAIL63/64/64,
v3 all-three FAIL60/62/61. No old teacher was retried or old labels consumed.

## Access and method

Three fresh owned Gaussian bias-free SwiGLU teachers, d32,width64,out32,
weight std.1/.2/.4. Complete float64 isolated-block vector responses at signed
continuous chosen inputs. Known activation/dimensions and bias-free metadata.
This access is not available from a token API by assumption.

Frozen Hessian search/refinement, then fixed bounded multi-residual rescue:
1..63candidate inputs, up to4roundsx512starts, original-V GN24steps;
64skips, other/excess/stalled results fail, no truncation. Every round is
committed and audited, including original-V losses, exact selection, seeded
starts, prefix, geometry and inventory. Magnitude/coupling/orientation stages
use only saved public odd/even observations. All three required rescue in
ONE round:61→64,60→64,62→64.

## Measured result

|Weight std|Max scale-group Frobenius relative error|Max absolute error|Seconds|Sampled peak RSS bytes|
|---|---:|---:|---:|---:|
|.1|2.9230621624e-5|.0356391399|8.631|284049408|
|.2|5.6561652477e-5|.4944648709|8.733|347717632|
|.4|1.1527661335e-4|9.0871620648|8.412|348717056|

All three passed ALL registered criteria: group relative error≤.001 at each
of four scales, min matched absolute gate cosine≥.9995, max oriented-gate
relative error≤.01, max rank-one coupling error≤.01,64correct orientations.
Inspect summary.json/public verification files for actual gate/coupling values,
per-scale errors and per-row percentiles; a group norm is not a per-input bound.
The absolute error9.0871620648 occurs at large input scale and is retained.

Each teacher used145440calibration vector evaluations and one sealed evaluation
of2048fresh points (512each at scales.04,1,4,30). Total442464calls =436320cal+
6144sealed. Controller26.0751s, max sampled teacher-tree RSS348717056bytes
(~332.56MiB), within180s/512MiB per teacher. Sampling is not an instantaneous
peak guarantee. Lightweight top-level launcher excluded, descendants included.

## Independent evidence and limits

Independent code/protocol review preceded entropy. Public outcome audit checked
all source/registration/observation/base/round/learner/student/manifest/verification
commitments, re-ran public fit_audit once each without fitting, recomputed
aggregate/numeric-threshold logic and checked resource scopes. It did NOT read
private teacher/heldout arrays or consumed markers and did not recalculate
sealed outputs a second time. The one-time isolated verifier generated those
metrics with frozen reviewed code; they are not fitting-side success flags.

Three random teachers are evidence for these finite cases, not an estimated
success probability across architectures, widths, precision or input domains.
No learned full-model tokenizer, attention composition or text-API reduction is
established. U/D reciprocal scaling ambiguity remains. Keep the private final
model gate CLOSED and preserve the full scientific objective.

The [primary GLU work](https://arxiv.org/html/2608.06631v1) also separates
isolated-vector access from final model outputs and discusses HVP extensions
and precision sensitivity. Future precision/scaling research must use a new
protocol; v4's labels may not be retuned or re-evaluated.

## Next delivery

Port the verified fixed numerical profile into a packaged candidate backend,
with strict NPZ inputs, real CLI/MCP routing, artifact reload/inference and export
checks. Prove the distributed route using the public known development fixture;
no new scientific success claim from transport tests. Document the exact
observation contract and failure modes in the installed agent skill. Maintain
original_weights/all_inputs/heldout_certificate=false for arbitrary user inputs
unless independently evaluated. This is separate from full-platform acceptance.
