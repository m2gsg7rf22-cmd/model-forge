# Multi-direction isolated GLU coverage development

## Formal v3 outcome: FAILED

Source a0d055655ee62d052d362ed3e2d4308d7261f733 was independently reviewed and
committed before entropy. CI40162660787 passed12tests/0failures/0errors/0skips
on each of Linux/macOS/Windows, including actual public learner integration,
plus types/lint. No native packages were rebuilt.

The three fresh owned teachers yielded59/61/60 initial candidates and60/62/61
after refinement. All were rejected by the registered63or64branch gate before
rescue, coupling fitting or heldout evaluation.436320calibration vector queries,
0sealed evaluation calls. Per-teacher elapsed2.60–2.78s; maximum sampled process
tree RSS246857728bytes (~235.42MiB), under180s/512MiB. The independent public
outcome audit checked commitments and recomputed public residuals; full-map
accuracy is unknown. The run does not measure63rescue efficacy because that
branch was never entered. Preserve v1/v2/v3 outcomes and do not retry teachers.

Formal evidence: experiments/evidence/isolated-glu-fullmap-v3/.

## Distinct post-failure development hypothesis

A residual subspace of dimension2(64−n), rather than a2D angular grid limited
to n=63, can initialize several missing rank-one directions. Use the saved
public Hessians and candidates only; no teacher arrays or heldout. This
extends the residual-rescue implementation direction; it does not establish a
new identifiability theorem or invent the published general rescue idea.
See [Shi/Fu's isolated GLU paper](https://arxiv.org/html/2608.06631v1).

New separate source: examples/isolated_glu/coverage_multi.py. V3 source and
its formal inputs/outputs remain unchanged. The five geometric helpers were
copied to preserve frozen version bytes and independently checked for AST
identity. For each development round:

1. Build two least-residual tangent components per retained direction, project
   into original V128; require component rank2n and residual dimension128−2n.
2. Draw256fixed-seed Gaussian coefficient vectors in that residual subspace;
   take two largest-absolute-eigenvalue eigenvectors from each mixture.
3. Refine512starts with24 tangent Gauss–Newton steps against ORIGINAL V.
4. Merge candidates with squaredloss≤1e-8 and absolute cosine<.9995; preserve
   the input prefix and never truncate extras to manufacture count64.

The fixed configuration allows at most4rounds, public seed20261025. All three
saved datasets reached64in the FIRST round. Unknown/stalled/excess counts remain
possible, not successful coverage. Residual dimension/energy, low loss and64
candidates are not truth certificates. No independent matching to private
teacher directions or weights was performed.

| Saved dataset | Input candidates | Output | Residual dimension | Seconds | Sampled RSS bytes |
|---|---:|---:|---:|---:|---:|
|0|60|64|8|3.722|104972288|
|1|62|64|4|3.393|104808448|
|2|61|64|6|3.226|104529920|

Total controller10.908s; each job limited to180s/512MiB with process_and_children
monitoring. Periodic samples do not guarantee instantaneous maximum RSS.
Two new synthetic controls, strict mypy and Ruff passed locally. No cross-platform
claim is made for this new development script (v3's12-test CI is separate).

Independent review checked every512candidate residual against original V,
replayed tangent geometry, all random mixture/eigenvector starts and exact
selection, verified hashes/prefix/norms/no duplicates. Final retained squared
residual maxima1.86e-16,1.36e-14,1.96e-13. These are calibration consistency
results, not full-map or teacher-identity measurements.

## Public multiscale integration only

A separate plan, saved before this additional step, fits the frozen multiscale
solver on these64directions and saved public observations in new development
folders. All three produced ranks2048/64/64, positive magnitudes and finite
rank-one projected couplings. These are fitting-side calibration diagnostics;
there is no heldout functional result or true-gate/coupling comparison.
Independent public audit recomputed small-design/orientation ranks and finite rank-one output checks; magnitude rank64 remains a fitting-side report because pre-projection B was not retained and no refit was permitted. Saved-calibration odd-component relative errors were3.01e-5/4.95e-5/8.58e-5 (small) and2.89e-5/4.98e-5/8.18e-5 (large). These are training-observation consistency metrics, not heldout generalization. See multiscale-addendum.json in the public evidence.

No new oracle calls, private truth, consumed labels or private final-model
assets were used. Generated raw NPZ files remain outside Git.

## Next action

Implement, test and independently review a DISTINCT v4 wrapper/protocol with
bounded multi-rescue, exact selection/residual/initialization artifact audit,
explicit rejection of stalled or65+counts, complete resource accounting and
one-time fresh evaluation. Freeze it before drawing fresh teachers. Existing
v3 thresholds/access/distribution remain the comparison baseline. The separate
O3 finite-observation iteration-budget study is still pending.

Universal/full-Transformer reconstruction and original stored-weight recovery
remain unresolved. Private final-model gate CLOSED.
