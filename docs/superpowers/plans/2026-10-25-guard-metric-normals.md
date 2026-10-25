# Guard metric normals — DRAFT, SYNTHETIC prerequisite only

Read the independent guard-aware direction validation and preserve every prior quality failure. Implement only a known-candidate derivative mapping on tiny synthetic cases before proposing an actual optimizer. Do not read saved candidate/calibration/consumed PUBLIC941027..941032data, query any target or update an actual candidate in this prerequisite.

## Derive first, then test

For physical candidate error e=f-Y and one gate g_i with fixed coupling C_i, define unit Jacobian response J_i(x)delta=silu'(x.g_i)*(x.delta)*(C_i x). These are derivatives of known candidate computation, never target gradients.

Prefer the smooth group squared-relative metric r2=||e_group||²/||Y_group||². Its gradient is2J_group*e_group/||Y_group||². First-order nonincrease of r2 is equivalent to that of relative error when ||e_group||>0; it says nothing about a finite nonlinear step. Zero response-group norm is invalid. Zero-error/zero-gradient degeneracy must be explicit; do not manufacture a constraint or infer finite retention from a flat derivative.

For maximum absolute group residual M=max_j|e_j|, a unique nonzero maximizer has gradient sign(e_j)*J_j. With ties, the directional derivative is the maximum of ALL active signed gradients; it cannot be replaced by one arbitrary argmax. Derive exact ties and a prospective near-tie inclusion policy, preserve provenance indices/signs/values and do not truncate silently to8planes. If a later selected plane budget is exceeded, fail closed pending separate method design. Non-maximal coordinates may become active under finite steps: exact full-forward guards remain mandatory.

A homogeneous inequality normal.T delta<=0 describes selected current first-order nonincrease. Current and initial metric bounds in the previous protocol are two bounds on the SAME current metric, not gradients evaluated at separate initial points. No actual experiment protocol is implied by this draft. Mapping into the cone solver must preserve row normalization semantics, its conservative geometry domain and no-target-access contract.

Test independent finite differences/complex-step for squared-relative gradients (including factor2/physicalunweightedY), independent Jacobian-coordinate derivatives for unique maxabs, exact tied/opposite/zero degeneracies, nonuniform scales, signs and originalindexprovenance. Include a case where picking one tiedmax would admit a direction that worsens another. Independent reviewer must challenge every derivative and ambiguity. Strictfinite/shape/dtype/row/outputbounds and nonmutation; <=180s/512MiB if worker controls are used. No actual fit/query/evaluation.

After synthetic source is reviewed and committed, a DISTINCT actual protocol would still need frozen starting artifact/calibration manifests, all exact current+initial relative/maxabsolute checks, rank/conditioning/tie/nochange failure policy, complete outside-Git numeric evidence with compact transport, fixedsolve/trialbudgets, prospective newPUBLIC941033onlyifchanged, pairedrelativeANDabsolute reporting and unchanged.001criterion. Do not register/use941033in this synthetic prerequisite. Universal reconstruction open/privatefinalmodelgateCLOSED.


Implementation checkpoint: examples/isolated_glu/guard_metric_normals.py;10focused controls/types/lint passed, independent review pending. Contract docs/glu-guard-metric-normals.md fixes near1e-12M, alllogicalrecords before8cap, computedzero omission/provenance, zero-M/reference/metricunderflow failures. Fullfourgroup ties reject; no actualfit/sourceapproval yet.
