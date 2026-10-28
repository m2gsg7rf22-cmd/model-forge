# Row-boundary and rank observability: zero-query development

Status: mathematical development following the independently audited fresh-width v1 failure. No new registered teachers, labels, queries or fitting. The v1 source, criteria and consumed marker stay immutable. This document is not a new data-generation approval or a successful reconstruction result.

## Why a structural gap can dominate

For an r-by-p observation matrix with r<p, the frozen selector has q=r available singular values, pads sigma(q+1..p) with zeros, and permits candidate k=q. Its final ratio is sigma(q)/floor, where floor=64*epsilon*max(r,p) in normalized singular units. That denominator comes from shape and numerical protection; it is not a measured noise component.

A literal matrix makes the mechanism explicit. Let delta=1e-6 and A=[[1,0,0],[0,delta,0]]. The normalized spectrum is (1,delta,0), and floor=64*epsilon*3. The two gap scores are1/delta and delta/floor. Since delta^2>floor, the final structural gap wins. If fine/coarse/Richardson all equal A, prefix stability is zero, so both frozen gap and stable select2 with certificate=false.

An independent literal square control diag(1,delta,delta) has spectrum(1,delta,delta); its final ratio is1 and both rules select1. This control changes the matrix; it is not an additional model probe or evidence that adding one SwiGLU probe repairs an observed failure. Full-rank square signal plus zero error and rank-one signal plus common error remain compatible interpretations. Neither selection is an exact-rank certificate.

The recorded literal experiment used six/nine fixed scalar entries, no RNG/teacher/model forward/fit, a monitored TREE worker and a synthetic TREE preflight. It demonstrates a matrix mechanism, not a newly validated model-reconstruction method. See the signed resource/results evidence under experiments/evidence/row-boundary-literal-v1/ after review.

## What the actual v1 audit established

The n32 cases have16*32=512 observation rows and p=32*33/2=528 columns. All three FP32 gap selections were512, versus reference32/96/192. The independently audited spectra/diagnostics confirmed the final padded-zero gap. Three-view stability avoided this boundary in the m16/m48 cases but returned null for proper reference192 in both channels. Near-parallel n16/m48 had reference96 at all three frozen cutoffs, while gap selected94 and stable abstained. Excluding the row boundary cannot by itself solve these other failures.

## Distinct objects that must remain distinct

- Matrix row capacity q is a sampling limit; a padded zero beyond q is structural.
- Rank of a saved finite observation matrix depends on arithmetic and noise.
- Reference numerical rank is defined by a cutoff and the sampled analytic matrix. It is not an exact rank over all real inputs.
- Atom-span dimension and original stored hidden width differ. Duplicate ideal-real SwiGLU units with identical g/u rows can be merged by summing their output columns, so original nonminimal width need not be identifiable from the ideal-real function. Floating-point summation/rounding can change under merging; this is not a bitwise equivalence proof for FP32 execution.
- A recovered subspace is not a complete GLU function or a decoder backbone. Required gate membership, parameter identification and independent full-function checks remain separate.

Even a known additive error bound does not guarantee exact rank. Let M0=diag(1,0,0), M1=diag(1,delta,0), and Z=(M0+M1)/2. Both errors have spectral norm delta/2, yet the two possible signals have different ranks. A method given only Z and an error ceiling at least delta/2 cannot distinguish these cases without more assumptions. Multiple identical views do not remove this ambiguity.

## Next design requirements, before any new data

1. Treat a row-limited selected boundary explicitly as censored/uncertain. Replacing a false rank with abstention is a diagnostic change, not successful recovery under the proper-reference endpoint.
2. Study an acquisition rule based on public geometry: at least ceil(p/n) probes gives rows>=p. For n32 this is17, rather than16. This removes structural padding beyond available rows; it does not guarantee signal/noise separation, precision robustness, gate recovery or success for near-parallel units.
3. Keep a possible finite paired16-versus17 comparison separate from algorithm changes and threshold choices. Use distinct fresh owned teachers and preregister nested-observation dependence, all cases, precision/noise controls and unchanged failure endpoints. Do not accept a new method against consumed v1 labels or use a width cap chosen from their outcomes.
4. Specify the observation/error contract needed for any stronger rank claim. An epsilon floor is not a bound on structured FD/FP32 error; common bias and weak directions require separate investigation.
5. A new producer/view/inventory/controller/source review and clean freeze would be required for different probe counts. The frozen v1 view and counters cap16 probes/545280 calls; merely editing a request cannot authorize17. Recompute all schemas/chunks/counters/byte/resource caps before entropy, with a distinct global marker and no retry.

Universal reconstruction, bias, HVP access, weaker interfaces and full backbone recovery remain open. The final private-model gate stays CLOSED.
