# Isolated GLU antipodal observation contract

Requires authorized full-vector evaluation of the *isolated* bias-free block at chosen continuous inputs, including negatives. Token outputs, final logits and passive intermediate traces are insufficient to fill this contract. Dimension d is1..64; output width is1..64. Hidden width need not be known for the even component.

Create the ordered probes with `modelforge.antipodal.canonical_probe_points(d)`: first e_0..e_(d-1), then e_i+e_j in lexicographic i<j order. Query every positive point and its negative: d(d+1) vector observations (1056 when d=32). Batching does not reduce that count; no query is issued by the reconstruction tool itself.

Store this exact public JSON schema (arrays shown symbolically, not literal placeholders to run):

```json
{
  "schema_version": 1,
  "observation_kind": "isolated_glu_antipodal",
  "family": "bias_free_swiglu",
  "authorized": true,
  "dimension": 32,
  "positive_outputs": [[0.0]],
  "negative_outputs": [[0.0]]
}
```

Replace the sample arrays with528 rows of32 outputs for this example; both matrices must have shape[d(d+1)/2,outputs]. Allowed families: bias_free_swiglu or bias_free_geglu with sigma(t)-sigma(-t)=t. No extra fields: do not embed weights, secrets or test data. A file is limited to16MiB; numbers must be real finite values. If canonical probes cannot legally be queried, retain the access gap rather than invent observations.

`reconstruct_antipodal(project_id, observations_path)` persists `quadratic-even.json`, the public observations and a report. Evaluate the result as E(x)=x^TQ[k]x for output k. It reconstructs E(x)=(f(x)+f(-x))/2 under the stated family assumption. It does not reconstruct the odd component or original/full GLU. The report's heldout_verified and numeric_certificate remain false: JSON conversion to float64 does not establish source precision, bias-free implementation, or model equivalence.

Use fresh authorized paired queries for independent heldout verification. A misspecified biased/non-GLU target can interpolate calibration probes and fail away from them. Do not fit or select methods against heldout responses. The orientation solver `recover_orientations` is conditional: candidate gate magnitudes/directions and rank-one coupling matrices must come from a separate learner-side procedure with preserved provenance. Do not pass private teacher parameters to it and describe the result as weight discovery.
