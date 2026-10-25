# Packaged isolated SwiGLU candidate route

This supported core route ports the frozen v4 learner for an isolated bias-free FP64 SwiGLU block with input32, hidden width64 and output32. It consumes authorized saved continuous full-vector observations, not generated text or final token logits. NumPy float64 storage does not certify the source precision, architecture, rights or query coverage. No teacher weights, heldout labels or oracle callbacks are supplied. The backend makes zero oracle calls.

## Observation contract

Profile: `isolated-swiglu32x64-fp64-v4`. Supply a regular `observations.json` and fixed siblings `hessians.npz` and `multiscale-observations.npz`, without symlink ancestors. JSON must contain exactly schema_version(integer1), profile, authorized(booleantrue), hessians_sha256 and multiscale_sha256 (lowercase64hex SHA256 of the exact NPZ bytes). Duplicate keys, nonfinite values, unexpected arrays, incorrect shapes/dtypes/hashes and oversized input are rejected. Each archive and inflated arrays are capped at128MiB; JSON at2MB. Arrays must be finite float64.

Hessians NPZ key `hessian` has shape(64,32,32,32), indexed probe/output/input/input. Probe RNG seed20261024 generates64 Gaussian vectors with std4. For each i<=j use h=.003 and [f(x+h*ei+h*ej)-f(x+h*ei-h*ej)-f(x-h*ei+h*ej)+f(x-h*ei-h*ej)]/(4*h*h), storing symmetric entries. This collection requires135168 full-vector evaluations, including repeated point calls.

Multiscale keys: small_x/small_odd shape(4096,32); large_x/large_odd shape(512,32); even_pos/even_neg shape(528,32). RNG931082 generates small_x standard Gaussian first, then large_x. small_odd=.5*(f(.001*x)-f(-.001*x)); large_odd=.5*(f(1000*x)-f(-1000*x)), without normalization. even_pos/even_neg are f(v)/f(-v), for canonical vectors e_i first then e_i+e_j with i<j in increasing nested order. This adds10272 evaluations: total145440 if collecting a new dataset. Collection needs existing authorization; this route does not perform provider calls or authorize spending.

## Execute through CLI or MCP

```sh
python -m modelforge.cli --home /absolute/workspace create candidate
python -m modelforge.cli --home /absolute/workspace engine reconstruct_glu32 '{"project_id":"PROJECT_ID","observations_path":"/absolute/data/observations.json"}'
python -m modelforge.cli --home /absolute/workspace engine predict_glu32 '{"project_id":"PROJECT_ID","artifact_id":"RETURNED_ARTIFACT_ID","inputs":[[32_NUMERIC_COORDINATES]]}'
python -m modelforge.cli --home /absolute/workspace findings PROJECT_ID
python -m modelforge.cli --home /absolute/workspace export PROJECT_ID /absolute/export.zip
```

Replace the illustrative32_NUMERIC_COORDINATES placeholder with32 numbers; prediction accepts1..4096 rows. MCP exposes `reconstruct_glu32(project_id,observations_path)`, `predict_glu32(project_id,artifact_id,inputs)`, `create_project`, `read_findings`, `export_project`; inspect the connected server schema first. Older stable servers lack this route.

Reconstruction uses a512MiB/180s process-tree cap and one BLAS thread. Prediction uses512MiB/30s. Fixed search failure, timeout or incomplete monitoring is a recorded failure, not a candidate to export. Only successful service findings authorize prediction/export, bound to the original manifest hash. Rehashed student/report files cannot silently replace an accepted operation. This is integrity/provenance checking, not protection against an attacker controlling the entire same-account service database.

## Artifact and interpretation

Accepted exports include only student.npz, manifest.json and report.json for each candidate, plus ordinary project findings. Raw observations are not automatically exported. Student contains gates(64,32) and rank-one couplings(64,32,32). Independently reload outside the service:

```python
from pathlib import Path
from modelforge.glu32_io import load_candidate
from modelforge.glu32 import predict
gates, couplings = load_candidate(Path('/absolute/extracted/candidate'))
y = predict(gates, couplings, x)  # finite float64 rows,32 coordinates
```

Standalone hashes establish self-consistency; retain the accepted service manifest hash separately for provenance. Generated metadata keeps assumptions_verified, heldout_verified, numeric_certificate, original_weights_recovered and all_inputs_equivalence_proven false. Prediction reference_verified is false. Validate independent heldout data under a preregistered protocol before stronger scientific claims. This fixed-profile finite approximation is not whole-Transformer, original stored-weight, arbitrary-input equivalence or universal reconstruction. The final private-model gate stays closed.
