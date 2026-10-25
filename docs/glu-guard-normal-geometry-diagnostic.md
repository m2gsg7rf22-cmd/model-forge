# READ-ONLY calibration normal geometry diagnostic

## Prospective scope

Determine whether the reviewed current guard-metric normals of the fixed failed-quality candidate lie inside the cone solver's conservative numerical geometry domain. This is not optimization, candidate improvement, target-gradient access or a quality evaluation. No KKT/direction solve or cone_direction call is made. No candidate, coupling, gate or calibration is changed; no new target call or fresh evaluation/941033reservation.

Bind saved candidate2f207b106c8dae1f31348c1b46b4ea4bbd08d1f667a5ddb78fa1c86a8002378b, compact1976byte metadata5aad40a6f33ab0b374e95086e4ab977f5b8e17a1d73d5d761ab03f7a039ba624, original fullfitreportd79a4c7818b5269feb89fcef0da8a197257c17162de5a9acdc30ed5c941a377b, raw observationsc8414e9215ac3e14967c23bccf4f0240efe83c6265b5339398f1851b074be210, manifest350c8650cd20facef31915b9b31622b9bc19f2b35d77285b64d07611821e4372 and original calibration manifestb0141882daf5def2a20e70e32051be46399c94e17c094e05dc35d54267501a96. Full3MBfitreport is HASH ONLY, not passed to the2MBmetadata parser. Compact metadata binds the fullreport and all candidate/calibration hashes. Guard inputs regenerated from882031must match the raw1024vectors at four scales.01/1/10/100; no query to recreate responses.

## Data flow and classification

Compute complete known candidate guard predictions/error once. For each of64units, call the reviewed metric_group on all four physical groups. Collect ALL computed logical records and group failures, rather than aborting on the first bad unit. Every group also has shared raw exact/near-max coordinate/sign/value provenance from the full error. If a group fails before producing normals, its error and raw provenance remain; do not claim complete computed normals for that group. A zero maximum has undefined finite signed-plane count here and is reported with logical_plane_count=null, never an invented count.

The fixed full-profile8logicalcap is applied BEFORE zero-normal omission. A diagnostic may retain up to32computed group planes as failed-profile evidence, but any full-profile logical count>8fails mapping and NEVER enters geometry. This extra STORAGE capacity does not expand the optimizer's domain or permit truncated constraints. Computed_FP64_zero records stay in the details even when absent from A. Empty A with a valid8logical profile is explicitly retained as empty/unconstrained geometry with mask0.

For mapped A only, normalize each finite/nonzero row and inspect every subset0..2^m-1, up to256perunit and16,384total. Persist spectra/ratios/masks. Independent normals with normalized subset singular ratio<=1e-5are outside the conservative domain; distinguish those from exact dependencies using Fraction rank on ORIGINAL binary64 normals. More rows than dimensions are necessarily dependent. Store numerical failures as failed subset records and keep the remaining masks. No linear system or candidate direction is solved. This is a fixed domain diagnostic, not a numerical completeness/forward-error/identifiability theorem.

Summary denominators retain all64units: mapping_failed versus mapping_succeeded; mapped units separately geometry_in_domain/geometry_outside_domain/geometry_failed, plus a distinct empty_geometry count. No unit is silently dropped and no failed mapping counts as a successful QP.

## Evidence and execution

Save raw physical inputs/responses/prediction/error, padded normal matrices with valid counts, per-subset spectra/lengths/status codes and complete group/metric/subset detail outside Git. Large details.json is hash-bound from compact summary metadata beside numeric-evidence.npz. The engine returns a bounded receipt, not all details. Fixed2KiBworker stdout and existing metadata parser limits remain unchanged. Exclusive attempt/root prevents automatic retries; failures remain preserved.

8NEWfocused synthetic controls passed; strict mypy and Ruff passed. Worst-size persistence control includes actual64unit×256subset full detail records,512logical records and matching NPZ dimensions, with details>2MBand a successful2KiBengine response. This is a test fixture for the exact persistence path, not an actual candidate or geometry result. Mapping/tied failure traces, zero-normal/empty geometry, near-independent/exact-dependent classifications and zero-maximum count semantics were checked.

Before actual saved-data reads, source/controller/preregistration/dependencies must be reviewed independently and frozen in a clean commit. Controller180s total/512MiBpolled worker tree/BLAS1, process-sampling limitations disclosed. Independently audit saved normals, provenance, all spectra/classifications and hashes without solve/query. The original failed controller/missing buildRSS and all prior quality failures remain unchanged. Guard is reused adaptive calibration, not independent heldout data. Private final-model gate CLOSED.

## Outcome

Pending source review and one registered read-only run. No applicability result exists at this snapshot.
