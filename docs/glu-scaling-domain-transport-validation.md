# Dimension-domain artifact transport: component validation, not experiment readiness

Source examples/isolated_glu/scaling_domain_store.py supports the pending sequential dimension/width diagnostic. This component creates no scientific teacher/probe, RNG, reference query or fit, reads no consumed941xxxdata/private assets and advertises no new backend. Approved pure math source at7256078 remains unchanged. The parent diagnostic protocol is still QUEUED: complete registered runner/controller, global byte inventory, vector-call reservation/counters, preregistration/source gate and shared180s/512MiBpolled-tree execution BEFORE scientific data generation.

## Storage contract

Exclusive new root and immutable simple basename NPZ files. Reject symlink ancestry, existing names, manifest replacement/size changes and NumPy-reserved field names file/allow_pickle. Arrays must be nonempty native float64, no objects/pickle or nonfinite values. Check aggregate nbytes BEFORE elementwise finite allocation; oversized broadcast views therefore cannot trigger huge finite masks. Serialize one uncompressed shard in bounded memory, then enforce serialized size<=8MiB before disk writes. NumPy header overhead can make a payload below the raw cap fail the serialized cap; failure occurs before writing or reserving the artifact.

Default store limit=1GiB-1MiB, leaving1MiB for future bounded reports/controller metadata. Manifest bytes also count against the store limit, and separately cannot exceed1MiB or8192records. The8192record cap accommodates future separate persisted inputs BEFORE query and returned responses AFTER query plus derived arrays; a full fixed-design inventory has NOT yet been implemented or verified to fit these caps. Future runner must freeze it before data. This is an upper disk reservation, not an assertion of the final experiment size.

Reserve exact serialized shard+manifest-line bytes and name BEFORE opening the artifact. Short/failed writes are terminal for that store object. Retain partial shard/manifest files, retain reservations, forbid further saves, never delete/overwrite/retry. Partial evidence is not a completed artifact. No fsync/crash-durability or malicious concurrent-writer guarantee; a killed process may leave a partial file/manifest without a terminal report. A restarted store cannot reuse the existing root. This is not a global one-time scientific reservation: the future controller must supply one before registered data.

Manifest verification checks strict JSON keys (duplicates rejected), bounded exact schema/count/line termination, positive bounded sizes, unique names, non-symlink files, streaming SHA256 and exact root inventory. Extra/missing/truncated/corrupt evidence fails. It does NOT decompress NPZ or independently audit numerical meaning, and a hash is not a scientific truth certificate.

## Explicit synthetic preflight

synthetic_preflight checks at least2GiB free space before creating output (the optional free_bytes seam is test-only; actual controller must omit it), then roundtrips three largest proposed profile SHAPES: raw inputs+responses(4,528,32), Hessian chunk(8,32,32,32), full symmetric basis(528,528)+spectrum. Zeros/ones/identity arrays only;0RNG/registered data/queries. No work occurs at import. The future runner must use these shard shapes or separately review a larger transport; combining all Hessians into one file would violate the contract.

This tests individual worst-shape transport, not total experiment disk/RAM/time. A real monitored synthetic pre-data run and final worker tree scope are still required. No claim that process-only receipts are tree monitoring, or polled RSS is a continuous peak/hard reservation.

## Verification

25 targeted tests passed in0.07s; strict mypy1source/Ruff2files passed. Initial20tests RED because module absent. Additional regressions observed RED then GREEN: reserved NumPy names(TypeError or silent dropped field); oversized broadcast view reaching finite allocation; replaced manifest symlink appending to external file; duplicate JSON keys accepted last-value-wins. Test writes remain temporary synthetic artifacts only. Other controls cover roundtrip/hash/exclusive root/name, strict dtype/finite/empty, path/symlink rejection, prewrite shard/total budget caps, preserved partial writes/fail-stop, inventory and corruption/truncation, worst-shape transport and free-disk rejection.

Independent source/transport review is saved separately after exact byte binding. It approves only this component, never a scientific experiment or resource-controller readiness. Private final-model gate CLOSED; broad precision/access/scaling and universal reconstruction unfinished. Last paired941033qualityFAILED and its candidate/one-time marker remain frozen.
