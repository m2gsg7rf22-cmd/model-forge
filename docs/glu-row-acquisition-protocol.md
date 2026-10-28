# Paired row acquisition v1: complete-source candidate

Status: independently reviewed DESIGN, implementation candidate awaiting FULL source approval and clean freeze. No registered data-generation approval is implied by fixture tests. The parent design is glu-row-acquisition-study-draft.md. Frozen fresh-variable-widthv1 remains failed, consumed and byte-identical; private-model gate CLOSED.

## Exact acquisition and conditions

Three DISTINCT owned generic bias-free input/output32 teachers use widths16/48/96, hash-permuted anonymous c00..02 identities. Each has paired FP64 and FP32-compute channels using the same original factors. Draw g/u/d independent normals scaled.2, then17 normal probes scaled4 and64 normal directions. For each teacher, a domain-separated SHA256 seed with bit255 set comes from a newly drawn256-bit entropy seed; seed/nonce/assignment canonical JSON is committed before any factor RNG. Domain includes /row-acquisition-v1/. Old seeds/data/teachers/markers are excluded. No duplicate/near-parallel mutation is performed in this distinct generic acquisition study.

Collect17probes*2steps*.3/.6*4signedcorners*528columns for each of6teacher/channel cases:430848vector slots in204callbacks. The first16 account for405504 and the extra probe25344. Preserve diagonal duplicates. Frozen RawCounter is reused with an exact430848 reservation inside its existing bounded upper limit; it persists inputs/attempt intent before callback and detached responses before derivative arithmetic. Strict finite schema governs persisted counts. Callback exceptions/mutation preserve partial lower bounds, terminate without retry and never invent zero counts.

Public data has6basecases,54FD-only archives, chunks8/8/1. RowView exposes12condition aliases in basecase order(c00fp64,c00fp32,...), each p16 then p17. The p16 view reads the exact first two committed chunks used by p17; it neither rewrites nor re-queries them. Public metadata contains anonymous identity, n/output/channel/probes/chunk geometry only. No width/factors/seed/nonce/private manifest sizes enter the estimator. This is logical source/interface separation, not OS sandboxing or adversarial/statistical blinding.

The public estimator reuses the unchanged generic fresh_rank_estimation.estimate routine through the FD-view interface, after exact6basecase/12condition/54archive schema checks. Its registered=False flag disables ONLY the obsolete24case guard; the new registered guard is mandatory. All12choices/bases/projectors/diagnostics and all54source hashes seal before truth is read. The unchanged rank_selection.py uses strict1e-10, gap10, stability.01 and resolution floor64epsilon*max(rows,p), with no new width cap, boundary exclusion or threshold tuning. For16, q512<p528; for17, q528 and candidate count527. Certificate always false.

## Own-prefix reference and paired endpoint

Only after all12sealed choices and24headers/descriptors/ranks/projectors validate does privileged evaluation verify the pre-estimation seed/truth/public commitments and open registration/context. Every condition uses its OWN analytic Hessian prefix:16 or17probes. FP32 smooth reference rounds factors once to float32 then differentiates in FP64; it is not a derivative of rounded execution. Gate/random-control arrays are exactly shared within each paired condition; a context digest binds this equality.

Reference cutoff order1e-11/1e-10/1e-9 and primary1e-10 are unchanged. Reference sensitivity, zero reference and unexpected fullp fail this proper-subspace study and stay in all denominators. Proper endpoint requires exact reference rank, geometric lost/extra<=.01, true-gate residual<=.01 and random min strictly above true-gate max. Geometric/relative-residual/per-SVD-component formulas and clipping/collision diagnostics reuse the frozen verified helpers; the new row-domain validator allows544rows. No truncation or QR compression discards weak components. All available singular vectors and structural zeros are retained. Relative Frobenius residual is not squared energy; per-component amplitudes square to normalized component energy fractions. Zero denominators use null/-1 sentinels and cannot rescue an endpoint.

Known/half/double-width controls remain diagnostic. Both gap and stable are reported separately for all6cases at each budget. Primary17 success requires all6; retain every paired pass16/pass17, reference-rank/sensitivity change, gain and loss. Do not choose a winning rule, infer independent six-teacher statistics, hide changed reference geometry or claim repair of the original24-case study. Channels and nested prefixes are dependent. Passing capacity rows>=p does not establish rank/width identification, excitation, noise robustness, useful gates, full functions or stored weights.

The controller validates exact12ordered IDs/probes/basecases/shared-context hashes, native reference/rank/qualifies fields, recomputes qualification from fixed report metrics, binds ranks/reference/sensitivity/qualifies to evaluation NPZs, and independently derives paired/by_budget/primary summaries. Canonical JSON comparison rejects integer/Boolean substitutions. Independent saved-output numerical audit remains required to verify the underlying computations against committed raw arrays.

## Exact inventory and resources

New allocation-free row_acquisition_inventory.py extends the frozen ShardSpec shapes without editing old files. Scientific shards:432truth,54public,24estimator,12evaluator=522. Truth includes6contexts,204input+204response archives and18reference chunks. Estimation has both512x528 and528x528 bases and both512/527 candidate diagnostics.

| Store | Byte upper bound including manifest |
|---|---:|
| truth | 252767552 |
| public | 80447400 |
| estimator | 160954800 |
| evaluator | 1018320 |

Ten maximum literal transport families require31403888upper-bound bytes including manifest. Largest shard6786616; per-store shard AND total caps are passed explicitly, including evaluator maximum86720. Combined exact planning ceiling534980568bytes includes8MiBmetadata/transient/final-report reserve; it is below1GiB. These are conservative per-field4096header/188manifest-record bounds, not measured occupancy. Actual>=2GiB free disk is checked before entropy. No independent1GiB allowance per store.

Default row_acquisition_controller.py JSON CLI wraps the controller in frozen run_json: shared180seconds,512MiB polled process-and-children RSS, BLAS1, combined stdout/stderr<=2048bytes. This includes controller NumPy/validation/imports and all workers/Git descendants; the small host launcher is excluded. RSS sampling can miss short peaks and is not an OS reservation. Per-stage resource receipts supplement the outer receipt. Direct controlled_run/run_protocol does not prove the complete resource contract.

Within the outer monitor, source/policy/runtime/inventory checks require clean committed discovery-lab and exact24dependency/provenance file pins plus required tests/docs/preregistration. Synthetic TREE-scope and ten maximum literal-zero transport roundtrips/hashes/schema checks precede actual disk and an exclusive fsynced global marker, paired-row-acquisition-v1-attempt.json, binding source/config/output BEFORE entropy. Output must be a new direct child of the durable experiments directory outside Git. Existing/partial marker/output means no retry or overwrite.

Collection, public estimation and privileged evaluation run sequentially with necessary inputs only. Source/config/review are checked between stages; seed/truth/public/seal and global marker hashes remain fixed. Final exact stores/names/schemas/manifest/caps/layout reject extra reports/NPZs/directories. Stage receipts are saved before failure rejection; killed/unreceived worker counters remain unknown/lower bounds. The outer monitor binds the final controller-report digest after termination, writing outer-monitor.json; any outer failure dominates inner success. Retained files plus reserve must remain<1GiB, metadata<=16files each<=128KiB, engine transients bounded/removed.

## Completion contract

Full independent source/preregistration/test/doc review, fixes and clean freeze precede any registered entropy/teacher/query. Then an independent verifier may start ONE defaultCLI run. Persist terminal counters/receipts/consumed marker before a separately monitored saved-output audit; never repeat the experiment to fix a failed criterion. Do not claim original forward values or random draws were independently replayed when the audit only verifies saved commitments and derived quantities. No paid API, private assets, native rebuild or old experiment repetition is involved. Broader universal/full-function/weight/bias/HVP/weak-access/backbone research remains open and the final private-model gate CLOSED.
