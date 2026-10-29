# Variable response view v1: read-only validation protocol draft

Status: UNIMPLEMENTED / UNREVIEWED DRAFT. The adapter component is separately reviewed, but no actual transcript read, numerical solver, fitting, fresh query, teacher generation or evaluation is authorized by this draft. Private final-model gate CLOSED.

## Purpose

Validate the complete approved chosen-input/full-response view under measured resource and lineage gates, without reconstructing or tuning a model. This is an engineering dataflow validation on disclosed owned calibration, not new scientific success. The six gate-count outcomes and all prior failures remain unchanged. See docs/glu-variable-width-response-view-design.md and docs/glu-variable-width-response-adapter.md.

## Prospective single attempt

A new `variable_response_controller.py` will use a distinct preregistration `variable-response-view-v1.json` and exclusive global `variable-response-view-v1-attempt.json` in the durable experiments directory. Refuse an existing marker or output root; no automatic retry after any failure. Persist a terminal receipt before any independent outcome audit. Do not alter frozen gate/row sources or old markers.

Only after complete source/protocol/inventory review and clean committed freeze:

1. Verify exact source/dependency/public-metadata pins, runtime version and Git ancestry/clean state. Pin NumPy/Python and `src/modelforge/execution.py` plus all imported runtime modules; no generator/search/refine/teacher import is needed. The source check itself opens no transcript.
2. Run the actual shared outer monitor (180s, 512MiB sampled process-and-children, BLAS1, stdout+stderr2KiB) and its literal child-scope preflight. Persist each resource receipt immediately before testing its success. RSS is sampled, not a continuous OS cap.
3. Require actual free disk at least2GiB and a conservative complete combined artifact bound below1GiB. Do not traverse a truth/teacher directory to measure prior artifacts; rely on the pinned independently audited parent/gate totals and explicit plan upper bounds.
4. Serialize exactly one maximum-shape literal input and response NPZ under monitoring, verify their bounded schemas and immutable roundtrip, then discard in-memory references. Preserve their two files, transport manifest and receipt. This is literal transport, no source transcript.
5. Recheck clean source/public metadata/runtime/inventory, then fsync an exclusive one-shot marker BEFORE opening any transcript. Bind output path, source commit, config, prior component approval, public metadata hashes, exact408descriptor commitment and limits. An ordinary failure consumes the marker; preserve partial receipts.
6. Load `load_manifest(repository)` from its four pinned public metadata files and open exactly the whitelisted transcript root through `open_view`. The parent controller may not enumerate or open adjacent context/reference/seed/evaluator/manifest files. Validate all408archives before any job delivery. Preserve `ViewValidationError.report` on an initial failure.
7. Consume jobs00through05 and34batches per job in order. Do not retain batches after checks. Record fixed pair index and input/response digest/shape only; check returned immutability, exact row count and finiteness, update a bounded delivery commitment chain, then release the batch. No training objective, split, random sampling, ranking, original factors, inferred width, saved candidate or predictive callback is permitted. These are disclosed calibration observations, not heldout predictions.
8. Call `finish()` once: rehash every allowed archive and public metadata without array decoding. Require complete counters, zero queries/fits and closed private gate. Recheck source/config/marker, disk, report bounds and exact new-output layout. Persist a bounded controller report and let the outer monitor finalize its own receipt. A kill/timeout leaves incomplete batches UNKNOWN; no inferred completion.

## Exact read accounting

Production dimensions imply204pairs/408archives/430848vectorrows; raw payload220,594,176bytes. The encoded cap544,768bytes per file bounds one whole whitelist pass by222,265,344bytes. There are three archive passes: validation408decode reads; delivery408decode reads; completion408hash-only reads. Thus a terminal complete view has1,224archive reads and816array decodes, with maximum666,796,032encoded read bytes. These are repeated IO bytes, not newly generated data or disk allocation. Metadata reads and code/source hashing are counted separately. Incomplete/failed reads are lower bounds and must not be padded to expected completion.

Require validation and completion counters in addition to delivery. Complete delivery is204pairs/430848rows/220594176payloadbytes. Status completed alone is insufficient. Archive hashes bind the data; the controller chain must bind ordered per-batch input/response digests and indices, not numerical model outputs. A batch delivered and then released still counts as delivered; a pair whose response fails validation does not. One partial/closed iterator cannot be restarted.

## Prospective storage/resource inventory

Two transport files at544,768each plus a4,096byte transport manifest are bounded by1,093,632bytes. Reserve8MiB for every new metadata/report/config/marker/delivery commitment file, all progress/partial/error variants and outer receipts; exact per-file caps and a literal layout are still to be implemented and independently verified. No copy of the220MBsource transcript is made. Prior paired-v1 conservative artifact upper bound534,980,568 plus gate-v1 actual retained135,203,053 plus the new9,482,240bound totals679,665,861bytes, below1GiB. Repository source/evidence is separately versioned; raw source archives remain outside Git. This planning arithmetic is not a measured free-disk or RSS receipt.

The adapter validator keeps one pair at most; the controller must not cache batches or return raw arrays through stdout. A literal maximum-shape transport preflight is not the actual data validation. All resource limits cover controller+children jointly; host launcher scope and final-write timing limitations must be stated explicitly. No paid API, background scheduling, native rebuild or GUI permission wait.

## Tests/review required before actual access

Use tiny literal profiles and temporary Git fixtures, never actual transcript arrays. Challenge source/public metadata/marker/output changes, failed TREE/disk/transport receipt preservation, one-shot refusal, forged descriptor, before-first and late validation errors, partial delivery, completion rehash, bounded report serialization and post-completion outer failure. A controlled synthetic end-to-end run must have deliberately distinct validation/delivery/completion counters. Full source approval must bind the registered source/profile/default CLI, exact inventory, preregistration and fixtures before execution. Component approval alone does not open the read gate.

## Outcome and next science

A successful read-only result means the permitted observation pipeline validated/delivered the registered calibration with preserved commitments/resources; it proves no gates, coverage, weights or function. An independent saved-receipt audit should verify lineage, source/marker chronology, layouts/counters/ordered chain and resource receipts. Avoid a second raw transcript traversal merely to repeat successful engineering checks; independently reviewed source/fixtures plus saved receipts can audit the declared engineering endpoint, with that limitation explicit.

Next design a distinct all-six-case full-map reconstruction study that uses the approved observations and a prospectively declared architecture/width hypothesis, optimizer/initialization, calibration/selection split and complete trace/resource bounds. Existing fixed-gate coupling polish failures cannot be relabeled as joint recovery. Fresh authorized owned full-function validation sealed from model selection remains necessary. Correlated local stencil rows are not independent broad-domain samples; text/logit-only access does not expose this view. Bias, precision, HVP/scaling, weaker access, decoder backbone and universal reconstruction remain open.
