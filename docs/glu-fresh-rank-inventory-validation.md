# Variable-width study: inventory contract validation

This is an independently reviewed planning component. It does not authorize teacher creation, RNG, queries, fitting or the full fresh study. The private final-model gate remains closed.

The new `examples/isolated_glu/fresh_rank_inventory.py` uses only standard-library integer and shape arithmetic. The known profile multiset belongs to producer/controller planning and must never enter the signed observation-only estimator view. Existing selection, math, storage and view sources remain unchanged.

| Store | Scientific shards | Total byte cap, including manifest | Per-shard cap |
|---|---:|---:|---:|
| Truth and raw query evidence | 1,608 | 278,964,064 | 2,101,248 |
| Public FD observations | 144 | 86,878,656 | 2,101,248 |
| Estimator choices and bases | 48 | 86,918,256 | 6,785,056 |
| Privileged evaluation | 24 | 1,229,184 | 57,280 |

The plan totals 1,824 scientific shards, 545,280 vector evaluations and 768 callback batches. These are proposed counts, not performed evaluations. Raw inputs and outputs each have 119,193,600 scalar payload bytes; four Hessian views have 115,015,680 bytes.

Seven maximum-shape literal transport examples plus their manifest are bounded by 17,591,940 bytes. Adding all store caps and an 8MiB allowance for bounded metadata, atomic temporary copies, engine requests/logs and the external marker yields **479,970,708 bytes**. The future pre-data gate requires at least 2GiB actual free disk. This disk inventory does not prove RSS, TREE monitoring, elapsed time or runtime resource availability.

Schemas preserve thin basis rows `q=512` at input/output dimension32, with full symmetric-coordinate dimension `p=528`. The width96 context has atoms `(192,528)` and preactivations `(16,96)`; evaluator gate residuals have shape `(6,96)`. Each NPZ has at most16 fields, immutable positive integer shape tuples of1–4 dimensions, bounded ASCII field names and no reserved `file`/`allow_pickle` fields. Payload and header/ZIP allowances are checked separately. Each manifest record uses a conservative188-byte bound. Future serializers must measure actual bytes and reject extras or budget overflow; the allowance is not a guarantee for every NumPy format/version.

Nineteen targeted tests passed, including seven maximum-shape zero-array roundtrips through the unchanged store, plus strict mypy for the new module and Ruff for both changed Python files. Independent review reproduced the arithmetic and constructor boundaries. It also constructed four empty real stores with the per-store limits, without creating scientific arrays or teachers.

The review caught and resolved mutable list shapes, unsafe/unbounded field names, excessive dimension counts and an integration mismatch between the small evaluator budget and the store's default8MiB shard cap. Failed probes remain preserved alongside corrected probes in `experiments/evidence/fresh-rank-inventory-v1/`. Seven prior dependencies were verified unchanged. Literal unit roundtrips are not monitored experiment receipts.

Still required before any fresh data: a distinct width96 producer and raw-response counter pipeline; nonce-protected seed/assignment commitments before generation; truth commitments before estimator launch; separate processes/views with all24 choices sealed before truth reveal; fixed energy formulas and all-case qualification; exact metadata inventory and runtime global-cap enforcement; independent full-source review, a clean frozen commit and monitored disk/transport/TREE gates. No new model reconstruction route or scientific success is claimed by this component.
