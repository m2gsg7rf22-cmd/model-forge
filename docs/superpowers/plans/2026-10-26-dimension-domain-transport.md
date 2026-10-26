# Dimension-domain bounded artifact transport implementation plan

> **For agentic workers:** inline implementation; parent owns repository files, independent reviewer writes outside Git.

**Goal:** Preserve sharded public diagnostic evidence within disk/transport limits before a scientific runner exists.
**Architecture:** Exclusive new store root, uncompressed finite-FP64 NPZ shards, streaming hash manifest, preallocated serialized-byte reservations. Fail-stop on any write error; never delete/overwrite/retry partial artifacts. A separately reviewed runner/controller supplies counters, registered data, monitoring, source gates and overall protocol.
**Tech Stack:** Python3.12, NumPy, pytest/mypy/Ruff.
**Spec:** docs/glu-scaling-observation-domain-draft.md; approved pure math source remains unchanged.

## Global constraints

No registered teachers/probes/RNG, actual reference calls, private assets, consumed941xxxdata or experiment approval. Each shard<=8MiB; entire store<=1GiB, with1MiB reserved for future bounded reports. Manifest overhead counts against available budget. Only strict finite float64 arrays; source arrays never pickle/object. Exclusive filenames and root, no symlink path ancestry; immutable completed artifacts, partial write remains evidence. Stream hashes without unbounded file reads. Preflight must check>=2GiB free disk, validate worst actual-shape synthetic transport before any scientific data, and eventually share180s/512MiBpolledtree budget (controller not implemented here).

## Task1

Files: examples/isolated_glu/scaling_domain_store.py; tests/test_glu_scaling_domain_store.py; docs/glu-scaling-domain-transport-validation.md.
Interfaces: ShardStore(root, shard_limit_bytes=8MiB,total_limit_bytes=1GiB-1MiB); save(name, **float64arrays)->dict(name/bytes/sha256); bytes_reserved; terminal; verify_manifest(root)->summary. Check free space and a worst-shape synthetic transport via separate explicit function; no work at import.

- [ ] RED tests absent module; finite dtype/shape/name/schema checks; size caps before any data write; exclusive newroot/filenames; root/name symlink/traversal rejection; output roundtrip/digests; failed write/partial manifest failstop; corrupted/extra/missing/truncated manifest rejection; synthetic maxprofile shard roundtrip and insufficient disk rejection.
- [ ] Implement minimal store; GREEN targeted tests, strict mypy/Ruff.
- [ ] Independent source/transport review on literal/zero/arange synthetic arrays only; capture signed evidence and material fixes with regressions.
- [ ] Atomic research checkpoint, commit/push. Parent diagnostic protocol remains queued: counters/runner/controller/preregistration/source freeze/actual experiment still required.

Implementation note: default global store1GiB-1MiB reserves report headroom; manifests count and have independent1MiB/8192record caps.8192 accommodates separate rawinput/response shards; exact full diagnostic inventory remains a required runner preregistration task, not proven here. No scientific data or process-tree resource guarantee is inferred from unit tests.
