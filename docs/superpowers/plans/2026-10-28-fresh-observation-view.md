# Fresh observation view implementation plan

> Inline parent implementation using executing-plans/TDD; independent reviewer owns only temporary evidence. User authorized autonomous execution without design questions. No teacher generation is authorized by this component plan.

**Goal:** Implement the observation-only filesystem view required for the distinct variable-width study; keep full-platform and full fresh-study requirements open.
**Spec:** docs/glu-rank-fresh-width-validation-draft.md, independently approved DESIGN ONLY; fullsource/data/run approval false.
**Architecture:** A new loader reads only public cases.json plus arrays/manifest.jsonl and whitelisted FD chunks. Width-dependent truth/context headers/manifest/assignments are outside this root, never parsed. The loader accepts public geometry and compute-channel metadata, not seeds/width/teacher type. Source-level isolation is not an OS sandbox.
**Stack:** Python3.12/NumPy, frozen shared bounded raw/read_arrays, svec, ShardStore and hash primitives. No RNG/teacher/forward imports or existing signed source changes.

## Task1 — public metadata/FD view

Files: examples/isolated_glu/fresh_rank_view.py, tests/test_glu_fresh_rank_view.py.

- [ ] Write literal synthetic fixtures and failing tests for absent FDView, then implement strict public schema. FDView(root:Path,cases_sha:str,manifest_sha:str); immutable Case(name,n,outputs,probes,compute); cases tuple; packed(case_name)->dict[fine/coarse/richardson,float64matrix]; verify_unchanged()->dict of used hashes and fixed public metadata commitments.
- [ ] Exact root inventory cases.json and arrays/. Exact JSON fields schema/steps/cases, schema=modelforge-fresh-rank-observations-v1, steps=[.3,.6]. Each case fields name,n,outputs,probes,compute; names c00..c11-fp64/fp32, matching compute fp64/fp32_compute. Strict nonbool ints n/output1..32/probes1..16, unique1..24cases. Production24case/fixed16probe validation remains future controller responsibility, not synthetic seam inference.
- [ ] cases.json and manifest each<=128KiB; duplicate JSON keys rejected. Require valid nonnull lowercase64hex external commitments. Manifest only name/bytes/sha256 with valid positive native-int bytes<=8MiB. Exact FD names case-{fine,coarse,richardson}-h00/h01.npz derived from ceil(probes/8), <=144 records. Arrays directory must have precisely those files and manifest.jsonl. Reject reference/context extras before any NPZ allocation.
- [ ] Load bounded shape hessian=(min(8,remaining),outputs,n,n) via shared hash/header validator; apply frozen svec, pack(probes*outputs,p); finite nativefloat64 only, return readonly arrays. No width-dependent headers or full truth metadata exist in this view. Verify public metadata/manifests/input hashes/inventory unchanged before return and at final explicit check.
- [ ] RED→GREEN tests: literal nontrivial svec, no access to adjacent truth sentinel, extra width/seed/class fields, full-manifest context record, symlink/extra file, digestnull/nonhex/bytesbool, metadata/hash mismatch, wrong NPZ shape/field/dtype, source mutation and duplicate keys/truncatedmanifest. Assert rejection before any read_arrays call when failure is metadata/manifest-level. Do not generate actual GLU data.
- [ ] Changed-path pytest, strict mypy source and Ruff; independent component source/schema/mutation review and exact hash evidence. Freeze component separately; full variable-width protocol remains queued/unimplemented and data forbidden.

## Subsequent required work (outside component approval)

Distinct secret seed/mapping generator and raw-return/counter runner (width96,16probes,24cases); independent nonce/assignment/truth commitments in correct chronology; two-process observation seal then privileged evaluation, fixed primary/side sensitivity/all-case qualification; exact aggregate byte/record inventory and monitored preflight/controller/global marker; independent fullsource approval/cleancommit BEFORE generation; terminal checkpoint then saved-output audit. Existing frozen selection/math/store, prior markers/candidates and private gate remain unchanged. Component success cannot mark this parent protocol or universal mission complete.

## Review focus

Legacy full-manifest parsing leaks context sizes. New root must reject it. Tiny mapping-space commitments require independent secret nonce/seed, future generator responsibility. Side-cutoff failure applies also to full-reference cases, future evaluator responsibility. No test in this plan proves resource availability, complete scientific outcome, OS security or statistical blinding.
