# GLU precision diagnostic implementation plan

Goal: measure the next numerical-access bottleneck using the existing public fixture.
Spec: docs/superpowers/specs/2026-10-25-glu-precision-design.md.
Tech: NumPy/core execution monitor; no additional dependencies.

1. Write tests for closed-form SwiGLU Hessian versus numerical small fixture, exact quartic Richardson cancellation, separate rounding/execution reference semantics, symmetric axis layout and spectral metrics controls. Observe missing-module failure, then implement helpers in examples/isolated_glu/precision.py. Validate only these new paths with pytest/mypy/Ruff.
2. Freeze the fixed sweep JSON and helper source, independent-review the no-leak/public-only scope and operation/metric counts, commit before executing the public diagnostic. Source hash checked before run. Parent owns repository; reviewer owns external evidence only.
3. Run once through run_json(180s,512MiB) in durable external experiment directory with single BLAS threads. Require process_and_children; retain every grid result/query count and observed resource usage. A timeout is preserved and not silently restarted. No fitting or heldout reads.
4. Independent outcome review checks all48results, formula controls/source commitments/counts and does not turn an optimum selected on public diagnostic into sealed success. Document measured tradeoffs, checkpoint via ResearchMemory and choose the next actual method/preregistration from evidence. No native rebuild or old successful test rerun.
