# Owned decoder parameter ambiguity

This bounded control advances the main reconstruction research. It does not build, train or integrate the gated private final model.

For the first self-attention layer of an owned tiny Llama decoder, choose a nonzero scalar c and replace the bias-free projections by Wq'=cWq and Wk'=Wk/c. The corresponding queries and keys scale reciprocally. Position-specific RoPE is linear and global scaling commutes with it:

`(R_i(c q_i))^T(R_j(k_j/c)) = (R_i q_i)^T(R_j k_j)`.

Repeating key heads in grouped-query attention is linear too. Thus attention scores, softmax probabilities, value aggregation and every subsequent decoder operation are unchanged in ideal real arithmetic. Q/K weights differ. This is a concrete ambiguity within a known architecture, even with all input/output behavior available. Unique original-parameter recovery requires extra information or an explicit convention; finite behavioral matching cannot decide which gauge was original. Equivalent computation and finite approximation are separate objectives.

The statement assumes the shown projections are bias-free, or that biases are rescaled as well. The example refuses biased Q/K projections because it changes weights only. It also assumes no overflow/underflow and does not establish identity for every finite-precision kernel, quantization or input. Layer normalization, output-head weights and tokenizer are unchanged in this witness; this is not a reconstruction algorithm for the backbone.

The actual first run used a project-owned random Llama with19,616 parameters,64 input sequences of length8 and all position/vocabulary logits. First-layer query/key weight L2 changes were0.64298/0.21720; observed maximum logit difference0, argmax disagreements0, exact numeric value equality true. These finite observations supplement the algebraic identity; they do not prove useful language quality or extraction from a text API. Exact initial source is preserved at commit79ba7b2 and its report is experiments/evidence/decoder-gauge-control.json. The later explicit bias guard does not retroactively change that source hash.

The first attempt to read a persistent Documents fixture stalled at its config file and was terminated without a result. The completed probe instead generated a fresh owned fixture in /private/tmp, with no downloads, then used execution.run_json with2GiB sampled RSS admission and120s timeout. Child-process inspection was unavailable under the local sandbox, so the returned monitor scope was process_only_children_unavailable. The example's elapsed field covers its model load/forward comparison after the wrapper had already prepared/imported the fixture runtime; it is not a cold-start or training-speed measurement. No weights were saved by the comparison script; fixture preparation itself creates owned private files outside Git.

Reproduce using the installed optional CPU engine and a fresh owned Llama fixture:

```sh
python -c 'from modelforge.runtime_hf import create_owned_fixture; create_owned_fixture("/private/tmp/my-owned-llama", model_type="llama", seed=20261021)'
python examples/decoder_gauge_control.py /private/tmp/my-owned-llama --output /private/tmp/new-gauge-report.json
```

The report destination must not exist. Keep fixture weights and tokens/teacher data outside public research exports. The script performs no fitting and does not save transformed weights. External provider access, model parity benchmarks and universal reconstruction remain unresolved.

Independent review of the later guarded source used a newly generated teacher with4 query heads/2 KV heads and verified the identity in actual grouped-query attention. Maximum observed logit difference0 and exact numeric value equality held on64×8 input positions. A separately created biased fixture was rejected with exit2 before forward/scaling, with no evidence output or file changes. Public-safe review records are decoder-gauge-independent.json and decoder-bias-guard-independent.json. A general matrix-valued Q/K gauge cannot automatically be carried through RoPE: arbitrary matrices need not commute with relative rotations; the demonstrated global scalar does.

Review also corrected the metric name: torch.equal checks exact tensor values, not byte identity (signed zeros can differ in bits). The source now calls this finite_logit_exact_value_equality. Historical reports preserve their original executed source hashes and explicitly record this naming-only postprocessing; no decoder rerun or numeric result change was needed.
