# Owned decoder backbone identification protocol

Status: RESEARCH IN PROGRESS. Main-platform owned-teacher research only. The private final-model gate stays CLOSED.

This document fixes, before any sealed fitting, what is observed, what is claimed identifiable, which parameter
transformations are treated as equivalent ("gauges"), and how success is decided. It generalizes the final-head
experiment (docs/representation-recovery.md) and the scalar Q/K witness (docs/decoder-identifiability.md) to every
block of a project-owned Hugging Face Llama decoder.

## Teacher family

Bias-free Llama decoder: token embedding E; per layer l an RMSNorm (gain g1), grouped-query self-attention
(Q, K, V, O; H query heads, KV key/value heads, head size d, RoPE with HF `rotate_half` pairing (i, i+d/2)),
an RMSNorm (gain g2) and a SwiGLU MLP `D(silu(G z) * U z)`; a final RMSNorm (gain gf) and an untied linear head W.
All weights are project-owned, generated locally. No downloaded, provider or closed weights are used.

## Observation ladder

| Level | Exposed to the learner | Status |
| --- | --- | --- |
| O3 sublayer | input token ids, residual input x_l of every block, attention sublayer output a_l, MLP sublayer output m_l, final normed hidden state, full logits | studied here |
| O2 block | HF `hidden_states` per block (a_l and m_l only through their sum), full logits | not studied yet |
| O1 logits | full logits for chosen token inputs | not studied yet |
| O0 text | sampled or argmax tokens only | documented insufficient for scale (argmax control in the head experiment) |

O3 is far stronger than any public text API. A positive O3 result says nothing about extracting closed models
through text endpoints; it establishes what is recoverable when internals are legitimately exposed, and it supplies
the exact gauge bookkeeping needed before weaker levels can even be scored.

## Equivalence (gauge) group under O3

Each transformation below changes weights but leaves every O3 observation unchanged in exact arithmetic, so no
amount of O3 data can choose between the two parameterizations. Original weights are therefore never claimed;
the learner reports one representative and the verifier compares gauge-invariant quantities.

1. Norm gain absorption: `Q (g1 * z) = (Q diag g1) z`; likewise K, V, G, U absorb their norm gains. Convention:
   learner norm gains are 1; compared objects are the absorbed products.
2. Head relabeling: KV groups may be permuted together with their query heads; query heads may be permuted inside a
   group.
3. RoPE-commuting Q/K gauge: for every head-group and frequency pair, `q_pair -> A q_pair`, `k_pair -> A^{-T} k_pair`
   with `A = c Rot(phi)`, `c != 0`. Only matrices commuting with every relative rotation survive; a general matrix
   gauge does not (docs/decoder-identifiability.md). With GQA the same A must act on all query heads of the group.
4. Value/output gauge: per KV group, `V_g -> A V_g`, `O_h -> O_h A^{-1}` for its query heads, any invertible A.
5. SwiGLU gauge: permutation of hidden units, and `U_u -> c U_u`, `D_:,u -> D_:,u / c`. The gate row G_u has no
   scale gauge because silu is not homogeneous.
6. Unreached subspace: if the normalized inputs z of a sublayer span only an r-dimensional subspace S, every weight's
   action on the orthogonal complement of S is never exercised. It is unidentifiable and functionally irrelevant on
   the observed input domain. Example: with vocabulary 16 and hidden width 32, block 0 sees at most 16 directions.

Compared invariants, all restricted to the reached subspace S with orthonormal basis B:
`OV_h = O_h V_g B`, `QK_h(r) = B^T Q_h^T R_r K_g B` for the relative offsets used, gate rows `G B` after unit
matching, rank-one terms `D_:,u (U_u B)` after matching, embeddings for observed tokens, and the final head.

## Method under study

Each sublayer is identified from its own exposed input/output pairs, after exact RMS normalization of the
observed residual input. The output projection (O or D) enters linearly and is eliminated in closed form
(variable projection: least squares inside the objective); the remaining nonlinear weights are fitted by
float64 L-BFGS from a fixed list of seeded starts, choosing the start with the lowest calibration loss only.
The reconstructed student is then assembled into a fresh decoder and run end to end.

## Success criteria (to be frozen in the sealed preregistration)

Functional: heldout end-to-end logits of the assembled student versus the private teacher, on inputs never used for
fitting. Parametric: maximum relative error of every invariant above after head/unit matching performed by the
verifier. A sublayer with low loss but mismatched invariants is reported as "functionally fitted, not identified".
Thresholds come from the pilot below and are fixed before the sealed teacher is generated; they are never relaxed
after the sealed evaluation.

## Pilot (not sealed evidence)

Pilot teachers are separate seeds in /private/tmp, never reused for sealed evaluation. See the pilot section of
docs/RESEARCH_LOG.md (2026-10-22) and experiments/evidence/backbone-pilot/summary.json. Summary: attention invariants approximately identified at weight scale 0.2 but not at 0.02; exact-width SwiGLU not identified by VarPro-L-BFGS or Levenberg-Marquardt; 3x overparameterized SwiGLU is functionally equivalent on heldout (relative loss 8.5e-8). Pilot cost ~870 s per teacher, above budget.
