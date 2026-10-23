# Truth-free SwiGLU gate identification (research in progress)

Setting: O3 observations of an owned Llama block (docs/decoder-backbone-protocol.md): the MLP sublayer's normalized
input z and output m are exposed. Unknown: `m = D (silu(G z) * U z)` with width 64. Gate rows G have no scale gauge;
U and D are identified only as rank-one products D[:,u] U[u]; units are identified up to permutation.

## Findings so far (pilot teachers, not sealed)

1. Exact-width fitting (VarPro + L-BFGS, Levenberg–Marquardt) stalls in spurious minima (relative loss 1e-3).
2. 3× overparameterized fits reach 1e-6–1e-8 relative loss and generalize to heldout inputs (functional
   reconstruction), and they contain many true gate rows: 38–53 of 64 within 5%.
3. Seed consensus (gate rows reproduced by an independent overparameterized fit) selects true gates with precision
   1.0 in the measured run (47/47), without any teacher access.
4. Freezing consensus gates and refitting at exact width: teacher 1 reached 63/64 gates within 5% with a bijective
   match; the unrecovered unit carried 0.64% of unit output energy. Teacher 2 reached only 41/64.
5. Three seeds instead of two raised teacher 2 from 41/64 to 53/64 within 5%.
6. Teacher 3 with three seeds: 52/64 within 5%. Typical identification across three teachers ≈ 80%.
7. Duplicate replacement and unfrozen polish did not improve over the frozen refit.

## Interpretation

Overparameterization turns a hard non-convex identification problem into an easier functional fit whose solutions
retain identifiable structure; cross-seed agreement separates identifiable directions from arbitrary ones.
Low-energy units appear statistically weakly identifiable. Completeness is teacher- and budget-dependent.

## Required before any claim

Preregistered sealed protocol with fresh teachers, several weight scales, fixed fit budgets, verifier-only teacher
access, U/D rank-one invariant checks, and extension from one sublayer to whole blocks with attention.

Engine: `src/modelforge/swiglu_identification.py`. Pilot code and evidence: `examples/backbone_pilot/`,
`experiments/evidence/backbone-pilot/`.
