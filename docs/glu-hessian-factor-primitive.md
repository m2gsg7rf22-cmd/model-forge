# Fixed-gate Hessian factor primitive: synthetic stage

This research primitive implements the local rank-at-most-one coupling update suggested by section3.5 of [the isolated GLU paper](https://arxiv.org/html/2608.06631v1). It has not been applied to saved FP32 research Hessians and does not establish a better reconstruction candidate. Earlier quality failures remain unchanged.

For one fixed gate vector g and output-by-input coupling C, write t=gᵀx, α=silu''(t), β=silu'(t), and s=gᵀg. Each output Hessian is

`L_g(C)[p,o] = α_p (C_o x_p) ggᵀ + β_p (g C_oᵀ + C_o gᵀ)`.

All inner products below use the FULL Hessian Frobenius norm. Off-diagonal entries occur twice. An unweighted packed upper triangle would define a different objective. For any residual tensor Y, including nonsymmetric test controls,

`L_g*(Y)_o = Σ_p [α_p (gᵀY_po g)x_p + β_p(Y_po+Y_poᵀ)g]`.

The input-space Gram K=L_g*L_g does not mix output rows:

`K = Σ_p [α_p²s² x_p x_pᵀ + 2α_pβ_p s(x_p gᵀ+g x_pᵀ) + β_p²(2sI+2ggᵀ)]`.

This closed form avoids constructing the large probe/Hessian design matrix. For a partial residual Y after excluding this unit, let T=L_g*Y and M=K^(-1/2). When K is positive definite, set Z=C K^(1/2). Then

`||Y-L_g(C)||² = ||Y||² - ||TM||² + ||Z-TM||²`.

Invertible right multiplication preserves rank. The best rank-at-most-one approximation of TM is its leading singular triplet; transform that approximation back by M to obtain C. This is a global minimizer of ONE fixed-gate local quadratic objective. It is not a proof that a full-model objective or unseen forward accuracy improves. A zero residual may yield rank zero. Actual candidate retention and nonzero-factor guards belong to a future experiment protocol.

The implementation accepts finite FP64 arrays with d<=32, outputs<=32 and at most64probes. It rejects whitening when the largest Gram eigenvalue is nonpositive or the smallest is <=1e-12 times the largest. No ridge or pseudoinverse silently substitutes for this guard. Gates stay fixed, so this primitive cannot repair direction errors. Prediction arithmetic, target-computation precision and stored weight precision remain separate concepts.

Five NEW synthetic tests verify the Hessian against independent forward finite differences; adjoint and Gram against an explicitly constructed operator basis; exact rank-one recovery; a general noisy-residual solution against independent Cholesky coordinates; and finite/shape/singular-whitening guards. They do not read experiment arrays, teacher weights or consumed evaluation files. Strict mypy and Ruff pass on the new source. Independent review is required before treating these controls as verified evidence.

Next, define a distinct source-frozen saved-Hessian polish protocol before any actual fit: objective and scaling, partial-residual convention, fixed unit-ranking rule, number of updates/sweeps, complete-residual retention rule, numerical/resource limits, and a separate new public validation set. No native rebuild, provider query, private-model work or supported production FP32 profile is implied. Universal reconstruction and the private final-model prerequisite remain open/closed respectively.
