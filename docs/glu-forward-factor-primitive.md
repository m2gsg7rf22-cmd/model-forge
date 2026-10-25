# Weighted full-forward factor primitive: synthetic stage

This is a separately implemented weighted forward-response adaptation of the rank-one whitening method, motivated by the independently audited saved-Hessian quality failure. It has not been fitted to an actual FP32 experiment and does not establish a better reconstruction candidate. It keeps the entire gate vector fixed, including magnitude and orientation.

For a fixed gate g, coupling C[outputs,input], inputs x_p, and positive residual multipliers w_p, define

`Phi[p,:] = w_p * silu(g.T x_p) * x_p.T`.

The unit's WEIGHTED output is `L(C)=Phi C.T`. Every response or partial residual passed to `adjoint` or `rankone_update` is already weighted ONCE: `Y[p,:]=w_p*physical_partial[p,:]`. The code does not multiply Y again. The squared objective therefore has loss weights w_p², not w_p. Multiplying raw FP32 responses in FP64 creates derived fitting arrays; it does not turn the physical target computation into FP64.

For Frobenius inner products, `L*(Y)=Y.T Phi` and `K=Phi.T Phi`. With T=L*(Y), positive-definite K and M=K^(-1/2), set Z=C K^(1/2). Then

`||Y-Phi C.T||² = ||Y||² - ||TM||² + ||Z-TM||²`.

Right multiplication by an invertible matrix preserves rank. A best rank-at-most-one approximation of TM is its leading singular triplet. Transform it back by M to obtain a local quadratic minimizer C. This is an exact-arithmetic statement. The FP64 implementation provides no floating-point optimality certificate near the conditioning guard, no guarantee of descent for another objective, and no unseen-data accuracy guarantee. Zero residuals may yield rank zero; future actual candidate validation must independently enforce nonzero-unit constraints.

The API accepts finite FP64 g[d<=32], x[1..4096,d], w[probes] strictly positive, and up to32outputs. It rejects nonpositive, nonfinite, wrongly shaped or non-FP64 multipliers. Whitening requires the smallest Gram eigenvalue to exceed1e-12times the largest, and the largest to be positive. No ridge or pseudoinverse silently changes the objective. APIs operate on derived weighted responses, not a newly advertised physical observation profile.

Six NEW synthetic tests cover independent sigmoid-based forward values; adjoint/Gram/inner products; exact rank-one and zero-residual recovery; a general solution in independent Cholesky coordinates; the equivalence of w=sqrt(integer counts) to physically repeated observations; and invalid inputs/singular whitening. Strict mypy and Ruff passed on the new source. No existing experiment arrays, original teacher weights, provider API or consumed evaluation files were read. No saved-candidate fit or native build occurred. Independent source/numerical review is required before using this evidence to design an actual experiment.

[Primary isolated GLU research](https://arxiv.org/html/2608.06631v1), section3.5, uses whitening for fixed-gate Hessian coupling updates; section4.3 uses a saved forward-error objective and separate guard pairs. The particular weighted full-forward local solver above is an algebraic adaptation, not a claim that the paper tested this ModelForge implementation or guarantees its success.

Next define a distinct protocol before any real collection or fitting: authorized new owned FP32 calibration, scale-balanced training objective, separate calibration guard data, original pre-Hessian-polish starting candidate, deterministic unit ranking/retention/sweep budget, complete proposal storage, process limits and a disjoint new public final evaluation. A guard used to select updates remains calibration. Public seeds941027/941028/941029 are consumed; a future941030evaluation must be prospectively committed and used once. Original quality failures, production FP64 contract and private final-model gate remain unchanged. Universal/full-model reconstruction remains open.
