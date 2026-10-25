# Synthetic guard-aware direction prerequisite

This component solves a small homogeneous cone-constrained local quadratic model of a KNOWN candidate. It has no real-data guard adapter, target-gradient access, saved-candidate fit or target query. It is not evidence that the prior quality failures have been repaired. Private final-model gate CLOSED.

## Mathematical scope and completeness in exact arithmetic

Minimize q(z)=d.T z+z.T K z subject to A z<=0, where K is positive definite, dimension1..32and0..8nonzero normals. The no-half convention gives stationarity2Kz+d+A.T lambda=0. Zero is feasible. Strict convexity gives a unique minimizer.

For a polyhedral cone, the negative objective gradient at the minimizer belongs to the conic hull of its active normals, including cones with empty interior. Choose a nonnegative representation with minimum support. If those normals were linearly dependent, a nonzero dependence permits moving the coefficients in one sign until a positive coefficient vanishes without becoming negative; this contradicts minimum support. Thus an independent active support exists, appears among the at most256subsets, and its equality KKT solution with nonnegative multipliers satisfies the full KKT conditions. Every candidate is checked against ALL original inequalities in normalized coordinates, not merely its active subset.

The basis is standard constrained convex optimization, not a new model reconstruction theorem. [Boyd and Vandenberghe, official Stanford book](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), optimality/KKT and polyhedral quadratic problems. ModelForge's eventual mapping of calibration derivatives to constraints remains prospective and unverified by this source.

## Numerical domain and explicit checks

Inputs are finite exactdtypefloat64; K must be exactly symmetric with finite eigenvalues min>1e-12max>0. Gate norm and row norms must be finite/nonzero. Normalize beta=maxeig(K), K'=K/beta,d'=d/beta and B_i=A_i/rowNorm_i. For each independent support I solve the equality system [[2K',B_I.T],[B_I,0]] [z,mu_I]=[-d',0]. No ridge/pseudoinverse. Inactive multipliers are zero. Original multipliers are lambda_i=beta*mu_i/rowNorm_i; both sets and raw original-coordinate residuals are returned.

The independent-normal SVD geometry cutoff is1e-5relative (sqrt of the1e-10residual tolerance); the separate K eigenvalue ratio remains1e-12. This conservative cutoff is a domain policy, not a derived forward-error bound. A numerically near-dependent support of at most n rows triggers an exact rational rank check on the ORIGINAL binary64 coefficients via Fraction(float), not rounded normalized B. If exactly independent, the ENTIRE problem is rejected even if another subset already passed approximate KKT. Exactly dependent supports are recorded and skipped; supports larger than n cannot be independent. All subsets are visited, so a bad smaller basis cannot be hidden by a larger redundant set. This is a conservative domain restriction. Values produced by expressions such as3*a or row1+row2 may acquire rounding and become exactly independent despite intended mathematical dependence.

Every numerical KKT ratio must be<=1e-10. Scales are componentwise, with explicit1floors:

- Primal: max_i max(0,(Bz)_i)/max(1,abs((Bz)_i)).
- Dual: max_i max(0,-mu_i)/max(1,abs(mu_i)).
- Stationarity: max_j abs((2K'z+d'+B.Tmu)_j)/max(1,abs((2K'z)_j),abs(d'_j),abs((B.Tmu)_j)).
- Complementarity: max_i abs(mu_i*(Bz)_i)/max(1,abs(mu_i),abs((Bz)_i)).

Empty maxima equal0. Nonfinite derived arithmetic fails closed. Component scales and signed raw residuals are stored. An unrelated huge multiplier/direction coordinate cannot hide a significant negative multiplier or infeasible coordinate, as happened in rejected development versions. A norm overflowing to infinity must not hide stationarity error. These finite residual checks do NOT prove exact feasibility, a forward-error/optimality-gap bound or numerical completeness near rank/conditioning thresholds.

Choose the smallest normalized objective among checked candidates, with increasing-subset-mask ties. Reject if none. Return z scaled radially to at most.05||gate||. Exact positive downscaling preserves homogeneous feasibility; postclip numerical primal violation is rechecked. Report pre/post objectives/norms/scaling. The KKT checks concern the PRECLIP solution; postclip is not certified as a trust-region minimizer or KKT point. No later caller may infer nonlinear full-forward quality from the local quadratic score.

## Evidence transport prerequisite

examples/isolated_glu/evidence_transport.py writes exclusive-attempt numerical NPZ files and small hash-bound JSON summaries. It supports1..64finitefloat64arrays, total<=32MiB, bounded names excluding reserved NumPy keywords; compact metadata<=16KiBand summary<=32KiB. Symlink ancestors are rejected before output creation. Invalid attempts remain reserved, without automatic retry. This narrow helper is not an arbitrary huge-report serializer: future adapters must put large array/diagnostic content outside their compact metadata.

A NEW synthetic test saved4.8MBnumeric evidence and received its compact metadata through run_json with a2KiBoutput limit, without increasing engine/metadata parser limits. Evidence hash, reload and summary matched. It does not repair or rerun the previous completed fit: that controller remains failed and its original build RSS receipt remains unknown.

## Software and independent validation

18NEWfocused controls passed: unconstrainedhalf-factor, analytic boundary/duals, duplicate/opposite/empty-interiorcones, radialcap, row/objective rescaling/nonmutation, finite/shape/conditioning guards, all256subsets, overflow/hidden-negative-dual/hidden-primal counterexamples, near-independent rejection at1e-14/1e-11/1e-6, a separated1e-3blocked-cone control and an exact general3Ddependency; plus large-evidence transport/bounds/attempts/symlinks. Strict mypy on2modules and Ruff on4changed source/test files passed. No unrelated successful test or native package was rebuilt.

Independent implementation review and synthetic geometric probes are pending; no actual data fitting is authorized before separately reviewing how real guard derivatives, maxabs ties, inactive coordinates, current/initial bounds and objective weights map into this primitive. The original exact full-forward trial guards and.001quality criterion cannot be relaxed. Consumed PUBLIC941027..941032must not select new directions or be represented as fresh validation. Universal reconstruction remains open.
