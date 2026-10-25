# Weighted forward gate-gradient primitive — DRAFT, not implemented

The bounded fixed-gate study improved finite paired relative errors but still failed the fixed .001quality criterion. Four sweeps reached the cap rather than proving stationarity. Guard vetoes and remaining coupling progress prevent attributing the residual to gate error. Do not append unregistered sweeps or relax the quality criterion.

Next investigate a different degree of freedom: analytically differentiate the KNOWN candidate's weighted forward objective with respect to its gates. This requires neither target gradients nor target weights. First implement a separate tiny synthetic primitive and independently review it; no actual saved-data fitting or new queries in this phase. Freeze the prior forward-factor and convergence implementations.

For candidate `f(x)=sum_i silu(g_i.T x)*(C_i x)`, physical observed targets Y and fixed positive residual multipliers w, define E_p=w_p*(f(x_p)-Y_p) and L=sum_p ||E_p||². The cost has no1/2factor, matching prior complete squared costs. With t_pi=g_i.T x_p and v_pi=C_i x_p, proposed gradients are

`dL/dg_i = 2 sum_p w_p*silu'(t_pi)*<E_p,v_pi>*x_p`,

`dL/dC_i = 2 sum_p w_p*silu(t_pi)*E_p*x_p.T`.

Multipliers are fixed from the calibration references, not differentiated. Targets enter as physical responses and are weighted once when constructing E. The C gradient is an unconstrained diagnostic; applying it directly would not preserve rank-one couplings. Gate-gradient computation alone does not change C, guarantee descent, identify original weights or prove gate error is dominant.

Verify the entire candidate forward against an independent implementation and central finite differences for gate, coupling and random joint directional derivatives. Include nonuniform weights, zero residual, shape/nonfinite bounds, and the exact factor-of-two convention. Small synthetic dimensions only, no consumed validation or private assets. Test changed source/types/lint; independent verifier must challenge sign/index/weighting and numerical assumptions.

Any later actual update method requires a DISTINCT prospective protocol defining parameterization (gate magnitudes/directions/orientations), descent/step budgets, training versus adaptive calibration guards, rank-one constraints, initialization/source/data hashes, one-attempt semantics, resource bounds and new independent final public points. Analytic candidate derivatives do not provide privileged target derivatives. Existing941027..941031evaluations remain consumed once used. Universal/full-model reconstruction and the private final-model prerequisite remain open/closed respectively.
