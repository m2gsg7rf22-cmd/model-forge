# Weighted candidate gate-gradient and local Gram: synthetic stage

This primitive differentiates the KNOWN candidate computation. It does not query the target, read target weights, or assume target-gradient access. No saved-data fitting or candidate update has been performed. The previous four-sweep result remains quality-failed and did not prove stationarity or convergence.

For f(x)=sum_i silu(g_i.T x)*(C_i x), physical unweighted targets Y, and fixed positive residual multipliers w, define E_p=w_p*(f(x_p)-Y_p) and L=sum_p ||E_p||². There is NO1/2factor. With t_pi=g_i.T x_p and v_pi=C_i x_p, the analytic real-arithmetic gradients are

`dL/dg_i = 2 sum_p w_p*silu'(t_pi)*<E_p,v_pi>*x_p`,

`dL/dC_i = 2 sum_p w_p*silu(t_pi)*E_p*x_p.T`.

The implementation uses silu'(t)=sigmoid(t)+t*sigmoid(t)*(1-sigmoid(t)), with a tanh-based stable sigmoid for FP64 candidate arithmetic. Weights are fixed from references, not differentiated. They enter E once; the derivative has a second multiplier through the chain rule. Doubling w multiplies loss AND both gradients by four. The C gradient is unconstrained: directly applying it generally breaks rank-one couplings. Computing gradients does not mutate any input or establish a valid reconstruction update.

For one gate and perturbation delta, the weighted candidate Jacobian action is

`J_i(delta)[p,o] = w_p*silu'(t_pi)*v_pi[o]*(x_p.T delta)`.

Its input-space Gram is

`K_i = sum_p (w_p*silu'(t_pi))²*||v_pi||²*x_p*x_p.T`.

This computes a d-by-d local matrix without allocating the full probe-output-by-gate Jacobian. For squared loss, J_i.T E equals half the gate gradient. A positive-definite Gram could define a local linearized Gauss-Newton direction, but this primitive performs NO inversion, ridge, line search, optimizer loop or update. A linearized minimizer is not a nonlinear descent or generalization guarantee. Any use must separately define conditioning, trust/step bounds, finite rejection and calibration guards before real fitting.

Contracts: finite FP64 arrays; at most64units,32input dimensions,32outputs and4096probes. Targets must be physical unweighted responses, not preweighted arrays. The API cannot infer from values whether a caller already weighted targets, so a real protocol must preserve raw observations and prove the weighting path from its manifest. Float64 storage or derivative arithmetic says nothing about the original target-computation precision. Analytic smooth-function gradients are not derivatives of every rounded bit pattern near saturation; finite checks reject nonfinite arithmetic.

Eight NEW focused synthetic tests compare physical forward output to an independent vectorized sigmoid implementation; all gate and coupling coordinates and a random joint directional derivative to independent centered differences; squared multiplier scaling; zero residual and input nonmutation; invalid shape/precision/weights; gate JVP to a directional difference; and the Gram to an explicit independent finite-difference design. Strict mypy and Ruff passed on the new source. No experiment arrays, consumed validation, target query or private model asset was accessed. Independent review is required before treating these controls as verified evidence.

[Primary isolated GLU research](https://arxiv.org/html/2608.06631v1), sections3.5and4.3, motivates separating direction changes and saved-response optimization. The formulas above are chain-rule candidate derivatives, not a new general model-extraction theorem or evidence that gate error caused our failed quality criterion.

Next: a DISTINCT prospective bounded gate-only update protocol must define full gate-vector parameterization, local conditioning/step budget, fixed rank-one couplings, training versus adaptive calibration guards, immutable initialization/source/data hashes, one-attempt semantics, process limits and new final PUBLIC941032points before fitting. Existing941027..941031sets stay consumed. Lower training loss is not final quality success. Universal/full-model reconstruction remains open and the private final-model gate CLOSED.
