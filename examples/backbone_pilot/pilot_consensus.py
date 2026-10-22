# ruff: noqa
# PILOT (2026-10-22): truth-free consensus of overparameterized SwiGLU gates across seeds. Not sealed evidence.
import sys, time, json, torch
exec(open("pilot_common.py").read())
torch.set_num_threads(2)
init, width, tlimit = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
model, cfg = make_teacher(init, 1); obs, _ = observe(model, 512)
tr = truth_params(model.model.layers[0])
x, a, m = obs[("x", 0)], obs[("a", 0)], obs[("m", 0)]
Z = unit(x + a).reshape(-1, 32); Mt = m.reshape(-1, 32)
def fit(seed):
    gen = torch.Generator().manual_seed(seed)
    G, U = ((0.3 * torch.randn((width, 32), generator=gen, dtype=torch.float64)).requires_grad_() for _ in range(2))
    def loss():
        F = torch.nn.functional.silu(Z @ G.T) * (Z @ U.T); D = torch.linalg.lstsq(F, Mt).solution.T
        return ((F @ D.T - Mt) ** 2).mean() / Mt.pow(2).mean()
    opt = torch.optim.LBFGS([G, U], lr=1, max_iter=200, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=100, line_search_fn="strong_wolfe")
    t0 = time.time()
    while time.time() - t0 < tlimit:
        opt.step(lambda: (opt.zero_grad(), (l := loss()).backward(), l)[2])
    return G.detach(), float(loss())
GA, la = fit(11); GB, lb = fit(23)
rel = torch.cdist(GA, GB) / GA.norm(dim=1, keepdim=True)
cons = rel.min(1).values < 0.05                     # truth-free: unit reproduced by an independent run
Gt = tr["G"]; relt = torch.cdist(GA, Gt).min(1).values / GA.norm(dim=1)
true_hit = relt < 0.05
k = int(cons.sum()); prec = int((cons & true_hit).sum())
cover = len(set(torch.cdist(GA[cons], Gt).argmin(1)[(relt[cons] < 0.05)].tolist())) if k else 0
print(json.dumps({"width": width, "loss_A": la, "loss_B": lb, "consensus_units": k, "consensus_units_that_are_true_gates": prec,
                  "precision": prec / k if k else None, "distinct_true_gates_covered": cover, "true_units": 64,
                  "true_hit_units_A_total": int(true_hit.sum())}))
