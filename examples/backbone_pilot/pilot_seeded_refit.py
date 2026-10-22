# ruff: noqa
# PILOT (2026-10-22): exact-width SwiGLU refit seeded by truth-free seed-consensus gates. Not sealed evidence.
import sys, time, json, torch
exec(open("pilot_common.py").read())
torch.set_num_threads(3)
init, width, tlimit = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
model, cfg = make_teacher(init, 1); obs, _ = observe(model, 512)
tr = truth_params(model.model.layers[0])
x, a, m = obs[("x", 0)], obs[("a", 0)], obs[("m", 0)]
Z = unit(x + a).reshape(-1, 32); Mt = m.reshape(-1, 32)
def fit(G0, U0, tl):
    G, U = G0.clone().requires_grad_(), U0.clone().requires_grad_()
    def loss():
        F = torch.nn.functional.silu(Z @ G.T) * (Z @ U.T); D = torch.linalg.lstsq(F, Mt).solution.T
        return ((F @ D.T - Mt) ** 2).mean() / Mt.pow(2).mean()
    opt = torch.optim.LBFGS([G, U], lr=1, max_iter=200, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=100, line_search_fn="strong_wolfe")
    t0 = time.time()
    while time.time() - t0 < tl:
        opt.step(lambda: (opt.zero_grad(), (l := loss()).backward(), l)[2])
    return G.detach(), U.detach(), float(loss())
rnd = lambda s, n: 0.3 * torch.randn((n, 32), generator=torch.Generator().manual_seed(s), dtype=torch.float64)
GA, UA, _ = fit(rnd(11, width), rnd(12, width), tlimit); GB, _, _ = fit(rnd(23, width), rnd(24, width), tlimit)
rel = torch.cdist(GA, GB) / GA.norm(dim=1, keepdim=True)
cons = GA[rel.min(1).values < 0.05]; consU = UA[rel.min(1).values < 0.05]
keep = []                                            # dedupe consensus units (truth-free)
for j in range(len(cons)):
    if all(float((cons[j] - cons[q]).norm() / cons[j].norm()) > 0.05 for q in keep): keep.append(j)
cons, consU = cons[keep], consU[keep]
n_free = 64 - len(cons)
G0 = torch.cat([cons, rnd(99, n_free)]); U0 = torch.cat([consU, rnd(98, n_free)])
G, U, l = fit(G0, U0, tlimit)
Gt = tr["G"]; d = torch.cdist(G, Gt) / Gt.norm(dim=1)[None, :]
match = d.argmin(1); relmin = d.min(1).values
bij = len(set(match.tolist())) == 64
print(json.dumps({"consensus_seed_gates": len(cons), "free_units": n_free, "exact_width_rel_loss": l,
                  "units_within_1pct": int((relmin < 0.01).sum()), "units_within_5pct": int((relmin < 0.05).sum()),
                  "bijective_matching": bij, "max_rel_gate_error": float(relmin.max())}))
