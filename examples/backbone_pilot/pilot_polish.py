# ruff: noqa
# PILOT (2026-10-22): frozen-consensus refit + duplicate replacement + unfrozen polish for truth-free seed-consensus gates. Not sealed evidence.
import sys, time, json, torch
exec(open("pilot_common.py").read())
torch.set_num_threads(3)
init, width, tlimit = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
model, cfg = make_teacher(init, 1); obs, _ = observe(model, 512)
tr = truth_params(model.model.layers[0])
x, a, m = obs[("x", 0)], obs[("a", 0)], obs[("m", 0)]
Z = unit(x + a).reshape(-1, 32); Mt = m.reshape(-1, 32)
def fit(G0, U0, tl, frozen=0):
    Gf = G0[:frozen].clone()
    G, U = G0[frozen:].clone().requires_grad_(), U0.clone().requires_grad_()
    Gall = lambda: torch.cat([Gf, G])
    def loss():
        F = torch.nn.functional.silu(Z @ Gall().T) * (Z @ U.T); D = torch.linalg.lstsq(F, Mt).solution.T
        return ((F @ D.T - Mt) ** 2).mean() / Mt.pow(2).mean()
    opt = torch.optim.LBFGS([G, U], lr=1, max_iter=200, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=100, line_search_fn="strong_wolfe")
    t0 = time.time()
    while time.time() - t0 < tl:
        opt.step(lambda: (opt.zero_grad(), (l := loss()).backward(), l)[2])
    return Gall().detach(), U.detach(), float(loss())
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
G, U, l = fit(G0, U0, tlimit, frozen=len(cons))
for rnd_i in range(2):                               # truth-free duplicate replacement
    dup = []
    for j in range(64):
        if any(float((G[j] - G[q]).norm() / G[j].norm()) < 0.05 for q in range(j)): dup.append(j)
    if not dup: break
    keepm = [j for j in range(64) if j not in dup]
    G0 = torch.cat([G[keepm], rnd(200 + rnd_i, len(dup))]); U0 = torch.cat([U[keepm], rnd(300 + rnd_i, len(dup))])
    G, U, l = fit(G0, U0, tlimit / 2, frozen=len(keepm))
G, U, l = fit(G, U, tlimit / 2, frozen=0)            # unfrozen polish
with torch.no_grad():
    F = torch.nn.functional.silu(Z @ G.T) * (Z @ U.T); D = torch.linalg.lstsq(F, Mt).solution.T
Gt = tr["G"]; d = torch.cdist(G, Gt) / Gt.norm(dim=1)[None, :]
match = d.argmin(1); relmin = d.min(1).values
bij = len(set(match.tolist())) == 64
W = lambda Dm, Um, u: torch.outer(Dm[:, u], Um[u])
wrel = [float((W(D, U, u) - W(tr["D"], tr["U"], int(match[u]))).abs().max() / W(tr["D"], tr["U"], int(match[u])).abs().max()) for u in range(64)] if bij else None
print(json.dumps({"consensus_seed_gates": len(cons), "final_rel_loss": l, "bijective_matching": bij,
                  "gates_within_1pct": int((relmin < 0.01).sum()), "gates_within_5pct": int((relmin < 0.05).sum()),
                  "max_rel_gate_error": float(relmin.max()),
                  "down_up_outer_max_rel_err": max(wrel) if wrel else None,
                  "down_up_outer_median_rel_err": sorted(wrel)[32] if wrel else None}))
