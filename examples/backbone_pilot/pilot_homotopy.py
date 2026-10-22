# ruff: noqa
# PILOT (2026-10-22): do overparameterized SwiGLU gate rows cluster at true gate rows? Not sealed evidence.
import sys, time, json, torch
exec(open("pilot_common.py").read())
torch.set_num_threads(2)
init, width, tlimit = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
model, cfg = make_teacher(init, 1); obs, _ = observe(model, 512)
tr = truth_params(model.model.layers[0])
x, a, m = obs[("x", 0)], obs[("a", 0)], obs[("m", 0)]
Z = unit(x + a).reshape(-1, 32); Mt = m.reshape(-1, 32)
gen = torch.Generator().manual_seed(7)
G, U = ((0.3 * torch.randn((width, 32), generator=gen, dtype=torch.float64)).requires_grad_() for _ in range(2))
def loss():
    F = torch.nn.functional.silu(Z @ G.T) * (Z @ U.T); D = torch.linalg.lstsq(F, Mt).solution.T
    return ((F @ D.T - Mt) ** 2).mean() / Mt.pow(2).mean()
opt = torch.optim.LBFGS([G, U], lr=1, max_iter=200, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=100, line_search_fn="strong_wolfe")
t0 = time.time()
while time.time() - t0 < tlimit:
    opt.step(lambda: (opt.zero_grad(), (l := loss()).backward(), l)[2])
with torch.no_grad():
    F = torch.nn.functional.silu(Z @ G.T) * (Z @ U.T); D = torch.linalg.lstsq(F, Mt).solution.T
    contrib = (F * F).mean(0).sqrt() * D.norm(dim=0)          # per-unit output energy
    Gt = tr["G"]
    dist = torch.cdist(G, Gt)                                   # [width, 64]
    nearest, which = dist.min(1)
    rel = nearest / Gt.norm(dim=1)[which]
    strong = contrib > contrib.max() * 1e-3
    covered = len(set(which[strong & (rel < 0.05)].tolist()))
    out = {"width": width, "rel_loss": float(loss()), "units_strong": int(strong.sum()),
           "strong_units_within_5pct_of_a_true_gate": int((strong & (rel < 0.05)).sum()),
           "true_gates_covered_within_5pct": covered, "true_units": 64,
           "median_rel_dist_strong": float(rel[strong].median()), "seconds": time.time() - t0}
print(json.dumps(out))
