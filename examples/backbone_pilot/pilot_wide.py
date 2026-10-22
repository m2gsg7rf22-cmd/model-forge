# ruff: noqa
# PILOT record (2026-10-22), not sealed evidence; run from this directory.
import sys, time, json, torch
exec(open("pilot_common.py").read())
init, width, tlimit = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
model, cfg = make_teacher(init, 1); obs, _ = observe(model, 512)
x, a, m = obs[("x", 0)], obs[("a", 0)], obs[("m", 0)]
Z = unit(x + a).reshape(-1, 32); Mt = m.reshape(-1, 32)
cal, held = slice(0, 256 * 8), slice(256 * 8, None)
Zc, Mc, Zh, Mh = Z[cal], Mt[cal], Z[held], Mt[held]
gen = torch.Generator().manual_seed(7)
G, U = ((0.3 * torch.randn((width, 32), generator=gen, dtype=torch.float64)).requires_grad_() for _ in range(2))
def feats(Zx): return torch.nn.functional.silu(Zx @ G.T) * (Zx @ U.T)
def loss():
    F = feats(Zc); D = torch.linalg.lstsq(F, Mc).solution.T
    return ((F @ D.T - Mc) ** 2).mean() / Mc.pow(2).mean()
opt = torch.optim.LBFGS([G, U], lr=1, max_iter=200, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=100, line_search_fn="strong_wolfe")
t0 = time.time(); hist = []
while time.time() - t0 < tlimit:
    def closure():
        opt.zero_grad(); l = loss(); l.backward(); return l
    opt.step(closure); hist.append(float(loss()))
with torch.no_grad():
    F = feats(Zc); D = torch.linalg.lstsq(F, Mc).solution.T
    hl = float(((feats(Zh) @ D.T - Mh) ** 2).mean() / Mh.pow(2).mean())
print(json.dumps({"width": width, "cal_rel_loss": hist[-1], "heldout_rel_loss": hl, "heldout_max_abs": float((feats(Zh) @ D.T - Mh).abs().max()), "m_rms": float(Mh.pow(2).mean().sqrt()), "seconds": time.time() - t0, "trace": hist[::max(1, len(hist)//6)]}))
