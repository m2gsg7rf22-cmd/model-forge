# ruff: noqa
# PILOT record (2026-10-22), not sealed evidence; run from this directory.
"""PILOT: LM with analytic Jacobian on SwiGLU sublayer. Not sealed evidence."""
import sys, time, json, torch
exec(open("pilot_common.py").read())
init, seed, layer_i, restarts, tlimit, nsub = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
model, cfg = make_teacher(init, seed); obs, _ = observe(model, 256)
layer = model.model.layers[layer_i]; tr = truth_params(layer)
x, a, m = obs[("x", layer_i)], obs[("a", layer_i)], obs[("m", layer_i)]
Zall = unit(x + a).reshape(-1, 32); Mall = m.reshape(-1, 32)
idx = torch.randperm(len(Zall), generator=torch.Generator().manual_seed(5))[:nsub]
Z, Mt = Zall[idx], Mall[idx]; nrm = float(Mt.pow(2).mean())
sig = torch.sigmoid
def parts(G, U):
    g = Z @ G.T; s = torch.nn.functional.silu(g); u = Z @ U.T
    ds = sig(g) * (1 + g * (1 - sig(g)))
    return s, u, ds
def jac(G, U, D):
    s, u, ds = parts(G, U); n = len(Z)
    h = s * u                                  # [n,k]
    jD = torch.einsum("oa,nk->noak", torch.eye(32, dtype=torch.float64), h).reshape(n * 32, 2048)
    jG = torch.einsum("ok,nk,ni->noki", D, u * ds, Z).reshape(n * 32, 2048)
    jU = torch.einsum("ok,nk,ni->noki", D, s, Z).reshape(n * 32, 2048)
    return torch.cat([jG, jU, jD], 1)
def resid(th):
    G, U, D = th[:2048].view(64, 32), th[2048:4096].view(64, 32), th[4096:].view(32, 64)
    s, u, _ = parts(G, U); return ((s * u) @ D.T - Mt).reshape(-1)
def full_loss(th):
    G, U, D = th[:2048].view(64, 32), th[2048:4096].view(64, 32), th[4096:].view(32, 64)
    return float((((torch.nn.functional.silu(Zall @ G.T) * (Zall @ U.T)) @ D.T - Mall) ** 2).mean() / Mall.pow(2).mean())
for r in range(restarts):
    t0 = time.time(); gen = torch.Generator().manual_seed(4000 + r)
    G = 0.3 * torch.randn(64, 32, generator=gen, dtype=torch.float64); U = 0.3 * torch.randn(64, 32, generator=gen, dtype=torch.float64)
    s, u, _ = parts(G, U); D = torch.linalg.lstsq(s * u, Mt).solution.T
    th = torch.cat([G.reshape(-1), U.reshape(-1), D.reshape(-1)]); lam = 1e-2; it = 0
    f = resid(th); loss = float(f.pow(2).mean()) / nrm
    while time.time() - t0 < tlimit and loss > 1e-24:
        J = jac(th[:2048].view(64, 32), th[2048:4096].view(64, 32), th[4096:].view(32, 64))
        JtJ = J.T @ J; g = J.T @ f; dg = JtJ.diagonal().clone(); del J
        ok = False
        while lam < 1e14:
            step = torch.linalg.solve(JtJ + lam * torch.diag(dg + 1e-15), -g)
            fn = resid(th + step); ln = float(fn.pow(2).mean()) / nrm
            if ln < loss: th, f, loss, lam, ok = th + step, fn, ln, max(lam / 3, 1e-15), True; break
            lam *= 4
        it += 1
        if not ok: break
    G, U, D = th[:2048].view(64, 32), th[2048:4096].view(64, 32), th[4096:].view(32, 64)
    cmp = compare_mlp({"G": G, "U": U, "D": D}, tr, Zall.view(-1, 1, 32))
    print(json.dumps({"restart": r, "iters": it, "sub_rel_loss": loss, "full_rel_loss": full_loss(th), "seconds": time.time() - t0, **cmp}), flush=True)
