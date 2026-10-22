# ruff: noqa
# PILOT record (2026-10-22), not sealed evidence; run from this directory.
import math, itertools, torch
from transformers import LlamaConfig, LlamaForCausalLM
torch.set_num_threads(4)
EPS = 1e-6
def make_teacher(init, seed):
    cfg = LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        initializer_range=init, rms_norm_eps=EPS, bos_token_id=2, eos_token_id=3, pad_token_id=0)
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n: p.copy_(1 + 0.3 * torch.randn_like(p))
    return model, cfg
def observe(model, N, T=8, seed=91):
    tokens = torch.randint(0, 16, (N, T), generator=torch.Generator().manual_seed(seed))
    obs, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, a, kw, i=i: obs.__setitem__(("x", i), (a[0] if a else kw["hidden_states"]).detach().double()), with_kwargs=True))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(lambda m, a, o, i=i: obs.__setitem__(("a", i), o.detach().double())))
        hooks.append(layer.mlp.register_forward_hook(lambda m, a, o, i=i: obs.__setitem__(("m", i), o.detach().double())))
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    for h in hooks: h.remove()
    obs["final_h"] = out.hidden_states[-1].double(); obs["logits"] = out.logits.double()
    return obs, tokens
def unit(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
def rope(x, theta=10000.0):
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d)
    f = torch.outer(torch.arange(x.shape[-2], dtype=torch.float64), inv)
    emb = torch.cat([f, f], -1); c, s = emb.cos(), emb.sin()
    return x * c + torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1) * s
def heads(z, Q, K, V, H=4, KV=2, d=8):
    n, t, _ = z.shape
    q = rope((z @ Q.T).view(n, t, H, d).transpose(1, 2))
    k = rope((z @ K.T).view(n, t, KV, d).transpose(1, 2)).repeat_interleave(H // KV, 1)
    v = (z @ V.T).view(n, t, KV, d).transpose(1, 2).repeat_interleave(H // KV, 1)
    s = (q @ k.transpose(-1, -2) / math.sqrt(d)).masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    return (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(n, t, H * d)
def truth_params(layer):
    sa, ml = layer.self_attn, layer.mlp
    g1 = layer.input_layernorm.weight.double(); g2 = layer.post_attention_layernorm.weight.double()
    return {"Q": sa.q_proj.weight.double() * g1, "K": sa.k_proj.weight.double() * g1, "V": sa.v_proj.weight.double() * g1,
            "O": sa.o_proj.weight.double(), "G": ml.gate_proj.weight.double() * g2, "U": ml.up_proj.weight.double() * g2,
            "D": ml.down_proj.weight.double()}
def varpro_solve(F, y):  # F [M,p], y [M,o] -> W [o,p]
    return torch.linalg.lstsq(F, y, driver="gelsd").solution.T
def _lbfgs(ps, loss_fn, rounds, iters):
    opt = torch.optim.LBFGS(ps, lr=1, max_iter=iters, tolerance_grad=1e-15, tolerance_change=1e-18, history_size=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = loss_fn(); l.backward(); return l
    for _ in range(rounds): opt.step(closure)
    with torch.no_grad(): return float(loss_fn())
def fit_attn(z, a, restarts=3, rounds=4, iters=300, scale=0.3):
    Z = z.reshape(-1, 32); A = a.reshape(-1, 32); nrm = A.pow(2).mean()
    best = None
    for r in range(restarts):
        gen = torch.Generator().manual_seed(2000 + r)
        Q, K, V = ((scale * torch.randn(s, generator=gen, dtype=torch.float64)).requires_grad_() for s in [(32, 32), (16, 32), (16, 32)])
        def loss():
            F = heads(z, Q, K, V).reshape(-1, 32)
            O = varpro_solve(F, A)
            return ((F @ O.T - A) ** 2).mean() / nrm
        l = _lbfgs([Q, K, V], loss, rounds, iters)
        if best is None or l < best[0]:
            with torch.no_grad():
                F = heads(z, Q, K, V).reshape(-1, 32); O = varpro_solve(F, A)
            best = (l, {"Q": Q.detach(), "K": K.detach(), "V": V.detach(), "O": O})
    return best
def fit_mlp(z, m, restarts=3, rounds=4, iters=300, scale=0.3):
    Z = z.reshape(-1, 32); Mt = m.reshape(-1, 32); nrm = Mt.pow(2).mean()
    best = None
    for r in range(restarts):
        gen = torch.Generator().manual_seed(3000 + r)
        G, U = ((scale * torch.randn((64, 32), generator=gen, dtype=torch.float64)).requires_grad_() for _ in range(2))
        def loss():
            F = torch.nn.functional.silu(Z @ G.T) * (Z @ U.T)
            D = varpro_solve(F, Mt)
            return ((F @ D.T - Mt) ** 2).mean() / nrm
        l = _lbfgs([G, U], loss, rounds, iters)
        if best is None or l < best[0]:
            with torch.no_grad():
                F = torch.nn.functional.silu(Z @ G.T) * (Z @ U.T); D = varpro_solve(F, Mt)
            best = (l, {"G": G.detach(), "U": U.detach(), "D": D})
    return best
def span_basis(z):
    Z = z.reshape(-1, 32); _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    r = int((S > S[0] * 1e-9).sum()); return Vh[:r].T  # [32,r]
def rel(a, b): return float((a - b).abs().max() / b.abs().max())
def compare_attn(est, tr, z):
    B = span_basis(z)
    def ov(p, h): return p["O"][:, h*8:(h+1)*8] @ p["V"][(h//2)*8:(h//2+1)*8] @ B
    def qk(p, h, r):
        d = 8; inv = 1.0 / 10000.0 ** (torch.arange(0, d, 2, dtype=torch.float64) / d)
        ang = r * inv; c = torch.cat([ang.cos(), ang.cos()]); s = torch.cat([ang.sin(), ang.sin()])
        R = torch.diag(c); half = d // 2
        R[:half, half:] += torch.diag(s[:half]); R[half:, :half] -= torch.diag(s[half:])  # R_r: q^T R k convention
        Qh = p["Q"][h*8:(h+1)*8] @ B; Kh = p["K"][(h//2)*8:(h//2+1)*8] @ B
        return Qh.T @ R @ Kh
    best = None
    for perm in itertools.permutations(range(4)):
        e = max(rel(ov(est, perm[h]), ov(tr, h)) for h in range(4))
        if best is None or e < best[0]: best = (e, perm)
    perm = best[1]
    gqa_consistent = all(perm[h] // 2 == perm[h ^ 1] // 2 for h in range(4))
    qk_err = max(rel(qk(est, perm[h], r), qk(tr, h, r)) for h in range(4) for r in range(8))
    return {"ov_rel_err_matched": best[0], "qk_rel_err_matched": qk_err, "head_perm": list(perm), "head_perm_gqa_consistent": gqa_consistent}
def compare_mlp(est, tr, z):
    B = span_basis(z)
    Ge, Gt = est["G"] @ B, tr["G"] @ B
    sim = torch.cdist(Ge, Gt); match = sim.argmin(1)
    bij = len(set(match.tolist())) == 64
    gate_err = rel(Ge, Gt[match])
    W = lambda p, u: torch.outer(p["D"][:, u], p["U"][u] @ B)
    w_err = max(rel(W(est, u), W(tr, int(match[u]))) for u in range(64)) if bij else None
    return {"mlp_rank_basis": B.shape[1], "gate_rel_err_matched": gate_err, "gate_matching_bijective": bij, "down_up_outer_rel_err_max": w_err}
