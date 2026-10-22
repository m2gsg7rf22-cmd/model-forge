# ruff: noqa
# PILOT record (2026-10-22), not sealed evidence; run from this directory.
"""PILOT v2: variable projection + reachable-subspace invariants. Not sealed evidence."""
import sys, time, json, math, itertools
import torch
from transformers import LlamaConfig, LlamaForCausalLM
exec(open("pilot_common.py").read())
init = float(sys.argv[1]); seed = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model, cfg = make_teacher(init, seed)
obs, tokens = observe(model, N)
rep = {"init": init, "seed": seed, "N": N}
t0 = time.time()
for i, layer in enumerate(model.model.layers):
    truth = truth_params(layer)
    x, a, m = obs[("x", i)], obs[("a", i)], obs[("m", i)]
    z1, z2 = unit(x), unit(x + a)
    r1 = int(torch.linalg.matrix_rank(z1.reshape(-1, 32))); r2 = int(torch.linalg.matrix_rank(z2.reshape(-1, 32)))
    t1 = time.time(); la, est_a = fit_attn(z1, a); ta = time.time() - t1
    t1 = time.time(); lm, est_m = fit_mlp(z2, m); tm = time.time() - t1
    rep[f"layer{i}"] = {"rank_z_attn": r1, "rank_z_mlp": r2, "attn_rel_loss": la, "mlp_rel_loss": lm,
        "attn_seconds": ta, "mlp_seconds": tm,
        **compare_attn(est_a, truth, z1), **compare_mlp(est_m, truth, z2)}
rep["seconds"] = time.time() - t0
print(json.dumps(rep, indent=1))
