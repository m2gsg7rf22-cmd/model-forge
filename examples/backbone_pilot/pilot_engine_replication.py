# ruff: noqa
# PILOT (2026-10-23): replicate the seed-consensus pipeline with the packaged engine on new teacher seeds. Not sealed.
import sys, json, torch
exec(open("pilot_common.py").read())
from modelforge import swiglu_identification as si
torch.set_num_threads(3)
teacher_seed, seconds = int(sys.argv[1]), float(sys.argv[2])
model, cfg = make_teacher(0.2, teacher_seed); obs, _ = observe(model, 512)
tr = truth_params(model.model.layers[0])
Z = unit(obs[("x", 0)] + obs[("a", 0)]).reshape(-1, 32); Y = obs[("m", 0)].reshape(-1, 32)
out = si.identify(Z, Y, 64, over_width=192, seconds=seconds)
rep = si.match_gates(out["result"].gate, tr["G"])
print(json.dumps({"teacher_seed": teacher_seed, "consensus": out["consensus_count"], "over_losses": out["overparameterized_losses"], "final_rel_loss": out["result"].relative_loss, **rep}))
