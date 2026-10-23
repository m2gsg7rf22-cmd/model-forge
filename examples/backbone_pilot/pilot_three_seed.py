# ruff: noqa
# PILOT (2026-10-23): three-seed consensus on teacher 2 (where two-seed consensus gave 43 gates). Not sealed.
import sys, json, torch
exec(open("pilot_common.py").read())
from modelforge import swiglu_identification as si
torch.set_num_threads(3)
teacher_seed, seconds = int(sys.argv[1]), float(sys.argv[2])
model, cfg = make_teacher(0.2, teacher_seed); obs, _ = observe(model, 512)
tr = truth_params(model.model.layers[0])
Z = unit(obs[("x", 0)] + obs[("a", 0)]).reshape(-1, 32); Y = obs[("m", 0)].reshape(-1, 32)
fits = [si.fit_swiglu(Z, Y, 192, seed=s, seconds=seconds) for s in (11, 23, 37)]
pool = torch.cat([fits[1].gate, fits[2].gate])
cons = si.consensus_gates(fits[0].gate, pool)[:64]
final = si.fit_swiglu(Z, Y, 64, seed=1011, seconds=seconds, frozen_gate=cons)
rep = si.match_gates(final.gate, tr["G"])
print(json.dumps({"teacher_seed": teacher_seed, "seeds": 3, "consensus": len(cons), "over_losses": [f.relative_loss for f in fits], "final_rel_loss": final.relative_loss, **rep}))
