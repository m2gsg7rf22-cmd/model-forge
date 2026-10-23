# ruff: noqa
"""Sealed SwiGLU identification run: prepare (private teacher) -> solve (separate process, z/m only) -> verify."""
import hashlib, json, os, secrets, subprocess, sys, torch
from pathlib import Path
exec(open(Path(__file__).with_name("pilot_common.py")).read())
stage, private, public = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
if stage == "prepare":
    private.mkdir(parents=True, exist_ok=False); public.mkdir(parents=True, exist_ok=False)
    seed = secrets.randbits(31); (private / "teacher_seed.json").write_text(json.dumps({"seed": seed}))
    model, cfg = make_teacher(0.2, seed); obs, _ = observe(model, 512)
    Z = unit(obs[("x", 0)] + obs[("a", 0)]).reshape(-1, 32); Y = obs[("m", 0)].reshape(-1, 32)
    torch.save({"z": Z, "m": Y}, public / "observations.pt")
    print(json.dumps({"observations_sha256": hashlib.sha256((public / "observations.pt").read_bytes()).hexdigest()}))
elif stage == "solve":   # no private path is passed to this stage
    from modelforge import swiglu_identification as si
    torch.set_num_threads(3)
    obs = torch.load(public / "observations.pt")
    out = si.identify(obs["z"], obs["m"], 64, over_width=192, seeds=(11, 23, 37), seconds=600)
    torch.save({"gate": out["result"].gate}, public / "solution.pt")
    (public / "solve.json").write_text(json.dumps({"consensus": out["consensus_count"], "over_losses": out["overparameterized_losses"], "final_rel_loss": out["result"].relative_loss,
        "solution_sha256": hashlib.sha256((public / "solution.pt").read_bytes()).hexdigest()}))
elif stage == "verify":
    from modelforge import swiglu_identification as si
    seed = json.loads((private / "teacher_seed.json").read_text())["seed"]
    model, cfg = make_teacher(0.2, seed); tr = truth_params(model.model.layers[0])
    sol = torch.load(public / "solution.pt")
    rep = {**json.loads((public / "solve.json").read_text()), **si.match_gates(sol["gate"], tr["G"])}
    (public / "verify.json").write_text(json.dumps(rep)); print(json.dumps(rep))
