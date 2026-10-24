# Resume ModelForge

The exact user messages המשך, Continue, Resume ModelForge resume this mission. Read AGENTS.md, MASTER_MISSION.md, FINAL_PRIVATE_MODEL_MISSION.md, RESEARCH_ROADMAP.md, RESEARCH_LOG.md and .modelforge/research_state.json. Check git status/log; do not discard existing work. Stable source is the public repository; research branch discovery-lab.

Run from the checkout with Python3.12+ and installed package (or PYTHONPATH=src):

```sh
python3.12 scripts/resume_research.py --root "$PWD"
```

This utility validates state, ledger integrity and checkpoint ancestry, recovers interrupted jobs to a resumable queue and reports the next eligible tasks. It does not run command strings, start training, install dependencies or schedule future agents. A corrupt/torn ledger must be reported and preserved, not silently replaced. Recovery must not steal an active process's lock or mistake a reused PID for its original owner.

Use .modelforge/research_state.json as the exact work queue and ledger head. checkpoint_base_commit identifies the verified ancestor upon which the checkpoint was written; the state cannot truthfully contain its own enclosing commit SHA. Current HEAD is checked dynamically. Immutable mission digests are in .modelforge/mission_manifest.json.

At session end: stop or checkpoint controlled experiments, write measured results/negative findings, update state/ledger atomically, commit successful work and push discovery-lab. Update this document with exact commands, evidence paths and blockers. Never claim execution continues after the active session. No paid calls or large uncontrolled training. Private final-model gate remains CLOSED until its full prerequisite is independently demonstrated; a small working model, good prose or passing tests cannot open it.

Development environment used for this checkpoint: /private/tmp/model-forge-v02-venv/bin/python with PYTHONPATH=src. This path is local convenience, not a portability requirement. Standard setup is python3.12 -m venv .venv then .venv/bin/python -m pip install -e '.[dev]'. Keep heavyweight engines optional; permanent docs/ledger require no GPU or target API.

Latest verified bounded checkpoint: source commit677ec67 on discovery-lab, with public evidence under experiments/evidence. Follow-up owned decoder head research adds a separate bounded experiment, not complete transformer reconstruction. Fetch discovery-lab explicitly before switching if a clone fetches only main. Never replace the immutable master documents or reset an existing dirty checkout. Cross-platform CI is a tests-only workflow; stable native assets stay unchanged.

The durable fallback checkout is outputs/model-forge-research-checkout; it was independently cloned and its actual resume report validated. The original outputs/model-forge checkout is preserved on its historical branch after a stalled object import. Private run artifacts remain outside Git under work/research-artifacts and outputs/model-forge-research-data. A source package includes mission state and ledger; its existence does not establish a new native release.

The Documents fallback later also stalled during fast-forward and was stopped without resetting it. A separate durable local-data clone at ~/.local/share/modelforge-research/repository was created and its actual resume validated at202a664. It avoids the observed Documents file-provider stalls. In this sandbox its temporary repository-lock write requires an approved filesystem scope; ordinary local use runs under the user's file permissions. The workspace AGENTS.md carries the exact machine path. Source commits and public ledger also remain on GitHub discovery-lab. Do not use a clone with a still-pending Git merge as the sole source of current state.

Latest engineering fix source:01a27b3. Relevant59-test regressions passed on all three platforms in CI39647361066; this repairs the prior Windows MCP Git cleanup hang without weakening ancestry. Newest state/ledger records identify the exact next main-platform decoder-backbone protocol task. Do not rebuild stable packages or repeat the completed quadratic/attention/head/cache experiments without a discriminating verification reason. The private final model gate remains closed.

2026-10-22 takeover (Claude Code): research venv /private/tmp/model-forge-v02-venv/bin/python (python3.12 is not on PATH; use the venv with PYTHONPATH=src). Backbone protocol drafted in docs/decoder-backbone-protocol.md, O3 pilot recorded (not sealed). Next executable research action: sealed_o3_backbone_protocol in .modelforge/research_state.json. Disk free was ~4.6 GiB, a hard constraint for any private-model asset work.

2026-10-22 private-model track (user-authorized): private assets in ~/.local/share/modelforge-private (models/modelforge-private-4b-v1 = fused QLoRA of Apache-2.0 Qwen3-4B-Instruct-2507 4-bit; adapters/, data/, evals/, run_eval.py, build_v2.py, train_v3.py, finalize.py, validate_e2e.py). Use `modelforge private chat|serve|eval --model modelforge-private-4b-v1`. Sealed test 0.892 equals base. Next private action: program-aided math, licensed Hebrew SFT data, larger eval n (see next_private_model_action in state). Next research action unchanged: sealed_o3_backbone_protocol.

2026-10-23: private-model improvement uses `python -m modelforge.improvement run|decide|leaderboard|weakest` (suite v2, private ledger/leaderboard under ~/.local/share/modelforge-private/improvement). Baseline v1-greedy validation 0.906. In progress: v4 targeted SFT (build_v4.py → data/sft-v4). Research: SwiGLU seed-consensus identification pilots in examples/backbone_pilot (pilot_consensus/pilot_frozen_refit/pilot_polish).

2026-10-23 later: v4 candidate (adapters/sft-v4 on modelforge-private-4b-v1) validation 0.919 vs 0.906, p=0.23, not retained yet; extension run via run_ext.sh writes decide-v4-merged.json. Best verified model remains modelforge-private-4b-v1. Research engine: src/modelforge/swiglu_identification.py (teacher1 63/64 gates, teacher2 41/64). MASTER_MISSION.md is immutable by manifest digest and is intentionally not edited.

2026-10-23 promotion: best verified private model is modelforge-private-4b-v2 (default_model marker in private home). Leaderboards: improvement/leaderboard-{validation,test}.json. Test split has now been used once for v1/v4; future candidates must be selected on validation (incl. hard tiers) and confirmed on test only once each.

2026-10-24 current research checkpoint: main mission resumed at900973a; latest user prerequisite keeps the final private-model gate CLOSED. Historical licensed-base records above remain history and are not the next authorized action. New owned isolated-GLU antipodal route reconstructs only E(x), with sealed bias-free heldout error2.33e-15 and a failing biased control. See docs/antipodal-reconstruction.md and experiments/evidence/antipodal-even-v1/. Do not rerun sealed heldout or completed pilots. Continue preregistration of iteration-budget SwiGLU v2 and missing gate/coupling science, distinguishing saved O3 access from stronger continuous isolated queries.

Durable core Python is ~/.local/share/modelforge-research/runtime/bin/python (editable checkout, NumPy/core tools only); use it for scripts/resume_research.py. The old temporary venv may be absent. Validated agent skill installed at ~/.codex/skills/model-forge; bundled source is skills/model-forge/. No host MCP configuration was changed. Core tests-only CI excludes native rebuilds, Torch, GUI and private-model assets.
