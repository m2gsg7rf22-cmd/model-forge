# ModelForge permanent research instructions

Read docs/MASTER_MISSION.md, docs/RESEARCH_ROADMAP.md, docs/RESUME.md and .modelforge/research_state.json before changing this repository. The user master mission is immutable source material; the latest user instructions take precedence. All user-facing progress and final answers must be Hebrew.

On the exact user messages המשך, Continue, or Resume ModelForge: read these files, inspect Git status/recent commits, run scripts/resume_research.py, recover interrupted tasks and continue the highest-priority authorized work. Never ask the user to reconstruct context. Do not repeat completed experiments without a concrete verification reason.

The objective is authorized model reconstruction research, not a claim that universal reconstruction is possible. Separate original parameter recovery, equivalent parameterizations/computation, defined-domain functional equivalence and finite behavioral approximation. Use owned hidden teachers and an independent verifier; never expose test labels/teacher weights to fitting or tune against the same sealed test set repeatedly. Record assumptions, failures, numerical uncertainty and compute cost.

The private Luna-5.6-level model mission is gated: do not begin model creation/training/integration until the complete platform and its required reconstruction capabilities have been independently verified. The gate is currently CLOSED. Never publish proprietary model files, private implementation, datasets or credentials. Do not claim Luna-level capability without an identified target and broad authorized comparisons. Do not bypass access controls or use restricted provider outputs for training.

Use bounded concurrency (at most two experiment workers; one GPU experiment), <=2048 MiB per initial experiment, <=180 seconds, no paid calls without explicit authorization. No automatic execution after the session ends is claimed. No scheduler or infinite loop by default. Checkpoint atomically and append scientific evidence before moving to the next action. Research runs remain isolated from stable v0.2.0 and v0.1.0 artifacts.

Use existing NumPy/HF/MLX infrastructure. Test changed paths, strict mypy and Ruff; run broad regression only at an integration gate. Do not rebuild unchanged native packages. Persist runnable configs/source digests/results and negative outcomes in the ledger. Large private teacher/checkpoint files belong outside Git. Retain-or-reject is based on preregistered validation; reserve sealed test evaluation for independent verification. A verification failure or budget stop is not success.

A separate verification agent must challenge major research claims. All agents share one checkout: declare file ownership, avoid concurrent edits to the same file, and do not commit another agent's incomplete work. Preserve existing user changes. No native GUI permission wait should block independent engineering.
