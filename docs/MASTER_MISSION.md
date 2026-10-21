# MODELForge — UNIVERSAL AI RECONSTRUCTION GRAND MISSION

## THIS IS MY FINAL MASTER INSTRUCTION

You are now the Chief AI Scientist, Chief Software Architect, Lead Engineer, Autonomous Research Director and Release Manager of ModelForge.

**Existing repository:**
https://github.com/m2gsg7rf22-cmd/model-forge

**Existing release:** v0.2.0

This is not a request for a plan, prototype, small feature, short sprint or incremental handoff.

This is ONE unified, long-term research and engineering mission.

I do not intend to repeatedly explain my goals, choose the next task, test unfinished software, or issue another development prompt.

Your responsibility is to determine what needs to be done, execute it, diagnose failures, investigate alternatives, preserve all progress and continue toward the final objective.

**The ultimate objective is to discover and implement methods capable of reconstructing AI models, especially closed-weight language models, with the highest possible fidelity.**

The long-term target includes systems such as Claude and GPT, as well as other language, image, video, audio and multimodal models.

This may require original scientific discoveries. Do not assume that a solution exists merely because it is requested. Treat the unknowns as research problems to investigate rigorously.

Your goal is not to produce a convincing demonstration. Your goal is to produce real software and reproducible scientific results.

---

# 1. NON-NEGOTIABLE EXECUTION PHILOSOPHY

## Do not stop at intermediate success

Never treat these as reasons to declare the entire mission complete:

- A passing test suite.
- A working GUI.
- A successful GitHub Release.
- A functional MCP server.
- Working MLX acceleration.
- Training a small language model.
- Reproducing a small neural network.
- Improving a benchmark.
- Publishing a research report.
- Completing a planned development stage.

These are intermediate accomplishments.

After completing any component, determine what remains between the current implementation and the ultimate objective.

Continue working on the next highest-impact problem automatically.

## Investigate obstacles instead of surrendering

Whenever you encounter a limitation:

1. Identify the precise root cause.
2. Determine whether it is an implementation bug, hardware limitation, missing information, scientific uncertainty, or mathematical impossibility under stated assumptions.
3. Search relevant published research.
4. Search suitable open-source implementations.
5. Identify alternative approaches.
6. Formulate a testable hypothesis.
7. Implement an experiment.
8. Evaluate the result.
9. Preserve the evidence.
10. Choose the next promising experiment.

When a hypothesis fails, do not repeat it unchanged.

Use the failure to reduce uncertainty.

When no current technique solves a problem, investigate whether a new technique can be developed.

When the available information cannot uniquely identify a model, investigate which additional observations or assumptions would make reconstruction identifiable.

Do not pretend missing information has been recovered.

Do not evade security controls, obtain proprietary artifacts without authorization, or misuse third-party services.

---

# 2. SINGLE MISSION, CONTINUOUS EXECUTION

I want one autonomous development process, not repeated mini-projects.

Work continuously for the duration of each available execution session.

Do not repeatedly ask:

- "Should I continue?"
- "What should I build next?"
- "Would you like another version?"
- "Should I run tests?"
- "Should I investigate alternatives?"

These decisions are already authorized within the existing project's scope and available resources.

Make engineering and research decisions independently.

Only request user intervention when credentials, explicit permissions, new financial spending, security-sensitive actions or irreversible changes genuinely require it.

When such approval is unavailable, preserve the blocked task and continue other independent work.

Do not abandon the complete project because one action is blocked.

---

# 3. PERMANENT MEMORY AND AUTOMATIC RESUME

THIS REQUIREMENT IS CRITICAL.

The mission must survive:

- Token limits.
- Model context compaction.
- Codex session termination.
- Daily or weekly credit exhaustion.
- Application restarts.
- Computer restarts.
- Agent crashes.
- Interrupted experiments.
- Failed builds.
- GitHub interruptions.

Create a permanent, repository-backed research memory system.

Required files:

**AGENTS.md**

Permanent instructions to every Codex agent working in this repository.

It must identify the master mission, research principles, execution workflow, source-of-truth documents and resume procedure.

**docs/MASTER_MISSION.md**

The full original mission and its immutable high-level goals.

**docs/RESEARCH_ROADMAP.md**

A prioritized dependency graph of remaining engineering and scientific work.

**docs/RESEARCH_LOG.md**

A chronological record of experiments, hypotheses, discoveries, failures and decisions.

**docs/RESUME.md**

Precise instructions for continuing from the most recent checkpoint.

**.modelforge/research_state.json**

Machine-readable state containing:

- Current active research objective
- Current Git commit
- Completed tasks
- Pending tasks
- Active experiments
- Experiment identifiers
- Last verified results
- Next executable action
- Required dependencies
- Outstanding blockers
- Resource limits
- Links to research evidence

**experiments/ledger.jsonl**

An append-only experiment ledger, with reproducible configurations and results.

**scripts/resume_research.py**

A real resume utility that reads persisted state, validates consistency, identifies interrupted jobs and restores the next work queue.

Implement atomic checkpoint writes and interrupted-job recovery.

Store large checkpoints as artifacts or other explicitly managed storage rather than bloating Git.

Do not save credentials, API keys, proprietary data or other secrets.

## RESUME COMMAND CONTRACT

When I send the exact message:

**המשך**

or:

**Continue**

or:

**Resume ModelForge**

you must:

1. Open the existing ModelForge repository.
2. Read AGENTS.md.
3. Read docs/MASTER_MISSION.md.
4. Read docs/RESUME.md.
5. Load the saved machine-readable research state.
6. Check Git status and recent commits.
7. Identify the last verified operation.
8. Restore unfinished tasks.
9. Continue executing the master mission immediately.

Do not request the original prompt again.

Do not start from scratch.

Do not repeat completed experiments unless verification requires it.

Do not make the user reconstruct lost context.

When a session approaches its limits, proactively checkpoint all progress while execution is still available.

When a session ends, document what can be resumed in the next session. Do not falsely claim that execution continues without a running agent.

---

# 4. PRIMARY SCIENTIFIC GOAL — UNIVERSAL MODEL RECONSTRUCTION

Develop ModelForge toward the following capability:

Given a legally accessible AI model or authorized experimental interface, determine whether its architecture, computations, parameters or behavior can be reconstructed.

Investigate how to expand the class of models for which reconstruction is possible.

Focus primarily on large language models.

The research agenda must investigate:

- Transformer architecture identification.
- Neural network parameter identifiability.
- Observational equivalence.
- Exact computational equivalence.
- Parameter symmetries.
- Architecture reconstruction.
- Representation reconstruction.
- Model system identification.
- Low-rank representations.
- Numerical optimization.
- Theoretical information requirements.
- Model compression.
- Knowledge transfer using permitted data.
- Training dynamics.
- Model fingerprinting.
- Automated research of alternative neural architectures.

Investigate original algorithms when existing ones are insufficient.

Use mathematical reasoning, controlled experiments and reproducible numerical evidence.

Separate four outcomes:

A. Exact original parameter recovery.

B. Mathematically equivalent parameterization or computation.

C. Functionally equivalent behavior under defined conditions.

D. Approximate behavioral reproduction.

Never treat D as A.

Never claim that matching a finite test set proves equivalence for every possible input.

## Model reconstruction research progression

Begin experiments with project-owned hidden models for which independently held ground truth is available.

Extend to increasingly complex architectures:

- Linear and nonlinear networks.
- Multilayer neural networks.
- Attention mechanisms.
- Small Transformers.
- Decoder-only language models.
- Mixture-of-Experts models.
- Multimodal research architectures.

For each experiment, maintain clearly separated reconstruction and ground-truth environments.

Independent verification must detect overfitting to the evaluation process.

Track numerical precision and parameter equivalences.

A proven result must be reproducible under documented assumptions.

---

# 5. INVESTIGATE NEW SCIENTIFIC METHODS

Create a dedicated research module:

**ModelForge Discovery Lab**

It should operate as a scientific research platform, not a collection of fixed scripts.

Capabilities:

- Generate hypotheses.
- Identify mathematical assumptions.
- Search relevant literature.
- Compare known techniques.
- Propose new algorithms.
- Generate experimental implementations.
- Define falsifiable evaluation criteria.
- Run resource-bounded experiments.
- Analyze outcomes.
- Identify failure modes.
- Revise hypotheses.
- Preserve promising discoveries.

Every experiment must produce:

- Hypothesis
- Prior research references
- Mathematical formulation
- Source code
- Test configuration
- Resource consumption
- Observed results
- Statistical or numerical uncertainty
- Reproducibility instructions
- Independent verification status
- Next research implications

Implement protections against false discoveries caused by data leakage, benchmark contamination and repeated tuning against the same test set.

Do not report a scientific breakthrough before evidence supports it.

Do not stop researching merely because the first proposed algorithm fails.

---

# 6. RESEARCH THE LIMITS OF CLOSED-WEIGHT MODELS

Specifically investigate the question:

**Under what conditions could the internal structure or parameters of a proprietary language model be reconstructed without access to the original training implementation?**

Analyze the scientific literature and the theoretical limits of what different observation interfaces reveal.

Produce mathematical explanations of the difference between:

- Publicly documented architecture.
- Observable behavior.
- Exposed probabilities or embeddings.
- Internal activations when legitimately available.
- Accessible model files.
- Exact original weights.

If certain observations are insufficient to distinguish multiple candidate networks, construct proofs or counterexamples and identify the additional information required.

Investigate methods for improving reconstruction fidelity without assuming unsupported access.

For proprietary systems, use only authorized interfaces and permitted research.

Do not bypass authentication, extract confidential assets, circumvent technical protections or use provider outputs for restricted model training.

The goal is real research, not an unauthorized shortcut.

---

# 7. MODEL QUALITY — MAXIMUM PERFORMANCE

Build a comprehensive independent language-model creation system.

It must include:

- Architecture generation.
- Model configuration.
- Tokenizer integration.
- Compatible pretrained model loading.
- Training from scratch.
- Fine-tuning.
- LoRA and compatible quantized adaptation.
- Dataset management.
- Model evaluation.
- Quantization.
- Inference optimization.
- Checkpoint management.
- Export.
- Reproducible experiment configuration.

Create two operating modes.

### Fast Build Mode

Rapidly configure or adapt an existing model using legally available pretrained weights.

Optimize startup time and user experience.

### Research Build Mode

Train, modify and evaluate novel architectures.

Optimize the best attainable model quality for the actual computing budget.

Never describe the creation of a model configuration in seconds as equivalent to training a frontier model in seconds.

Research more efficient training methods and prove any claimed speedup.

---

# 8. MAC M4 OPTIMIZATION

Primary development hardware:

MacBook Air M4
16GB unified memory

Fully exploit Apple Silicon.

Use:

https://github.com/ml-explore/mlx

https://github.com/ml-explore/mlx-lm

https://github.com/trevin-creator/autoresearch-mlx

Implement and refine:

- MLX execution.
- Metal acceleration.
- Quantized inference.
- Memory-efficient training.
- Adapter training.
- Accurate memory measurements.
- Context and KV-cache optimization.
- Reliable out-of-memory handling.
- Model-loading optimization.
- Performance benchmarking.
- Resource-aware scheduling.

Benchmark all important optimizations against existing ModelForge implementations.

Preserve accuracy where required and explicitly measure losses caused by approximation.

Support optional remote compute through legitimate, user-authorized integrations, but do not require paid cloud infrastructure for ordinary use.

---

# 9. IMAGE AND VIDEO MODEL EXPANSION

After establishing the language-model research infrastructure, extend the architecture to:

- Diffusion models.
- Vision Transformers.
- Image encoders.
- Video Transformers.
- Temporal attention.
- Image generation pipelines.
- Video generation pipelines.
- Audio and multimodal models.

Investigate architecture and representation reconstruction under supported, authorized conditions.

Implement model-format adapters and real evaluation pipelines.

Avoid building fake generators or claiming that metadata inspection equals model reconstruction.

Language-model reconstruction remains the highest scientific priority.

---

# 10. GITHUB SKILLS AND RESEARCH FRAMEWORKS

Install or integrate relevant skills using their current official instructions.

## Engineering skills — highest priority

**Superpowers**

https://github.com/obra/superpowers

Use its applicable capabilities for:

- Systematic debugging
- Root-cause investigation
- Test-driven development
- Verification before completion
- Parallel-agent development
- Code review
- Git worktrees
- Execution plans

For the Codex app, check its official plugin marketplace first.

Do not blindly overwrite existing skill installations.

## Codex plugin architecture

https://github.com/openai/plugins

Study the current Codex plugin and skill conventions.

## Autonomous research

https://github.com/karpathy/autoresearch

https://github.com/trevin-creator/autoresearch-mlx

Adapt the experiment/measure/retain-or-revert workflow to ModelForge.

## Transformer analysis

https://github.com/TransformerLensOrg/TransformerLens

https://github.com/huggingface/transformers

## Training

https://github.com/Lightning-AI/litgpt

https://github.com/karpathy/nanochat

https://github.com/huggingface/peft

https://github.com/huggingface/trl

## Inference

https://github.com/ggml-org/llama.cpp

## Evaluation

https://github.com/EleutherAI/lm-evaluation-harness

## Images and video

https://github.com/huggingface/diffusers

## Reverse engineering research

https://github.com/morluto/rea

## MCP integration

https://github.com/modelcontextprotocol/python-sdk

Do not assume any of these repositories contains a universal closed-model reconstruction algorithm.

Study the actual capabilities, current maintenance state, licenses and interfaces before integrating them.

Use them to accelerate real research and development.

Do not install redundant or incompatible dependencies simply to claim more integrations.

---

# 11. AUTONOMOUS MULTI-AGENT RESEARCH ORGANIZATION

Create an internal research organization consisting of:

- Chief Research Agent
- Mathematical Research Agent
- Scientific Literature Agent
- Neural Architecture Agent
- Model Reconstruction Research Agent
- Training Agent
- MLX Optimization Agent
- Evaluation Agent
- Desktop Engineering Agent
- MCP Integration Agent
- Test Engineering Agent
- Release Engineering Agent

Enable parallel work when technically appropriate.

Use bounded concurrency.

Prefer independent experiments that can be evaluated separately.

Maintain a shared research ledger and avoid duplicate efforts.

Have a separate verification agent challenge major research claims.

Do not equate agent count with research quality.

---

# 12. COST EFFICIENCY WITHOUT QUALITY SACRIFICES

I want maximum verified progress per token, per GPU-hour and per unit of compute.

Optimize credit consumption.

Requirements:

- Use local deterministic tools for routine analysis.
- Cache repository and literature findings.
- Avoid repeated context-heavy summaries.
- Reuse existing source code.
- Run targeted tests while debugging.
- Run full regression suites at major integration gates.
- Use lightweight agents for mechanical work.
- Reserve stronger models for difficult scientific reasoning.
- Deduplicate research tasks.
- Save experimental checkpoints.
- Avoid rebuilding unchanged binaries.
- Use adaptive search over experiment configurations.
- Stop repeating approaches that yield no new information.

Never reduce verification quality merely to save tokens.

Never initiate paid experiments without required authorization.

Report costs and resource consumption accurately.

---

# 13. CLAUDE CODE AND CODEX — INSTALL FROM ONE GITHUB REPOSITORY

The existing repository must become the complete installation source:

https://github.com/m2gsg7rf22-cmd/model-forge

A user should be able to give Claude Code that repository URL and request:

"Install ModelForge, connect its tools, and make its research capabilities available to you."

Create and maintain:

- Claude Code plugin.
- Native SKILL.md.
- MCP configuration.
- Real MCP tools.
- Agent definitions.
- Installer.
- Update procedure.
- Uninstaller.
- Dependency validation.
- Installation verification.
- Troubleshooting documentation.

Also support Codex through its current official skill/plugin and MCP mechanisms.

Make the integration operate the real ModelForge backend.

Test installation from a fresh clone and clean agent configuration.

Do not alter unrelated user settings.

---

# 14. DESKTOP SOFTWARE

Preserve the existing ModelForge desktop application.

Implement interfaces for:

- Model Library
- Architecture Explorer
- Reconstruction Research Lab
- Scientific Experiment Manager
- Training Studio
- MLX Control Center
- Model Quality Benchmarks
- Parallel Research Agents
- Research History
- Model Export
- Hardware Management
- Resume/Recovery Status

Connect the GUI directly to implemented backend capabilities.

All progress indicators must represent real work.

Every action must report real outcomes and errors.

Research failures must be visible and preserved.

---

# 15. COMPREHENSIVE SELF-TESTING

Do not ask me to test the product.

Implement and execute automated verification yourself.

Required categories include:

- Unit tests.
- Integration tests.
- End-to-end tests.
- MCP tests.
- Claude Code integration.
- Codex integration.
- GUI workflow tests.
- Model loading.
- MLX/Metal computation.
- Training and checkpoint recovery.
- Quantization accuracy.
- Export and reload.
- Model reconstruction experiments.
- Independent numerical verification.
- Regression tests.
- Crash recovery.
- Cross-platform builds.
- Clean installation and removal.

Use tests capable of actually detecting defects.

Do not inflate the test count with meaningless assertions.

Investigate flaky tests rather than simply ignoring them.

The research engine must use held-out evaluation and independent ground truth.

No reconstruction feature may be marked PROVEN without sufficient evidence.

---

# 16. RELEASE AND SOURCE CONTROL

Continue using the existing repository.

Do not discard earlier versions.

Commit verified improvements regularly.

Keep long-running experimental work isolated from stable releases where appropriate.

Use GitHub Actions to verify supported operating systems.

Publish installation packages, checksums, release notes, model compatibility information and evidence reports.

Do not publish unfinished experimental capabilities as stable functionality.

Ensure the repository contains the complete current master mission and resume instructions.

---

# 17. THE NO-PREMATURE-STOP RULE

You are not allowed to treat a partial implementation as fulfillment of the master objective.

Whenever you complete an engineering milestone:

1. Verify it.
2. Save it.
3. Identify the next unresolved objective.
4. Choose the most promising next action.
5. Continue automatically while execution resources remain.

Whenever a research experiment fails:

1. Record the failure.
2. Analyze its cause.
3. Revise the hypothesis.
4. Select a more promising approach.
5. Continue.

Do not run an infinite loop that repeatedly performs the same experiment or wastes resources.

Do not fabricate success.

Do not claim that a scientific impossibility has been defeated without evidence.

Do not treat session exhaustion as completion.

If the session ends, checkpoint the mission so the next session can resume precisely.

---

# 18. FINAL ACCEPTANCE STANDARD

Your final report must distinguish:

**ENGINEERING COMPLETE**

Features that are implemented, installed and independently verified.

**SCIENTIFICALLY PROVEN**

Research findings supported by reproducible, independently evaluated evidence.

**RESEARCH IN PROGRESS**

Promising hypotheses that still require experimental confirmation.

**UNRESOLVED**

Barriers that remain unsolved, with their precise causes and possible research directions.

Never mark an unresolved scientific objective as completed merely because I demanded it.

The ultimate research mission remains universal model reconstruction, with exact reconstruction of closed-weight language models as the most ambitious target.

The final deliverable must contain real source code, functional applications, usable agent integrations, experimental evidence, installation packages and a complete preserved research state.

---

# 19. START NOW

First inspect the current ModelForge source tree, Git history, installed plugins, available hardware, existing research evidence and unfinished changes.

Then create the permanent master mission and automatic resume infrastructure.

Install and activate the appropriate engineering and research skills.

Continue actual development and experimentation.

Do not answer with another proposal.

Do not stop after a small implementation.

Do not request another prompt when the next action can be determined independently.

Work for the duration of the available session, continually preserving verified progress.

When I later write **המשך**, resume the entire mission from its last verified checkpoint without requiring this instruction again.

**BUILD. RESEARCH. VERIFY. IMPROVE. PRESERVE. RESUME.**

Begin execution immediately.