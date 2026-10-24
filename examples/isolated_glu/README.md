# Isolated GLU development pilot (not a stable engine)

These NumPy-only scripts study one owned bias-free SwiGLU fixture, d32/width64/output32. They are **not** a universal model reconstruction route or a text-API extraction method. The development seed is public; the teacher is not secret. Do not call this a sealed experiment. No private final-model training, downloads or paid queries occur.

From an installed source checkout, choose a new absolute DATA directory (prepare refuses an existing directory):

```sh
python examples/isolated_glu/hessian_pilot.py prepare /absolute/new-development-data
python examples/isolated_glu/refine.py /absolute/new-development-data
python examples/isolated_glu/multiscale.py prepare /absolute/new-development-data
```

Use the existing core interpreter if Python on PATH lacks ModelForge: `~/.local/share/modelforge-research/runtime/bin/python`. Dependencies are NumPy and the installed core. Keep BLAS at one thread, as the scripts do. For process-tree RSS/timeout bounds invoke each command with `modelforge.execution.run_json(..., {}, cwd=MONITOR_DIRECTORY, memory_limit_mb=512, timeout_seconds=180)`; see tests/test_glu_pilot.py. The command scripts alone have a 150-second child timeout for fitting, but do not enforce a parent RSS limit.

`hessian_pilot.py solve DATA` consumes saved public hessians.npz only; `refine.py DATA` consumes Hessians and initial candidate directions; `multiscale.py solve DATA` consumes public multiscale-observations.npz and refined candidate directions. These solve stages do not use the development teacher. They currently assume this fixed size/probe contract, not arbitrary dimensions/activations. Do not pass untrusted or incompatible NPZ files. `allow_pickle=False` is enforced; shape/finite/type hardening required for a public service is not complete. The scripts are not registered in MCP/CLI as a supported full-GLU engine.

The candidate artifact `multiscale-solution.npz` contains gates[64,32] and rank-one couplings[64,32,32]. For batch x[N,32], evaluate `t=x@gates.T; prediction=einsum('nw,wod,nd->no', t*.5*(1+tanh(t/2)), couplings, x)`. Up/down branch factors have reciprocal scaling ambiguity; the artifact deliberately stores their products.

Hessians use 135168 vector queries; multiscale/even fitting adds10272, total145440, before 256 development validation calls. The independent review adds2048 fresh validation calls. Query scales and precision are essential: arbitrary text inputs do not expose these responses. No claim of original stored weights, all-input equivalence, complete Transformer or Luna-level capability follows. Fresh secret teachers and an independently reviewed preregistered sealed protocol are required next.
