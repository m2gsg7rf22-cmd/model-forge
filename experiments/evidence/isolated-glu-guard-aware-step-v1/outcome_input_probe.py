from pathlib import Path
import json,hashlib,subprocess
import numpy as np
P=Path('/private/tmp/modelforge-guard-aware-step-review');B=Path('/Users/adamfaruhi123/.local/share/modelforge-research/experiments');D=B/'fp32-guard-aware-gate-step-v1-20261025-195d9a6';R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');old=B/'fp32-guarded-gate-step-v1-20261025-3d3b622';cal=B/'fp32-guarded-forward-polish-v1-20261025-db8fb33/calibration'
paths={'candidate':old/'candidate/candidate.npz','compact_metadata':old/'evaluation-recovery/candidate/fit-report.json','full_fit_report':old/'candidate/fit-report.json','observations':cal/'observations.npz','manifest':cal/'manifest.json','original_manifest':B/'fp32-fullmap-dev-v1-20261025-8d2d457/public/manifest.json'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();actual={k:sha(v) for k,v in paths.items()};worker=json.loads((D/'worker/worker-report.json').read_text());assert actual==worker['inputs_before']==worker['inputs_after']
with np.load(D/'worker/step/initial-problem.npz') as initial,np.load(paths['candidate']) as c,np.load(paths['observations']) as o:
 assert np.array_equal(initial['G'],c['gates']) and initial['C'].tobytes()==c['couplings'].tobytes()
 for key in ['train_x','train_y','guard_x','guard_y']:assert np.array_equal(initial[key],o[key])
subprocess.run(['git','merge-base','--is-ancestor','195d9a67e0993dbefb5bb6fb3b26a95fe15db3a6','HEAD'],cwd=R,check=True)
v=json.loads((P/'outcome-audit.json').read_text());v.update(six_original_inputs_independently_rehashed=actual,initial_problem_exact_original_arrays=True,actual_source_is_current_head_ancestor=True,input_probe_sha256=sha(Path(__file__)),outcome_probe_sha256=sha(P/'outcome_probe.py'))
v['limitations']=[s for s in v['limitations'] if not s.startswith('Frozen input')];v['limitations'].append('All6originalinputfiles independently rehashed; only candidate and calibration arrays decoded; original fullfitreport hash-only.')
(P/'outcome-audit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'audit_sha256':sha(P/'outcome-audit.json'),'all6inputhashes':True,'initial_arrays_bitwise_equal':True,'actualsource_ancestor':True}))
