from pathlib import Path
import importlib.util,json
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');out=Path('/private/tmp/modelforge-cubic-review')
def load(name):
 s=importlib.util.spec_from_file_location(name,repo/f'examples/isolated_glu/{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('cubic_candidate');o=load('cubic_observe');rng=np.random.default_rng(18423);E,C,Q,S=rng.normal(size=(4,9,3));rho=.03
# Independent polynomial Taylor control retains septic remainder, not only purecubic.
def signed(t):
 even=E*t*t;odd=.25*C*t**3+Q*t**5+S*t**7
 return even+odd,even-odd
fine=signed(rho);coarse=signed(2*rho);rhs=m.cubic_rhs(*fine,*coarse,rho);expected=C-16*S*rho**4;error=float(np.max(np.abs(rhs-expected)));assert error<1e-12;assert np.allclose(m.bridge(rhs)/(.25*.001**3),rhs,rtol=1e-15,atol=1e-15)
x=rng.normal(size=(11,3));calls=[]
def fake_query(a):calls.append(a.copy());return a*a+a**3
obs=o.observe(x,fake_query);assert len(calls)==4 and sum(len(a) for a in calls)==44;assert all(np.array_equal(a,b) for a,b in zip(calls,[.03*x,-.03*x,.06*x,-.06*x]));assert np.array_equal(obs['small_x'],x)
# Probe only metadata/loader arguments, never read calibration or invoke learner.
m.sha=lambda p:'0'*64;config={'calibration_manifest_sha256':'0'*64,'expected_files':{n:'0'*64 for n in ['hessians.npz','multiscale.npz','curvature-pair.npz']},'telemetry_sha256':'0'*64};additional={'channel':'fp32_compute','rho':.03,'coarse_rho':.06,'vector_calls':16384,'calibration_manifest_sha256':'0'*64,'raw_sha256':None}
m.read_json=lambda p:additional if p.parent.name=='additional' else {'channel':'fp32_compute','files':config['expected_files']};received=[]
def loader(path,shapes,expected_hash):
 if path.name=='signed.npz':received.append(expected_hash);raise RuntimeError('Stopped before any learner; signedhashprobe')
 return {}
m.read_arrays=loader
try:m.build(Path('cal'),Path('telemetry.npz'),out/'null-rawhash-control',config,Path('additional'),'0'*64)
except ValueError as e:guard={'null_rawhash_rejected':str(e),'signed_loader_reached':False}
except RuntimeError as e:assert 'signedhashprobe' in str(e);guard={'null_rawhash_rejected':False,'signed_loader_expected_hash':received[-1]}
else:raise AssertionError('Unexpected tinyprobe return')
result={'normalized_cubic_Richardson_septic_remainder_max_abs_error':error,'quintic_term_cancelled_and_septic_expected_negative16rho4_verified':True,'bridge_recovers_rhs_and_is_not_physical0_001_observation':True,'tinycollector_signed_batches_match_rhos':True,'tinycollector_actual_rows':44,'null_rawhash_guard':guard,'no_actual_teacherqueries_fitting_or_evaluation':True};(out/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
