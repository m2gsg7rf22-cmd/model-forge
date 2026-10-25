from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');O=Path('/private/tmp/modelforge-forward-factor-review');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('forwardfactor',R/'examples/isolated_glu/forward_factor.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(770214);cases=[]
for dim in (1,3,5):
 g=rng.normal(size=dim);x=rng.normal(size=(23,dim));w=np.exp(rng.uniform(-1,1,size=23));physical=rng.normal(size=(23,4));Y=w[:,None]*physical
 t=x@g;phi=(w*t/(1+np.exp(-t)))[:,None]*x
 C=rng.normal(size=(4,dim));assert np.allclose(m.operator(g,C,x,w),phi@C.T,rtol=1e-12,atol=1e-13)
 assert np.allclose(m.adjoint(g,Y,x,w),Y.T@phi,rtol=1e-12,atol=1e-13)
 assert np.allclose(m.gram(g,x,w),phi.T@phi,rtol=1e-12,atol=1e-13)
 assert np.isclose(np.sum((phi@C.T)*Y),np.sum(C*m.adjoint(g,Y,x,w)),rtol=1e-12,atol=1e-12)
 # Independent QR coordinate solve, not normal-equation whitening.
 Q,T=np.linalg.qr(phi,mode='reduced');W=Y.T@Q;u,sv,v=np.linalg.svd(W,full_matrices=False);Z=(u[:,0]*sv[0])[:,None]*v[:1];expected=np.linalg.solve(T,Z.T).T
 actual,report=m.rankone_update(g,Y,x,w);error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));assert error<1e-11
 objective=float(np.sum((Y-phi@actual.T)**2));lower=float(np.sum((Y-Q@(Q.T@Y))**2)+np.sum(sv[1:]**2));assert np.isclose(objective,lower,rtol=1e-12,atol=1e-12);assert np.isclose(objective,report['weighted_objective'],rtol=1e-12)
 assert report['target_semantics']=='already_weighted_once' and report['weights_semantics']=='residual_multipliers_not_squared_loss_weights'
 exact=rng.normal(size=(4,1))*rng.normal(size=(1,dim));recovered,_=m.rankone_update(g,phi@exact.T,x,w);assert np.allclose(recovered,exact,rtol=1e-11,atol=1e-12)
 zero,_=m.rankone_update(g,np.zeros((23,4)),x,w);assert np.array_equal(zero,np.zeros_like(zero))
 cases.append({'dimension':dim,'qr_solution_relative_error':error,'weighted_objective':objective})
# Positive finite weights remain mandatory. Near-singular forward features reject.
x=rng.normal(size=(23,2));x[:,1]*=1e-7;rejected=False
try:m.rankone_update(np.array([1.,0.]),np.ones((23,2)),x,np.ones(23))
except ValueError as exc:assert 'whitening' in str(exc);rejected=True
assert rejected
for wbad in [np.zeros(23),-np.ones(23),np.full(23,np.inf),np.ones(23,dtype=np.float32)]:
 try:m.features(np.ones(2),np.ones((23,2)),wbad)
 except ValueError:pass
 else:raise AssertionError('invalid weight accepted')
result={'synthetic_cases':cases,'independent_qr_coordinate_solution':True,'ill_conditioned_guard_rejected':True,'invalid_weights_rejected':True,'physical_y_weighted_once':True,'actual_experiment_arrays_read':0,'teacher_queries':0,'saved_candidate_fits':0}
(O/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
files=['examples/isolated_glu/forward_factor.py','tests/test_glu_forward_factor.py','docs/glu-forward-factor-primitive.md'];review={'approved':True,'scope':'synthetic-only weighted fixed-gate full-forward rank-at-most-one local quadratic primitive','sources':{p:sha(R/p) for p in files},'blocking_findings':[],'independent_probe_sha256':sha(O/'probe.py'),'probe_results_sha256':sha(O/'probe-results.json'),'limitations':['Mathematical local minimizer in exact arithmetic; no floating-point certificate near condition threshold.','Response arrays must already have residual multipliers applied once; squared objective uses w squared.','No saved-data fit, forward-quality improvement, guard selection or universal reconstruction established.','A future actual protocol must freeze independent calibration train/guard data, ranking, retention, sweeps, resource bounds and a fresh one-time public evaluation.','Previous Hessian-polish quality failure preserved; private gate closed.']}
(O/'source-review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(result));print('review_sha256',sha(O/'source-review.json'))
