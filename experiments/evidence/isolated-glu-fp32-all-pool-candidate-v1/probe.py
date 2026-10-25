from pathlib import Path
import importlib.util,json
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');out=Path('/private/tmp/modelforge-pool-review')
def load(name):
 s=importlib.util.spec_from_file_location(name,repo/f'examples/isolated_glu/{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('pool_candidate');e=load('pool_evaluate');v=np.array([[1.,0,0],[-1.,0,0],[0,1,0],[0,0,1]]);losses=np.array([100.,50.,1e9,1e6]);before=v.copy();idx=m.pool_indices(v,losses,3);assert idx==[1,3,2] and np.array_equal(before,v)
for width in [2,4]:
 try:m.pool_indices(v,losses,width)
 except ValueError as error:assert 'no truncation' in str(error)
 else:raise AssertionError('Countmismatch accepted')
try:m.pool_indices(np.eye(75),np.zeros(75),64)
except ValueError:pass
else:raise AssertionError('Overwidthtruncated')
# Real evaluation function is called only on a deliberately invalid synthetic input,
# before public_reference is reachable; no model/query/fit is performed.
reference_calls=[]
def forbidden_reference():reference_calls.append(True);raise AssertionError('Reference reached')
e.public_reference=forbidden_reference;guards={}
for name,method,expected in [('wrong-method','wrong_method','Method/calibration'),('wrong-calibration','isolated-glu-fp32-all-pool-candidate-v1','Method/calibration')]:
 p=out/name;p.mkdir(exist_ok=False);(p/'public').mkdir();np.savez(p/'candidate.npz',gates=np.ones((64,32)),couplings=np.ones((64,32,32)));(p/'public/manifest.json').write_text('{"channel":"fp32_compute"}');manifest_sha=e.sha(p/'public/manifest.json') if name=='wrong-method' else 'a'*64;(p/'fit-report.json').write_text(json.dumps({'candidate_built':True,'candidate_sha256':e.sha(p/'candidate.npz'),'method':method,'calibration_manifest_sha256':manifest_sha}))
 try:e.evaluate(p)
 except ValueError as error:assert expected in str(error);guards[name]=str(error)
 else:raise AssertionError(name+' guard failed')
 assert (p/'evaluation-attempt.json').exists()
 try:e.evaluate(p)
 except FileExistsError:guards[name+'_repeat']='FileExistsError'
 else:raise AssertionError('Repeatedinvalid evaluation allowed')
assert len(reference_calls)==0
result={'high_losses_not_filtered':True,'minimum_loss_antipodal_representative_selected':idx[0]==1,'sorted_greedy_indices':idx,'under_and_over_knownwidth_rejected_without_truncation':True,'over75_to64_rejected':True,'input_vectors_unchanged':True,'method_calibration_checks_before_reference':guards,'reference_function_calls':0,'no_actual_fit_or_evaluation_queries':True};(out/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
