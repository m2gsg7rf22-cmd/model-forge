from pathlib import Path
import importlib.util,json
import numpy as np
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');s=importlib.util.spec_from_file_location('fp32full',repo/'examples/isolated_glu/fp32_fullmap.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);root=Path('/private/tmp/modelforge-fp32-fullmap-review');calls=[]
def synthetic(x):return np.asarray((np.tanh(x.astype(np.float32))+np.float32(.125)),dtype=np.float64)
def query(x):calls.append(len(x));return synthetic(x)
data=m.collect_multiscale(query);assert sum(calls)==10272;assert np.array_equal(data['small_odd'],.5*(synthetic(data['small_x']*.001)-synthetic(-data['small_x']*.001)));assert np.array_equal(data['large_odd'],.5*(synthetic(data['large_x']*1000)-synthetic(-data['large_x']*1000)));assert np.array_equal(data['even_pos'],synthetic(m.canonical_probe_points(32)))
# Rejection paths must refuse before any teacher/evaluation access.
case=root/'reject';case.mkdir(exist_ok=True);m.write(case/'fit-report.json',{'candidate_built':False});m.public_weights=lambda:(_ for _ in ()).throw(AssertionError('Teacher touched after rejection'))
try:m.evaluate(case)
except ValueError as e:assert 'No accepted' in str(e)
else:raise AssertionError('Rejected candidate evaluated')
# No fitting: replace loader + learner with an interruption counter.
case=root/'interrupted-repair';case.mkdir(exist_ok=True);m.sha=lambda p:'a'*64;m.read_json=lambda p:{'channel':'fp32_compute','files':{n:'a'*64 for n in ['hessians.npz','multiscale.npz','curvature-pair.npz']}};m.read_arrays=lambda p,shapes,expected_hash:{'hessian':np.zeros((1,))} if 'hessian' in shapes else {};invocations=[]
def interrupted(h):invocations.append(True);raise RuntimeError('Synthetic interrupted worker, no fitting')
m.glu32.directions=interrupted
for _ in range(2):
 try:m.fit(case)
 except (RuntimeError,FileExistsError):pass
assert len(invocations)==1 and (case/'fit-attempt.json').exists()
result={'multiscale_actual_callback_row_calls':sum(calls),'callback_batch_inventory':calls,'all_saved_channels_match_synthetic_fp32_response_difference':True,'rejection_refuses_before_teacher_access':True,'interrupted_mock_learner_entries':len(invocations),'fit_attempt_marker_exists':(case/'fit-attempt.json').exists(),'no_actual_fitting_queries_or_private_access':True};(root/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
