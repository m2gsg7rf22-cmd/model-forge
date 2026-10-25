import importlib.util,json
from pathlib import Path
import numpy as np
P=Path('/private/tmp/modelforge-guard-aware-evaluation-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sp=importlib.util.spec_from_file_location('final_batch_eval',R/'examples/isolated_glu/guard_aware_evaluate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
shared=np.empty((2,2))
def recycled(x):shared[:]=x*1.0005;return shared
root=P/'final-shared-buffer';r=m.evaluate_batches(root,lambda x:x.copy(),recycled,recycled,seed=33,rows=2,dimension=2)
with np.load(root/'paired-validation.npz') as aggregate:
 for group in range(4):
  with np.load(root/f'group-{group}-student.npz') as data:assert np.array_equal(aggregate['prediction'][group*2:group*2+2],data['prediction'])
assert r['status']=='completed' and r['all_groups_relative_at_most_1e_minus3'] is True and all(r[k]==8 for k in ['calls_reserved','calls_attempted','calls_returned','calls_persisted'])
calls=[]
def fail_second(x):
 calls.append(1)
 if len(calls)==2:raise RuntimeError('second synthetic query failure')
 return x.copy()
root2=P/'final-query-failure'
try:m.evaluate_batches(root2,fail_second,lambda x:x.copy(),lambda x:x.copy(),seed=34,rows=2,dimension=2)
except m.EvaluationError:pass
else:raise AssertionError('query failure accepted')
r2=json.loads((root2/'progress.json').read_text());assert [r2[k] for k in ['calls_reserved','calls_attempted','calls_returned','calls_persisted']]==[8,4,2,2] and len(r2['groups'])==1
root3=P/'final-predict-failure'
def fail_prediction(x):raise RuntimeError('prediction failed')
try:m.evaluate_batches(root3,lambda x:x.copy(),fail_prediction,lambda x:x.copy(),seed=35,rows=2,dimension=2)
except m.EvaluationError:pass
else:raise AssertionError('prediction failure accepted')
r3=json.loads((root3/'progress.json').read_text());assert r3['calls_returned']==r3['calls_persisted']==2 and (root3/'group-0-reference.npz').exists()
result={'synthetic_seeds':[33,34,35],'shared_output_buffers_now_detached':True,'complete_counter_profile':[8,8,8,8],'failed_second_query_counter_profile':[8,4,2,2],'reference_preserved_before_predict_failure':True,'actual_seed_generation':False,'actual_reference_factory_calls':0,'actual_markers_created':False}
(P/'final-batch-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
