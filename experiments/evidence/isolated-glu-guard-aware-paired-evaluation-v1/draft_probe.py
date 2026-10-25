import importlib.util,json
from pathlib import Path
import numpy as np
P=Path('/private/tmp/modelforge-guard-aware-evaluation-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
spec=importlib.util.spec_from_file_location('ind_eval_draft',R/'examples/isolated_glu/guard_aware_evaluate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=np.array([[1.,1e-200]]);b=np.array([[1.,0.]])
underflow=m.metrics(a,b);assert underflow==(0.,1e-200)
def mutate(x):x[:]=7;return x.copy()
root=P/'draft-mutating-callback';r=m.evaluate_batches(root,mutate,lambda x:x.copy(),lambda x:x.copy(),seed=25,rows=2,dimension=2)
with np.load(root/'group-0-inputs.npz') as group,np.load(root/'paired-validation.npz') as all:
 mismatch=not np.array_equal(group['inputs'],all['inputs'][:2])
assert mismatch and r['status']=='completed'
result={'synthetic_seed':25,'actual_queries':0,'actual_seed_generation':False,'nonzero_delta_underflow_reported_zero_relative':True,'maximum_absolute_nonzero':underflow[1],'query_mutation_causes_completed_mismatched_saved_inputs':mismatch}
(P/'draft-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
