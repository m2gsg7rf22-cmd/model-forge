import importlib.util,json
from pathlib import Path
import numpy as np
P=Path('/private/tmp/modelforge-guard-aware-evaluation-review');R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
spec=importlib.util.spec_from_file_location('ind_eval_fixed',R/'examples/isolated_glu/guard_aware_evaluate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:m.metrics(np.array([[1.,1e-200]]),np.array([[1.,0.]]))
except ValueError:pass
else:raise AssertionError('underflow still accepted')
for stage in ['query','baseline','student']:
 def mutate(x):x.setflags(write=True);x[:]=7;return x.copy()
 f={'query':lambda x:x.copy(),'baseline':lambda x:x.copy(),'student':lambda x:x.copy()};f[stage]=mutate
 root=P/('fixed-mutate-'+stage)
 try:m.evaluate_batches(root,f['query'],f['student'],f['baseline'],seed=27,rows=2,dimension=2)
 except m.EvaluationError:pass
 else:raise AssertionError('mutation accepted')
 assert json.loads((root/'progress.json').read_text())['status']=='failed'
# Ordinary prediction callback can reuse its output buffer across batches.
shared=np.empty((2,2))
def reused(x):shared[:]=2*x;return shared
root=P/'fixed-reused-prediction-buffer';r=m.evaluate_batches(root,lambda x:x.copy(),reused,lambda x:3*x,seed=28,rows=2,dimension=2)
with np.load(root/'paired-validation.npz') as a,np.load(root/'group-0-student.npz') as first:
 mismatch=not np.array_equal(a['prediction'][:2],first['prediction'])
assert mismatch
result={'underflow_now_rejected':True,'all3_input_mutation_stages_rejected':True,'new_reused_prediction_buffer_gap':mismatch,'synthetic_seeds':[27,28],'actual941033_generated':False,'actual_reference_factory_calls':0}
(P/'fixed-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
